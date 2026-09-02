"""Materialize explicit Databricks integration specs as native LangChain tools."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from databricks_langchain import DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain_mcp_adapters.sessions import create_session

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks_langchain import MCPServer

from databricks_mason.integrations import (
    Integration,
    MCPService,
    Sandbox,
    downscope_wire,
)
from databricks_mason.runtime.workspace import workspace_client as _default_workspace_client
from databricks_mason.runtime.workspace import workspace_headers


def _server_from_integration(
    integration: Integration,
    client: WorkspaceClient,
) -> DatabricksMCPServer:
    host = client.config.host.rstrip("/")
    if isinstance(integration, (Sandbox, MCPService)):
        service = "system.ai.sandbox" if isinstance(integration, Sandbox) else integration.service
        return DatabricksMCPServer(
            name=integration.id,
            url=f"{host}/ai-gateway/mcp-services/{service}",
            headers=workspace_headers() or None,
            workspace_client=client,
            timeout=120.0,
        )
    catalog, schema, function_name = integration.function.split(".")
    return DatabricksMCPServer.from_uc_function(
        catalog=catalog,
        schema=schema,
        function_name=function_name,
        name=integration.id,
        headers=workspace_headers() or None,
        workspace_client=client,
        timeout=120.0,
    )


def _sandbox_interceptor(
    sandboxes: dict[str, tuple[Sandbox, DatabricksMCPServer]],
):
    async def interceptor(request: Any, handler: Any) -> Any:
        binding = sandboxes.get(request.server_name)
        if binding is None:
            return await handler(request)

        sandbox, server = binding
        async with create_session(server.to_connection_dict()) as session:
            await session.initialize()
            return await session.call_tool(
                request.name,
                request.args,
                meta={"downscope": downscope_wire(sandbox)},
            )

    return interceptor


def mcp_client(
    servers: Sequence[DatabricksMCPServer],
    *,
    sandboxes: dict[str, tuple[Sandbox, DatabricksMCPServer]] | None = None,
) -> DatabricksMultiServerMCPClient:
    """Build a native client whose Sandbox policy closes over the explicit selection."""

    server_list = list(servers)
    names = [server.name for server in server_list]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        rendered = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"MCP server names must be unique; duplicates: {rendered}.")
    for name, (sandbox, server) in (sandboxes or {}).items():
        if sandbox.id != name or server.name != name or server not in server_list:
            raise ValueError(f"Sandbox binding {name!r} does not match its MCP server.")
    interceptors = [_sandbox_interceptor(sandboxes)] if sandboxes else []
    # DatabricksMCPServer is a subclass of MCPServer, so coerce the type for the API
    servers_as_mcp = cast("list[MCPServer]", server_list)
    return DatabricksMultiServerMCPClient(servers_as_mcp, tool_interceptors=interceptors)


async def load_tools(
    integrations: Sequence[Integration],
    *,
    extra_servers: Sequence[DatabricksMCPServer] = (),
    workspace_client: WorkspaceClient | None = None,
    existing_tools: Sequence[Any] = (),
) -> list:
    """Resolve only ``integrations`` and return their native LangChain tools."""

    selected = tuple(integrations)
    supplied_servers = tuple(extra_servers)
    names = [item.id for item in selected]
    names.extend(server.name for server in supplied_servers)
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        rendered = ", ".join(repr(name) for name in duplicates)
        raise ValueError(
            f"Integration and MCP server names must be unique; duplicates: {rendered}."
        )

    declared_servers: list[DatabricksMCPServer] = []
    sandbox_bindings: dict[str, tuple[Sandbox, DatabricksMCPServer]] = {}
    if selected:
        client = workspace_client or _default_workspace_client()
        for item in selected:
            server = _server_from_integration(item, client)
            declared_servers.append(server)
            if isinstance(item, Sandbox):
                sandbox_bindings[item.id] = (item, server)
    servers = [*declared_servers, *supplied_servers]
    if not servers:
        return []
    tools = await mcp_client(servers, sandboxes=sandbox_bindings).get_tools()
    tool_names = [
        name
        for tool in (*existing_tools, *tools)
        if isinstance(name := getattr(tool, "name", None), str)
    ]
    duplicates = sorted(name for name, count in Counter(tool_names).items() if count > 1)
    if duplicates:
        rendered = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"LangGraph MCP tool names must be unique; duplicates: {rendered}.")
    return tools


async def mcp_tools(extra_servers: list[DatabricksMCPServer] | None = None) -> list:
    """Fail loudly for the retired manifest-backed API instead of dropping integrations."""

    del extra_servers
    raise RuntimeError(
        "mcp_tools() no longer discovers tools from agent.toml; migrate the selected "
        "integrations to DATABRICKS_TOOLS and call "
        "load_tools(DATABRICKS_TOOLS, extra_servers=build_mcp_servers())."
    )
