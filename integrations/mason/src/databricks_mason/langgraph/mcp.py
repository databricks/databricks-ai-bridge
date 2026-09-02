"""Build MCP tools for the agent from the servers declared in ``agent.toml`` (plus any the agent adds).

``mcp_tools()`` is the entry point: it reads the MCP servers declared in ``agent.toml``
(sandbox/mcp + uc_function), fetches their LangChain tools with sandbox downscoping applied, and
returns them. An agent with its own hand-built servers passes them as ``extra_servers``; leaving
``agent.toml`` empty simply yields no declared servers. Typical agent use::

    tools = await mcp_tools()  # just the agent.toml servers
    tools = await mcp_tools(build_mcp_servers())  # agent.toml servers + the agent's own
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from databricks_langchain import DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain_mcp_adapters.sessions import create_session

if TYPE_CHECKING:
    from databricks_langchain import MCPServer

from databricks_mason.runtime.tool_manifest import ToolRecord, downscope_wire, load_tools
from databricks_mason.runtime.workspace import workspace_client, workspace_headers

logger = logging.getLogger(__name__)


def _server_from_tool(tool: ToolRecord) -> DatabricksMCPServer | None:
    client = workspace_client()
    host = client.config.host.rstrip("/")
    if tool.kind in {"sandbox", "mcp"}:
        return DatabricksMCPServer(
            name=tool.id,
            url=f"{host}/ai-gateway/mcp-services/{tool.service}",
            headers=workspace_headers() or None,
            workspace_client=client,
            timeout=120.0,
        )
    if tool.kind == "uc_function":
        catalog, schema, function_name = (tool.function or "").split(".")
        return DatabricksMCPServer.from_uc_function(
            catalog=catalog,
            schema=schema,
            function_name=function_name,
            name=tool.id,
            headers=workspace_headers() or None,
            workspace_client=client,
            timeout=120.0,
        )
    return None


def _declared_servers() -> list[DatabricksMCPServer]:
    """The MCP servers declared in the agent's ``agent.toml`` (may be empty)."""
    tools = load_tools(expected_framework="langgraph")
    return [server for tool in tools if (server := _server_from_tool(tool)) is not None]


def _sandbox_interceptor():
    async def interceptor(request: Any, handler: Any) -> Any:
        tools = {tool.id: tool for tool in load_tools(expected_framework="langgraph")}
        tool = tools.get(request.server_name)
        if tool is None or tool.kind != "sandbox":
            return await handler(request)

        server = _server_from_tool(tool)
        if server is None:
            raise RuntimeError(f"Could not build sandbox MCP server {tool.id!r}.")
        async with create_session(server.to_connection_dict()) as session:
            await session.initialize()
            return await session.call_tool(
                request.name,
                request.args,
                meta={"downscope": downscope_wire(tool)},
            )

    return interceptor


def _has_sandbox_tool() -> bool:
    return any(tool.kind == "sandbox" for tool in load_tools(expected_framework="langgraph"))


def mcp_client(servers: list[DatabricksMCPServer]) -> DatabricksMultiServerMCPClient:
    """A multi-server MCP client over ``servers`` with the sandbox downscoping interceptor attached.

    The interceptor is derived from the ``agent.toml`` manifest, so sandbox tools run downscoped
    regardless of how the caller drives the returned client (``get_tools`` or otherwise). Callers who
    build their own client instead take on applying downscoping themselves.
    """
    interceptors = [_sandbox_interceptor()] if _has_sandbox_tool() else []
    # DatabricksMCPServer is a subclass of MCPServer, so coerce the type for the API
    servers_as_mcp: list[MCPServer] = servers  # type: ignore[name-defined,assignment]
    return DatabricksMultiServerMCPClient(servers_as_mcp, tool_interceptors=interceptors)


async def mcp_tools(extra_servers: list[DatabricksMCPServer] | None = None) -> list:
    """Fetch LangChain MCP tools for the agent (with sandbox downscoping). Fail-open to ``[]``.

    Includes the MCP servers declared in ``agent.toml``; pass ``extra_servers`` to add servers the
    agent builds itself. Returns an empty list when there are no servers or the fetch fails, so it is
    safe to spread straight into an agent's tool list.
    """
    servers = [*_declared_servers(), *(extra_servers or [])]
    if not servers:
        return []
    try:
        return await mcp_client(servers).get_tools()
    except Exception:
        logger.warning("Failed to fetch MCP tools; continuing without them.", exc_info=True)
        return []
