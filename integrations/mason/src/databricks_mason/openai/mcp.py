"""Attach explicit Databricks integrations to an OpenAI Agents SDK agent."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from agents import Agent
from agents.mcp import MCPServerStreamableHttpParams
from databricks.sdk import WorkspaceClient
from databricks_openai.agents import McpServer

from databricks_mason.integrations import (
    Integration,
    MCPService,
    Sandbox,
    UCFunction,
    downscope_wire,
)
from databricks_mason.runtime.workspace import (
    workspace_client as _default_workspace_client,
)
from databricks_mason.runtime.workspace import (
    workspace_headers,
)

TContext = TypeVar("TContext")


class _SandboxMcpServer(McpServer):
    """MCP server that enforces the selected Sandbox scope on every call."""

    def __init__(self, *, sandbox: Sandbox, **kwargs: Any) -> None:
        self._sandbox = sandbox
        super().__init__(**kwargs)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Any:
        incoming_meta = kwargs.pop("meta", None)
        meta = dict(incoming_meta) if isinstance(incoming_meta, dict) else {}
        meta["downscope"] = downscope_wire(self._sandbox)
        return await super().call_tool(tool_name, arguments, meta=meta, **kwargs)


def _transport_params() -> MCPServerStreamableHttpParams | None:
    headers = workspace_headers()
    if not headers:
        return None
    return MCPServerStreamableHttpParams(url="", headers=headers)


def _server_from_integration(
    integration: Integration,
    workspace_client: WorkspaceClient,
) -> McpServer:
    host = workspace_client.config.host.rstrip("/")
    if isinstance(integration, MCPService):
        return McpServer(
            url=f"{host}/ai-gateway/mcp-services/{integration.service}",
            name=integration.id,
            workspace_client=workspace_client,
            timeout=120.0,
            params=_transport_params(),
        )
    if isinstance(integration, Sandbox):
        return _SandboxMcpServer(
            sandbox=integration,
            url=f"{host}/ai-gateway/mcp-services/system.ai.sandbox",
            name=integration.id,
            workspace_client=workspace_client,
            timeout=120.0,
            params=_transport_params(),
            tool_filter={"allowed_tool_names": ["sandbox", "run_code"]},
        )
    if isinstance(integration, UCFunction):
        catalog, schema, function_name = integration.function.split(".")
        return McpServer.from_uc_function(
            catalog=catalog,
            schema=schema,
            function_name=function_name,
            name=integration.id,
            workspace_client=workspace_client,
            timeout=120.0,
            params=_transport_params(),
        )
    raise TypeError(f"Unsupported integration: {type(integration).__name__}")


def _validate_server_names(agent: Agent[Any], integrations: Sequence[Integration]) -> None:
    owners: dict[str, str] = {}
    candidates = [
        *((server.name, "existing agent") for server in agent.mcp_servers),
        *((integration.id, "Databricks integration") for integration in integrations),
    ]
    for name, owner in candidates:
        if previous_owner := owners.get(name):
            raise ValueError(
                f"MCP server name {name!r} is used by both {previous_owner} and {owner}."
            )
        owners[name] = owner


def _claim_tool(tool_owners: dict[str, str], name: str, owner: str) -> None:
    if previous_owner := tool_owners.get(name):
        raise ValueError(
            f"MCP tool {name!r} is advertised by both {previous_owner!r} and {owner!r}."
        )
    tool_owners[name] = owner


async def bind_tools(
    agent: Agent[TContext],
    integrations: Sequence[Integration],
    *,
    stack: AsyncExitStack,
    workspace_client: WorkspaceClient | None = None,
) -> Agent[TContext]:
    """Connect selected integrations and return an isolated clone of ``agent``.

    The caller owns ``stack`` and must keep it open for as long as the returned agent can run.
    Closing the stack disconnects every server materialized by this call. Existing servers on the
    input agent are preserved, and their lifecycle remains owned by the caller that supplied them.
    Existing servers are not eagerly inspected because dynamic tool filters require request
    context; the Agents SDK discovers and validates the full MCP tool set during each run. Server
    names and newly materialized tool names are validated before cloning.
    """

    servers: list[McpServer] = []
    if integrations:
        _validate_server_names(agent, integrations)
        tool_owners = {
            name: "local agent tool"
            for tool in agent.tools
            if isinstance(name := getattr(tool, "name", None), str)
        }
        client = workspace_client if workspace_client is not None else _default_workspace_client()
        for integration in integrations:
            server = await stack.enter_async_context(_server_from_integration(integration, client))
            for tool in await server.list_tools():
                _claim_tool(tool_owners, tool.name, integration.id)
            servers.append(server)
    return agent.clone(
        tools=[*agent.tools],
        mcp_servers=[*agent.mcp_servers, *servers],
        handoffs=[*agent.handoffs],
        input_guardrails=[*agent.input_guardrails],
        output_guardrails=[*agent.output_guardrails],
    )
