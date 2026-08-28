"""Create LangGraph MCP tools from ``agent.toml`` at request time."""

from __future__ import annotations

import logging
from typing import Any

from agent.mason.tool_manifest import ToolRecord, downscope_wire, load_tools
from agent.mcps import build_mcp_servers
from databricks.sdk import WorkspaceClient
from databricks_langchain import (  # ty: ignore[unresolved-import]
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langchain_mcp_adapters.sessions import create_session  # ty: ignore[unresolved-import]

logger = logging.getLogger(__name__)


def _server_from_tool(tool: ToolRecord) -> DatabricksMCPServer | None:
    workspace_client = WorkspaceClient()
    host = workspace_client.config.host.rstrip("/")
    if tool.kind in {"sandbox", "mcp"}:
        return DatabricksMCPServer(
            name=tool.id,
            url=f"{host}/ai-gateway/mcp-services/{tool.service}",
            workspace_client=workspace_client,
            timeout=120.0,
        )
    if tool.kind == "uc_function":
        catalog, schema, function_name = (tool.function or "").split(".")
        return DatabricksMCPServer.from_uc_function(
            catalog=catalog,
            schema=schema,
            function_name=function_name,
            name=tool.id,
            workspace_client=workspace_client,
            timeout=120.0,
        )
    return None


def _all_servers(tools: tuple[ToolRecord, ...]) -> list[DatabricksMCPServer]:
    manifest = [server for tool in tools if (server := _server_from_tool(tool)) is not None]
    names = {server.name for server in manifest}
    for server in build_mcp_servers():
        if server.name not in names:
            manifest.append(server)
            names.add(server.name)
    return manifest


async def _sandbox_tool_interceptor(request: Any, handler: Any) -> Any:
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


async def mcp_tools() -> list:
    """Return freshly materialized LangChain MCP tools for one agent request."""
    tools = load_tools(expected_framework="langgraph")
    servers = _all_servers(tools)
    if not servers:
        return []
    interceptors = (
        [_sandbox_tool_interceptor] if any(tool.kind == "sandbox" for tool in tools) else []
    )
    try:
        return await DatabricksMultiServerMCPClient(
            servers,
            tool_interceptors=interceptors,
        ).get_tools()
    except Exception:
        logger.warning("Failed to fetch MCP tools; continuing without them.", exc_info=True)
        return []
