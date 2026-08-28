"""Create OpenAI Agents SDK MCP objects from ``agent.toml`` at request time."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from agent.mason.tool_manifest import ToolRecord, downscope_wire, load_tools
from agent.mcps import build_mcp_servers
from agents.mcp import MCPServer, MCPServerManager  # ty: ignore[unresolved-import]
from databricks.sdk import WorkspaceClient
from databricks_openai.agents import McpServer  # ty: ignore[unresolved-import]


class _SandboxMcpServer(McpServer):
    def __init__(self, *, downscope: dict[str, list[dict[str, str]]], **kwargs: Any):
        self._mason_downscope = downscope
        super().__init__(**kwargs)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Any:
        incoming_meta = kwargs.pop("meta", None)
        meta = dict(incoming_meta) if isinstance(incoming_meta, dict) else {}
        meta["downscope"] = self._mason_downscope
        return await super().call_tool(tool_name, arguments, meta=meta, **kwargs)


def _server_from_tool(tool: ToolRecord) -> MCPServer | None:
    workspace_client = WorkspaceClient()
    host = workspace_client.config.host.rstrip("/")
    if tool.kind == "sandbox":
        return _SandboxMcpServer(
            url=f"{host}/ai-gateway/mcp-services/system.ai.sandbox",
            workspace_client=workspace_client,
            timeout=120.0,
            name=tool.id,
            tool_filter={"allowed_tool_names": ["sandbox", "run_code"]},
            downscope=downscope_wire(tool),
        )
    if tool.kind == "mcp":
        return McpServer(
            url=f"{host}/ai-gateway/mcp-services/{tool.service}",
            workspace_client=workspace_client,
            timeout=120.0,
            name=tool.id,
        )
    if tool.kind == "uc_function":
        catalog, schema, function_name = (tool.function or "").split(".")
        return McpServer.from_uc_function(
            catalog=catalog,
            schema=schema,
            function_name=function_name,
            workspace_client=workspace_client,
            timeout=120.0,
            name=tool.id,
        )
    return None


def _all_servers() -> list[MCPServer]:
    manifest = [
        server
        for tool in load_tools(expected_framework="openai")
        if (server := _server_from_tool(tool)) is not None
    ]
    names = {server.name for server in manifest}
    for server in build_mcp_servers():
        if server.name not in names:
            manifest.append(server)
            names.add(server.name)
    return manifest


async def connect(stack: AsyncExitStack) -> list[MCPServer]:
    """Open freshly materialized MCP servers for one agent request."""
    servers = _all_servers()
    if not servers:
        return []
    manager = await stack.enter_async_context(MCPServerManager(servers))
    return manager.active_servers
