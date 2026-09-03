"""Build MCP servers for the agent from the ones declared in ``agent.toml`` (plus any the agent adds).

``connected_mcp_servers()`` is the main entry point: it reads the MCP servers declared in
``agent.toml`` (sandbox/mcp + uc_function), connects them, and lists their tools with sandbox
downscoping applied. An unavailable server is logged and omitted so the agent can continue with the
healthy subset. An agent with its own hand-built servers passes them as ``extra_servers``; leaving
``agent.toml`` empty simply yields no declared servers. Typical use::

    async with connected_mcp_servers() as servers:  # just the agent.toml servers
        agent = Agent(..., mcp_servers=servers)

    async with connected_mcp_servers(build_mcp_servers()) as servers:
        agent = Agent(..., mcp_servers=servers)  # agent.toml servers + the agent's own

``mcp_servers()`` remains available when callers need the unconnected server objects directly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from databricks_openai.agents import McpServer

from databricks_mason.runtime.tool_manifest import ToolRecord, downscope_wire, load_tools
from databricks_mason.runtime.workspace import workspace_client

logger = logging.getLogger(__name__)

_FRAMEWORK = "openai"


class _DownscopedMcpServer(McpServer):
    """An ``McpServer`` that injects a sandbox downscope into every ``call_tool``.

    The Databricks sandbox MCP applies the downscope from the call's ``_meta``; the Agents SDK does
    not surface a per-call hook, so bind the manifest's downscope to the server and add it on each
    invocation. Only sandbox bindings need this — plain MCP / UC-function servers use the base class.
    """

    def __init__(self, *args: Any, downscope: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._downscope = downscope

    async def call_tool(self, tool_name, arguments, **kwargs):
        meta = {**(kwargs.pop("meta", None) or {}), "downscope": self._downscope}
        return await super().call_tool(tool_name, arguments, meta=meta, **kwargs)


def _server_from_tool(tool: ToolRecord) -> McpServer | None:
    client = workspace_client()
    host = client.config.host.rstrip("/")
    if tool.kind in {"sandbox", "mcp"}:
        url = f"{host}/ai-gateway/mcp-services/{tool.service}"
        if tool.kind == "sandbox":
            return _DownscopedMcpServer(
                url=url,
                name=tool.id,
                workspace_client=client,
                timeout=120.0,
                downscope=downscope_wire(tool),
            )
        return McpServer(url=url, name=tool.id, workspace_client=client, timeout=120.0)
    if tool.kind == "uc_function":
        catalog, schema, function_name = (tool.function or "").split(".")
        return McpServer.from_uc_function(
            catalog=catalog,
            schema=schema,
            function_name=function_name,
            name=tool.id,
            workspace_client=client,
            timeout=120.0,
        )
    return None


def _declared_servers() -> list[McpServer]:
    """The MCP servers declared in the agent's ``agent.toml`` (may be empty)."""
    tools = load_tools(expected_framework=_FRAMEWORK)
    return [server for tool in tools if (server := _server_from_tool(tool)) is not None]


async def mcp_servers(extra_servers: list[McpServer] | None = None) -> list[McpServer]:
    """Build unconnected MCP server objects with sandbox downscoping.

    Includes the MCP servers declared in ``agent.toml``; pass ``extra_servers`` to add servers the
    agent builds itself. Returns an empty list when there are no servers or manifest construction
    fails. Prefer :func:`connected_mcp_servers` unless the caller manages server lifecycles itself.
    """
    try:
        return [*_declared_servers(), *(extra_servers or [])]
    except Exception:
        logger.warning("Failed to build MCP servers; continuing without them.", exc_info=True)
        return []


@asynccontextmanager
async def connected_mcp_servers(
    extra_servers: list[McpServer] | None = None,
) -> AsyncIterator[list[McpServer]]:
    """Yield the connected MCP servers that can list tools, omitting unavailable servers.

    Tool lists are cached after the health check so the Agents SDK does not repeat the request when
    a run starts. Each successful connection remains open until the context exits.
    """
    async with AsyncExitStack() as active_stack:
        active_servers: list[McpServer] = []
        for server in await mcp_servers(extra_servers):
            server_stack = AsyncExitStack()
            try:
                connected_server = await server_stack.enter_async_context(server)
                connected_server.cache_tools_list = True
                await connected_server.list_tools()
            except Exception:
                logger.warning(
                    "Failed to connect to or list tools from MCP server %r; continuing without it.",
                    getattr(server, "name", "unknown"),
                    exc_info=True,
                )
                try:
                    await server_stack.aclose()
                except Exception:
                    logger.warning(
                        "Failed to clean up unavailable MCP server %r.",
                        getattr(server, "name", "unknown"),
                        exc_info=True,
                    )
                continue

            active_stack.push_async_callback(server_stack.aclose)
            active_servers.append(connected_server)

        yield active_servers
