"""Build MCP servers for the agent from the ones declared in ``agent.toml`` (plus any the agent adds).

``mcp_servers()`` is the entry point: it reads the MCP servers declared in ``agent.toml``
(sandbox/mcp + uc_function) and returns Agents SDK ``McpServer`` objects, with sandbox downscoping
applied. Hand them to ``Agent(mcp_servers=...)``; the Agents SDK connects and lists their tools
lazily inside ``Runner.run``. An agent with its own hand-built servers passes them as
``extra_servers``; leaving ``agent.toml`` empty simply yields no declared servers. Typical use::

    servers = await mcp_servers()  # just the agent.toml servers
    servers = await mcp_servers(build_mcp_servers())  # agent.toml servers + the agent's own

Unlike a fetch-once tool list, these are connection objects: open them for the life of the request
(e.g. via ``AsyncExitStack``), because the SDK lists each server's tools only when the run needs them.
"""

from __future__ import annotations

import logging
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
    """Build the agent's MCP servers (with sandbox downscoping). Fail-open to ``[]``.

    Includes the MCP servers declared in ``agent.toml``; pass ``extra_servers`` to add servers the
    agent builds itself. Returns an empty list when there are no servers or construction fails, so it
    is safe to spread straight into ``Agent(mcp_servers=...)``. The SDK connects and lists each
    server's tools lazily during the run — health-check them at connect time if one bad server must
    not fail the whole request.
    """
    try:
        return [*_declared_servers(), *(extra_servers or [])]
    except Exception:
        logger.warning("Failed to build MCP servers; continuing without them.", exc_info=True)
        return []
