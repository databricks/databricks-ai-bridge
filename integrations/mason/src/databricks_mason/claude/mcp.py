"""MCP servers for the agent, from the ones declared in ``agent.toml`` (plus any the agent adds).

``mcp_servers()`` reads the MCP servers declared in ``agent.toml`` (sandbox/mcp + uc_function) and
returns Anthropic MCP-connector server dicts (``{"type": "url", "name", "url", "authorization_token"}``)
to hand the Tool Runner as ``mcp_servers=`` (with beta ``mcp-client-2025-11-20`` and an ``mcp_toolset``
tool). Fail-open to ``[]`` — an empty ``agent.toml`` yields no servers, and any build error is logged
and skipped so it never fails the request.

First cut: the Databricks-hosted MCP services are wired best-effort with the caller's bearer token;
sandbox downscoping is not applied through the connector yet.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_mason.runtime.tool_manifest import (
    ToolManifestError,
    ToolRecord,
    load_tools,
)
from databricks_mason.runtime.workspace import workspace_client

logger = logging.getLogger(__name__)

_FRAMEWORK = "claude"

# The beta header the Anthropic MCP connector requires; pass it to the runner alongside mcp_servers.
MCP_CONNECTOR_BETA = "mcp-client-2025-11-20"


def _bearer_token(client: Any) -> str | None:
    try:
        auth = client.config.authenticate() or {}
    except Exception:
        return None
    return (auth.get("Authorization") or "").removeprefix("Bearer ").strip() or None


def _server_from_tool(tool: ToolRecord, client: Any, host: str, token: str | None) -> dict | None:
    if tool.kind in {"sandbox", "mcp"}:
        server = {
            "type": "url",
            "name": tool.id,
            "url": f"{host}/ai-gateway/mcp-services/{tool.service}",
        }
    elif tool.kind == "uc_function":
        server = {
            "type": "url",
            "name": tool.id,
            "url": f"{host}/api/2.0/mcp/functions/{tool.function}",
        }
    else:
        return None
    if token:
        server["authorization_token"] = token
    return server


def _declared_servers() -> list[dict]:
    tools = load_tools(expected_framework=_FRAMEWORK)
    if not tools:
        return []
    client = workspace_client()
    host = client.config.host.rstrip("/")
    token = _bearer_token(client)
    return [s for tool in tools if (s := _server_from_tool(tool, client, host, token)) is not None]


def mcp_servers(extra_servers: list[dict] | None = None) -> list[dict]:
    """Build the agent's MCP-connector server dicts. Fail-open to ``[]``.

    Includes the servers declared in ``agent.toml``; pass ``extra_servers`` to add your own. Returns
    an empty list when there are none or construction fails, so it is safe to spread into the runner.
    """
    try:
        return [*_declared_servers(), *(extra_servers or [])]
    except ToolManifestError:
        raise
    except Exception:
        logger.warning("Failed to build MCP servers; continuing without them.", exc_info=True)
        return []
