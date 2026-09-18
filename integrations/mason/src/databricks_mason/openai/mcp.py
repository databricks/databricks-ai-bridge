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

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from databricks_openai.agents import McpServer

from databricks_mason.runtime.auth import AuthError
from databricks_mason.runtime.tool_manifest import (
    ToolRecord,
    downscope_wire,
    load_tools,
)
from databricks_mason.runtime.workspace import workspace_client, workspace_headers

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

_FRAMEWORK = "openai"


def _auth_error(
    error: BaseException, integration_id: str, seen: set[int] | None = None
) -> AuthError | None:
    from databricks.sdk.errors import PermissionDenied, Unauthenticated

    seen = set() if seen is None else seen
    if id(error) in seen:
        return None
    seen.add(id(error))
    if isinstance(error, AuthError):
        return (
            error
            if error.integration_id
            else AuthError(error.code, str(error), error.status_code, integration_id)
        )
    nested_errors = [
        *getattr(error, "exceptions", ()),
        error.__cause__,
        error.__context__,
    ]
    for nested in nested_errors:
        if nested is not None and (classified := _auth_error(nested, integration_id, seen)):
            return classified
    status = getattr(error, "status_code", None) or getattr(
        getattr(error, "response", None), "status_code", None
    )
    code = getattr(getattr(error, "error", None), "code", None)
    if isinstance(error, (PermissionError, PermissionDenied)) or status == 403:
        return AuthError("MCP_PERMISSION_DENIED", "MCP permission denied.", 403, integration_id)
    if isinstance(error, Unauthenticated) or status == 401:
        return AuthError(
            "MCP_USER_AUTHORIZATION_INVALID", "MCP authorization was rejected.", 401, integration_id
        )
    if code == -32042:
        return AuthError(
            "MCP_AUTHORIZATION_REQUIRED",
            "Authorize the configured service in Databricks before retrying.",
            401,
            integration_id,
        )
    return None


def _tool_error(result: Any, integration_id: str) -> AuthError | None:
    structured = getattr(result, "structuredContent", None)
    detail = structured.get("error", structured) if isinstance(structured, dict) else {}
    code = detail.get("code") or detail.get("error_code") if isinstance(detail, dict) else None
    if code in (403, "PERMISSION_DENIED", "MCP_PERMISSION_DENIED"):
        return AuthError("MCP_PERMISSION_DENIED", "MCP permission denied.", 403, integration_id)
    if code in (401, "UNAUTHENTICATED", "MCP_USER_AUTHORIZATION_INVALID"):
        return AuthError(
            "MCP_USER_AUTHORIZATION_INVALID", "MCP authorization was rejected.", 401, integration_id
        )
    if code == -32042:
        return AuthError(
            "MCP_AUTHORIZATION_REQUIRED",
            "Authorize the configured service in Databricks.",
            401,
            integration_id,
        )
    return None


class _ConfiguredMcpServer(McpServer):
    """Keep configured auth failures out of SDK best-effort discovery and tool results."""

    _mason_configured = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, failure_error_function=self._raise_tool_error, **kwargs)

    def _raise_tool_error(self, context: Any, error: Exception) -> str:
        raise _auth_error(error, self.name) or AuthError(
            "MCP_TOOL_FAILED", "The configured MCP tool failed.", 502, self.name
        ) from None

    async def connect(self):
        try:
            return await super().connect()
        except Exception as error:
            raise _auth_error(error, self.name) or AuthError(
                "MCP_TOOL_FAILED",
                "Could not connect to the configured MCP service.",
                502,
                self.name,
            ) from None

    async def list_tools(self, *args, **kwargs):
        try:
            return await super().list_tools(*args, **kwargs)
        except Exception as error:
            raise _auth_error(error, self.name) or AuthError(
                "MCP_TOOL_FAILED", "Could not discover configured MCP tools.", 502, self.name
            ) from None

    async def call_tool(self, tool_name, arguments, **kwargs):
        try:
            call = getattr(McpServer.call_tool, "__wrapped__", McpServer.call_tool)
            result = await call(self, tool_name, arguments, **kwargs)
        except Exception as error:
            raise _auth_error(error, self.name) or AuthError(
                "MCP_TOOL_FAILED", "The configured MCP tool failed.", 502, self.name
            ) from None
        if getattr(result, "isError", False) and (error := _tool_error(result, self.name)):
            raise error
        return result


class _DownscopedMcpServer(_ConfiguredMcpServer):
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


def _server_from_tool(
    tool: ToolRecord,
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> McpServer | None:
    if tool.kind not in {"sandbox", "mcp", "uc_function", "genie_one"}:
        return None
    mode = tool.auth or "app"
    if workspace_client_for is None and mode == "user" and os.getenv("DATABRICKS_APP_NAME"):
        raise AuthError(
            "MCP_USER_AUTHORIZATION_MISSING",
            "This deployed MCP integration requires request-user authorization.",
            integration_id=tool.id,
        )
    try:
        client = workspace_client_for(mode) if workspace_client_for else workspace_client()
    except Exception as error:
        raise _auth_error(error, tool.id) or AuthError(
            "MCP_CLIENT_CONFIGURATION_FAILED", "Could not configure the MCP client.", 500, tool.id
        ) from None
    host = client.config.host.rstrip("/")
    if tool.kind in {"sandbox", "mcp", "genie_one"}:
        url = (
            f"{host}/api/2.0/mcp/genie"
            if tool.kind == "genie_one"
            else f"{host}/ai-gateway/mcp-services/{tool.service}"
        )
        if tool.kind == "sandbox":
            return _DownscopedMcpServer(
                url=url,
                name=tool.id,
                workspace_client=client,
                timeout=120.0,
                downscope=downscope_wire(tool),
            )
        if tool.kind == "genie_one":
            return _ConfiguredMcpServer(
                name=tool.id,
                workspace_client=client,
                timeout=120.0,
                params={"url": url, "headers": workspace_headers()},
            )
        return _ConfiguredMcpServer(url=url, name=tool.id, workspace_client=client, timeout=120.0)
    if tool.kind == "uc_function":
        catalog, schema, function_name = (tool.function or "").split(".")
        return _ConfiguredMcpServer.from_uc_function(
            catalog=catalog,
            schema=schema,
            function_name=function_name,
            name=tool.id,
            workspace_client=client,
            timeout=120.0,
        )
    return None


def _declared_servers(
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> list[McpServer]:
    """The MCP servers declared in the agent's ``agent.toml`` (may be empty)."""
    tools = load_tools(expected_framework=_FRAMEWORK)
    return [
        server
        for tool in tools
        if (server := _server_from_tool(tool, workspace_client_for=workspace_client_for))
        is not None
    ]


async def mcp_servers(
    extra_servers: list[McpServer] | None = None,
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> list[McpServer]:
    """Build the agent's configured servers with request identity and sandbox downscoping.

    Includes the MCP servers declared in ``agent.toml``; pass ``extra_servers`` to add servers the
    agent builds itself. Configured failures propagate. Extra servers keep their original identity;
    the request resolver is never forwarded to them.
    """
    return [
        *_declared_servers(workspace_client_for=workspace_client_for),
        *(extra_servers or []),
    ]
