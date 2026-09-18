"""Build MCP tools for the agent from the servers declared in ``agent.toml`` (plus any the agent adds).

``mcp_tools()`` is the entry point: it reads the MCP servers declared in ``agent.toml``
(sandbox/mcp + uc_function), fetches their LangChain tools with sandbox downscoping applied, and
returns them. An agent with its own hand-built servers passes them as ``extra_servers``; leaving
``agent.toml`` empty simply yields no declared servers. Typical agent use::

    tools = await mcp_tools()  # just the agent.toml servers
    tools = await mcp_tools(build_mcp_servers())  # agent.toml servers + the agent's own
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from databricks_langchain import DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain_mcp_adapters.sessions import create_session

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks_langchain import MCPServer

from databricks_mason.runtime.auth import AuthError
from databricks_mason.runtime.tool_manifest import ToolRecord, downscope_wire, load_tools
from databricks_mason.runtime.workspace import workspace_client, workspace_headers

logger = logging.getLogger(__name__)


def _server_from_tool(
    tool: ToolRecord,
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> DatabricksMCPServer | None:
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
        return DatabricksMCPServer(
            name=tool.id,
            url=url,
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


def _declared_servers(
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> list[DatabricksMCPServer]:
    """The MCP servers declared in the agent's ``agent.toml`` (may be empty)."""
    tools = load_tools(expected_framework="langgraph")
    return [
        server
        for tool in tools
        if (server := _server_from_tool(tool, workspace_client_for=workspace_client_for))
        is not None
    ]


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


def _sandbox_interceptor(
    tools: tuple[ToolRecord, ...],
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
):
    declared = {tool.id: tool for tool in tools}

    async def interceptor(request: Any, handler: Any) -> Any:
        tool = declared.get(request.server_name)
        if tool is None:
            return await handler(request)
        try:
            if tool.kind == "sandbox":
                server = _server_from_tool(tool, workspace_client_for=workspace_client_for)
                if server is None:
                    raise RuntimeError(f"Could not build sandbox MCP server {tool.id!r}.")
                async with create_session(server.to_connection_dict()) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        request.name,
                        request.args,
                        meta={"downscope": downscope_wire(tool)},
                    )
            else:
                result = await handler(request)
        except Exception as error:
            raise _auth_error(error, tool.id) or AuthError(
                "MCP_TOOL_FAILED", "The configured MCP tool failed.", 502, tool.id
            ) from None
        if getattr(result, "isError", False) and (error := _tool_error(result, tool.id)):
            raise error
        return result

    return interceptor


def mcp_client(
    servers: list[DatabricksMCPServer],
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
    tools: tuple[ToolRecord, ...] | None = None,
) -> DatabricksMultiServerMCPClient:
    """A multi-server MCP client over ``servers`` with the sandbox downscoping interceptor attached.

    The interceptor is derived from the ``agent.toml`` manifest, so sandbox tools run downscoped
    regardless of how the caller drives the returned client (``get_tools`` or otherwise). Callers who
    build their own client instead take on applying downscoping themselves.
    """
    snapshot = tuple(load_tools(expected_framework="langgraph")) if tools is None else tools
    interceptors = (
        [_sandbox_interceptor(snapshot, workspace_client_for=workspace_client_for)]
        if snapshot
        else []
    )
    # DatabricksMCPServer is a subclass of MCPServer, so coerce the type for the API
    servers_as_mcp: list[MCPServer] = servers  # type: ignore[name-defined,assignment]
    return DatabricksMultiServerMCPClient(
        servers_as_mcp, tool_interceptors=interceptors, handle_tool_errors=True
    )


async def mcp_tools(
    extra_servers: list[DatabricksMCPServer] | None = None,
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> list:
    """Fetch declared tools with request identity and protected sandbox downscoping.

    Includes the MCP servers declared in ``agent.toml``; pass ``extra_servers`` to add servers the
    agent builds itself. Explicit-auth failures propagate; legacy and optional customer servers
    retain their existing identity and per-server best-effort discovery behavior.
    """
    snapshot = tuple(load_tools(expected_framework="langgraph"))
    servers = [
        server
        for tool in snapshot
        if (server := _server_from_tool(tool, workspace_client_for=workspace_client_for))
        is not None
    ]
    result = []
    if servers:
        client = mcp_client(servers, workspace_client_for=workspace_client_for, tools=snapshot)
        explicit = {tool.id for tool in snapshot if tool.auth is not None}

        async def fetch_declared(server: DatabricksMCPServer) -> list:
            try:
                return await client.get_tools(server_name=server.name)
            except Exception as error:
                if server.name in explicit:
                    raise _auth_error(error, server.name) or AuthError(
                        "MCP_TOOL_FAILED",
                        "Could not discover configured MCP tools.",
                        502,
                        server.name,
                    ) from None
                logger.warning(
                    "Failed to fetch MCP tools from server %r; continuing without it.", server.name
                )
                return []

        groups = await asyncio.gather(
            *(fetch_declared(server) for server in servers), return_exceptions=True
        )
        for group in groups:
            if isinstance(group, BaseException):
                raise group
            result.extend(group)
    if extra_servers:
        optional_client = mcp_client(extra_servers, tools=())

        async def fetch_optional(server: DatabricksMCPServer) -> list:
            try:
                return await optional_client.get_tools(server_name=server.name)
            except Exception:
                logger.warning(
                    "Failed to fetch optional MCP tools from server %r; continuing without it.",
                    server.name,
                )
                return []

        for group in await asyncio.gather(*(fetch_optional(server) for server in extra_servers)):
            result.extend(group)
    return result
