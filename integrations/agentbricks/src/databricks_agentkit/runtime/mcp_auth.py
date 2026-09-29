"""Shared authentication helpers for framework-specific MCP adapters."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from databricks.sdk import WorkspaceClient

from databricks_agentkit.runtime.auth import AuthError

WorkspaceClientResolver = Callable[[str], WorkspaceClient]


def resolve_mcp_workspace_client(
    mode: str,
    integration_id: str,
    workspace_client_for: WorkspaceClientResolver | None,
    default_client: Callable[[], WorkspaceClient],
) -> WorkspaceClient:
    """Resolve one manifest binding to its request-user or App client.

    Request-aware templates pass ``RequestAuthContext.client_for`` as ``workspace_client_for``.
    Databricks Apps sets ``DATABRICKS_APP_NAME`` in deployed compute, so a user-auth binding there
    fails closed when invocation plumbing did not supply that request-owned resolver. Local runs
    retain the environment-authenticated client fallback.
    """
    if workspace_client_for is None and mode == "user" and os.getenv("DATABRICKS_APP_NAME"):
        raise AuthError(
            "MCP_USER_AUTHORIZATION_MISSING",
            "This deployed MCP integration requires request-user authorization.",
            integration_id=integration_id,
        )
    try:
        return workspace_client_for(mode) if workspace_client_for else default_client()
    except Exception as error:
        raise mcp_auth_error(error, integration_id) or AuthError(
            "MCP_CLIENT_CONFIGURATION_FAILED",
            "Could not configure the MCP client.",
            500,
            integration_id,
        ) from None


def mcp_auth_error(
    error: BaseException, integration_id: str, seen: set[int] | None = None
) -> AuthError | None:
    """Classify nested credential failures without exposing upstream error details."""
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
        if nested is not None and (classified := mcp_auth_error(nested, integration_id, seen)):
            return classified
    status = getattr(error, "status_code", None) or getattr(
        getattr(error, "response", None), "status_code", None
    )
    code = getattr(getattr(error, "error", None), "code", None)
    if isinstance(error, (PermissionError, PermissionDenied)) or status == 403:
        return AuthError("MCP_PERMISSION_DENIED", "MCP permission denied.", 403, integration_id)
    if isinstance(error, Unauthenticated) or status == 401:
        return AuthError(
            "MCP_USER_AUTHORIZATION_INVALID",
            "MCP authorization was rejected.",
            401,
            integration_id,
        )
    if code == -32042:
        return AuthError(
            "MCP_AUTHORIZATION_REQUIRED",
            "Authorize the configured service in Databricks before retrying.",
            401,
            integration_id,
        )
    return None


def mcp_tool_error(result: Any, integration_id: str) -> AuthError | None:
    """Classify structured MCP authorization failures returned as tool results."""
    structured = getattr(result, "structuredContent", None)
    detail = structured.get("error", structured) if isinstance(structured, dict) else {}
    code = detail.get("code") or detail.get("error_code") if isinstance(detail, dict) else None
    if code in (403, "PERMISSION_DENIED", "MCP_PERMISSION_DENIED"):
        return AuthError("MCP_PERMISSION_DENIED", "MCP permission denied.", 403, integration_id)
    if code in (401, "UNAUTHENTICATED", "MCP_USER_AUTHORIZATION_INVALID"):
        return AuthError(
            "MCP_USER_AUTHORIZATION_INVALID",
            "MCP authorization was rejected.",
            401,
            integration_id,
        )
    if code == -32042:
        return AuthError(
            "MCP_AUTHORIZATION_REQUIRED",
            "Authorize the configured service in Databricks.",
            401,
            integration_id,
        )
    return None
