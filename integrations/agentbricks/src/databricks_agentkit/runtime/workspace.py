"""Construct Databricks SDK clients with workspace routing when required."""

from __future__ import annotations

import os

from databricks.sdk import WorkspaceClient


def workspace_headers() -> dict[str, str]:
    """Return headers required to route an account-host request to its workspace."""
    workspace_id = os.getenv("DATABRICKS_WORKSPACE_ID", "").strip()
    return {"X-Databricks-Org-Id": workspace_id} if workspace_id else {}


def mcp_headers() -> dict[str, str]:
    """Return MCP headers, including opt-in test routing for MAS LiteSwap.

    ``DATABRICKS_MAS_TRAFFIC_ID`` is intentionally scoped to MCP requests. It is a test-only
    escape hatch for routing managed MCP calls to a MAS LiteSwap unit; arbitrary environment
    variables are never forwarded.
    """
    headers = workspace_headers()
    traffic_id = os.getenv("DATABRICKS_MAS_TRAFFIC_ID", "").strip()
    if traffic_id:
        headers["x-databricks-traffic-id"] = traffic_id
    return headers


def workspace_client() -> WorkspaceClient:
    """Return the environment-authenticated client for the active workspace.

    ``databricks apps run-local`` can authenticate through an account-level vanity host while
    exposing the target workspace through ``DATABRICKS_WORKSPACE_ID``. The SDK needs the same
    routing header as the Databricks CLI for those profiles. Ordinary workspace hosts and deployed
    Apps continue to use the SDK's default authentication chain.
    """
    headers = workspace_headers()
    if not headers:
        return WorkspaceClient()
    return WorkspaceClient(custom_headers=headers)
