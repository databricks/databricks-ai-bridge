"""
Shared fixtures for MCP integration tests.

These tests require a live Databricks workspace with a pre-created UC function.
They are NOT run by default — set RUN_MCP_INTEGRATION_TESTS=1 to enable.

Prerequisites:
    Run setup_workspace.py once to create the test UC function.

Environment Variables:
======================
Required:
    RUN_MCP_INTEGRATION_TESTS  - Set to "1" to enable MCP integration tests
    DATABRICKS_HOST            - Workspace URL
    DATABRICKS_CLIENT_ID       - Service principal client ID
    DATABRICKS_CLIENT_SECRET   - Service principal client secret
"""

from __future__ import annotations

import os

import pytest

from databricks_mcp import DatabricksMCPClient

# =============================================================================
# Env var gate — skip entire module if not enabled
# =============================================================================

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MCP_INTEGRATION_TESTS") != "1",
    reason="MCP integration tests disabled. Set RUN_MCP_INTEGRATION_TESTS=1 to enable.",
)

# =============================================================================
# Constants (must match setup_workspace.py)
# =============================================================================

CATALOG = "integration_testing"
SCHEMA = "databricks_ai_bridge_mcp_test"
FUNCTION_NAME = "echo_message"

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def workspace_client():
    """
    Create a WorkspaceClient using environment variables.

    The SDK auto-detects auth from env vars (e.g. DATABRICKS_HOST,
    DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET for OAuth M2M).
    """
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


@pytest.fixture(scope="session")
def uc_function_url(workspace_client):
    """Construct MCP URL for the single test UC function."""
    base_url = workspace_client.config.host
    return f"{base_url}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}/{FUNCTION_NAME}"


@pytest.fixture(scope="session")
def uc_schema_url(workspace_client):
    """Construct MCP URL for the full test schema (all functions)."""
    base_url = workspace_client.config.host
    return f"{base_url}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"


@pytest.fixture(scope="session")
def mcp_client(uc_function_url, workspace_client):
    """DatabricksMCPClient pointed at the single test UC function."""
    return DatabricksMCPClient(uc_function_url, workspace_client)


@pytest.fixture(scope="session")
def schema_mcp_client(uc_schema_url, workspace_client):
    """DatabricksMCPClient pointed at the full test schema."""
    return DatabricksMCPClient(uc_schema_url, workspace_client)


@pytest.fixture(scope="session")
def cached_tools_list(mcp_client):
    """
    Cache the list_tools() result for the session to minimize API calls.

    Skips all dependent tests if the function doesn't exist.
    """
    try:
        tools = mcp_client.list_tools()
    except Exception as e:
        pytest.skip(
            f"Could not list tools from MCP server — is the test function set up? "
            f"Run setup_workspace.py first. Error: {e}"
        )
    return tools


@pytest.fixture(scope="session")
def cached_call_result(mcp_client, cached_tools_list):
    """
    Cache a call_tool() result for the session.

    Uses the first tool from cached_tools_list.
    """
    tool_name = cached_tools_list[0].name
    return mcp_client.call_tool(tool_name, {"message": "hello"})


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
