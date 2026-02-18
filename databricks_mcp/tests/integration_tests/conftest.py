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

VS_SCHEMA = "databricks_ai_bridge_vs_test"
VS_INDEX = "delta_sync_managed"

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
# Vector Search Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def vs_mcp_url(workspace_client):
    """Construct MCP URL for a single VS index."""
    base_url = workspace_client.config.host
    return f"{base_url}/api/2.0/mcp/vector-search/{CATALOG}/{VS_SCHEMA}/{VS_INDEX}"


@pytest.fixture(scope="session")
def vs_schema_mcp_url(workspace_client):
    """Construct MCP URL for all VS indexes in a schema."""
    base_url = workspace_client.config.host
    return f"{base_url}/api/2.0/mcp/vector-search/{CATALOG}/{VS_SCHEMA}"


@pytest.fixture(scope="session")
def vs_mcp_client(vs_mcp_url, workspace_client):
    """DatabricksMCPClient pointed at a single VS index."""
    return DatabricksMCPClient(vs_mcp_url, workspace_client)


@pytest.fixture(scope="session")
def vs_schema_mcp_client(vs_schema_mcp_url, workspace_client):
    """DatabricksMCPClient pointed at all VS indexes in a schema."""
    return DatabricksMCPClient(vs_schema_mcp_url, workspace_client)


@pytest.fixture(scope="session")
def cached_vs_tools_list(vs_mcp_client):
    """Cache the VS list_tools() result; skip if VS MCP endpoint unavailable."""
    try:
        return vs_mcp_client.list_tools()
    except Exception as e:
        pytest.skip(f"VS MCP endpoint not available: {e}")


@pytest.fixture(scope="session")
def cached_vs_call_result(vs_mcp_client, cached_vs_tools_list):
    """Cache a VS call_tool() result for the session."""
    tool = cached_vs_tools_list[0]
    return vs_mcp_client.call_tool(tool.name, {"query": "test"})


# =============================================================================
# Genie Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def genie_space_id():
    """Get the Genie Space ID from the GENIE_SPACE_ID environment variable."""
    space_id = os.environ.get("GENIE_SPACE_ID")
    if not space_id:
        pytest.skip("GENIE_SPACE_ID environment variable not set")
    return space_id


@pytest.fixture(scope="session")
def genie_mcp_url(workspace_client, genie_space_id):
    """Construct MCP URL for a Genie space."""
    base_url = workspace_client.config.host
    return f"{base_url}/api/2.0/mcp/genie/{genie_space_id}"


@pytest.fixture(scope="session")
def genie_mcp_client(genie_mcp_url, workspace_client):
    """DatabricksMCPClient pointed at a Genie space."""
    return DatabricksMCPClient(genie_mcp_url, workspace_client)


@pytest.fixture(scope="session")
def cached_genie_tools_list(genie_mcp_client):
    """Cache the Genie list_tools() result; skip if Genie MCP endpoint unavailable."""
    try:
        return genie_mcp_client.list_tools()
    except Exception as e:
        pytest.skip(f"Genie MCP endpoint not available: {e}")


@pytest.fixture(scope="session")
def cached_genie_call_result(genie_mcp_client, cached_genie_tools_list):
    """Cache a Genie call_tool() result for the session."""
    tool = cached_genie_tools_list[0]
    return genie_mcp_client.call_tool(tool.name, {"Query": "How many rows are there?"})


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
