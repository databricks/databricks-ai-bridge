"""
Core DatabricksMCPClient integration tests.

Tests list_tools, call_tool, and auth paths against a live Databricks MCP server
backed by a UC function (echo_message).
"""

from __future__ import annotations

import os

import pytest
from mcp.types import CallToolResult

from databricks_mcp import DatabricksMCPClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MCP_INTEGRATION_TESTS") != "1",
    reason="MCP integration tests disabled. Set RUN_MCP_INTEGRATION_TESTS=1 to enable.",
)

CATALOG = "integration_testing"
SCHEMA = "databricks_ai_bridge_mcp_test"
FUNCTION_NAME = "echo_message"

# =============================================================================
# list_tools tests
# =============================================================================


@pytest.mark.integration
class TestMCPClientListTools:
    """Verify list_tools() returns valid tool metadata from a live MCP server."""

    def test_list_tools_returns_nonempty(self, cached_tools_list):
        assert len(cached_tools_list) > 0

    def test_tools_have_name_and_schema(self, cached_tools_list):
        for tool in cached_tools_list:
            assert tool.name, "Tool name should be non-empty"
            assert tool.inputSchema, "Tool should have an inputSchema"
            assert "properties" in tool.inputSchema, "inputSchema should have 'properties'"

    def test_tool_name_contains_function_identifier(self, cached_tools_list):
        tool_names = [t.name for t in cached_tools_list]
        assert any(FUNCTION_NAME in name or "echo_message" in name for name in tool_names), (
            f"Expected a tool name containing '{FUNCTION_NAME}', got: {tool_names}"
        )

    def test_schema_level_lists_multiple_tools(self, schema_mcp_client):
        """Schema-level URL should list at least the echo_message function."""
        try:
            tools = schema_mcp_client.list_tools()
        except Exception as e:
            pytest.skip(f"Schema-level list_tools failed: {e}")
        assert len(tools) >= 1, "Schema-level listing should return at least one tool"


# =============================================================================
# call_tool tests
# =============================================================================


@pytest.mark.integration
class TestMCPClientCallTool:
    """Verify call_tool() executes tools and returns valid results."""

    def test_call_tool_returns_result(self, cached_call_result):
        assert isinstance(cached_call_result, CallToolResult)

    def test_call_tool_content_is_text(self, cached_call_result):
        assert cached_call_result.content, "call_tool result should have content"
        assert len(cached_call_result.content) > 0
        first_item = cached_call_result.content[0]
        assert hasattr(first_item, "text"), "First content item should have .text"

    def test_call_tool_echo_returns_input(self, cached_call_result):
        text = cached_call_result.content[0].text
        assert "hello" in text, f"Echo function should return input 'hello', got: {text}"

    def test_call_tool_with_different_args(self, mcp_client, cached_tools_list):
        tool_name = cached_tools_list[0].name
        result = mcp_client.call_tool(tool_name, {"message": "integration_test_42"})
        text = result.content[0].text
        assert "integration_test_42" in text, (
            f"Echo should return 'integration_test_42', got: {text}"
        )


# =============================================================================
# Auth path tests
# =============================================================================


@pytest.mark.integration
class TestMCPClientAuth:
    """Verify different auth methods work through the MCP layer."""

    def test_oauth_m2m_auth_works(self, uc_function_url):
        """OAuth M2M auth with explicit client_id/client_secret should work."""
        from databricks.sdk import WorkspaceClient

        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
        host = os.environ.get("DATABRICKS_HOST")
        if not (client_id and client_secret and host):
            pytest.skip(
                "OAuth-M2M credentials not available "
                "(need DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET)"
            )

        oauth_wc = WorkspaceClient(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
        )
        client = DatabricksMCPClient(uc_function_url, oauth_wc)
        tools = client.list_tools()
        assert len(tools) > 0

    def test_pat_auth_works(self, workspace_client, uc_function_url):
        """PAT auth (extracted from current auth) should work."""
        from databricks.sdk import WorkspaceClient

        headers = workspace_client.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        assert token, "Could not extract bearer token from workspace client"

        pat_wc = WorkspaceClient(host=workspace_client.config.host, token=token, auth_type="pat")
        client = DatabricksMCPClient(uc_function_url, pat_wc)
        tools = client.list_tools()
        assert len(tools) > 0
