"""
Core DatabricksMCPClient integration tests for Vector Search MCP.

Tests list_tools and call_tool against a live Databricks MCP server
backed by a Vector Search index.
"""

from __future__ import annotations

import os

import pytest
from mcp.types import CallToolResult

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MCP_INTEGRATION_TESTS") != "1",
    reason="MCP integration tests disabled. Set RUN_MCP_INTEGRATION_TESTS=1 to enable.",
)

# =============================================================================
# list_tools tests
# =============================================================================


@pytest.mark.integration
class TestVSMCPClientListTools:
    """Verify list_tools() returns valid tool metadata from VS MCP server."""

    def test_list_tools_returns_nonempty(self, cached_vs_tools_list):
        assert len(cached_vs_tools_list) > 0

    def test_tools_have_name_and_schema(self, cached_vs_tools_list):
        for tool in cached_vs_tools_list:
            assert tool.name, "Tool name should be non-empty"
            assert tool.inputSchema, "Tool should have an inputSchema"
            assert "properties" in tool.inputSchema, "inputSchema should have 'properties'"


# =============================================================================
# call_tool tests
# =============================================================================


@pytest.mark.integration
class TestVSMCPClientCallTool:
    """Verify call_tool() executes VS search tools and returns results."""

    def test_call_tool_returns_result_with_content(self, cached_vs_call_result):
        assert isinstance(cached_vs_call_result, CallToolResult)
        assert cached_vs_call_result.content, "call_tool result should have content"
        assert len(cached_vs_call_result.content) > 0
