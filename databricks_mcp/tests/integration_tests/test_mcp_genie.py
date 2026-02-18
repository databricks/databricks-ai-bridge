"""
Core DatabricksMCPClient integration tests for Genie MCP.

Tests list_tools and call_tool against a live Databricks MCP server
backed by a Genie space. These test our bridge code's URL construction,
auth forwarding, and response parsing — NOT the Genie service itself
(that's covered by PR #325's Genie integration tests).
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
class TestGenieMCPClientListTools:
    """Verify list_tools() returns valid tool metadata from Genie MCP server."""

    def test_list_tools_returns_nonempty(self, cached_genie_tools_list):
        assert len(cached_genie_tools_list) > 0

    def test_tools_have_name_and_schema(self, cached_genie_tools_list):
        for tool in cached_genie_tools_list:
            assert tool.name, "Tool name should be non-empty"
            assert tool.inputSchema, "Tool should have an inputSchema"
            assert "properties" in tool.inputSchema, "inputSchema should have 'properties'"


# =============================================================================
# call_tool tests
# =============================================================================


@pytest.mark.integration
class TestGenieMCPClientCallTool:
    """Verify call_tool() executes Genie tools and returns results."""

    def test_call_tool_returns_result_with_content(self, cached_genie_call_result):
        assert isinstance(cached_genie_call_result, CallToolResult)
        assert cached_genie_call_result.content, "call_tool result should have content"
        assert len(cached_genie_call_result.content) > 0
