"""
Integration tests for get_databricks_resources() and tool name normalization.

Validates that DatabricksMCPClient correctly extracts Databricks resource
metadata from live MCP tool names.
"""

from __future__ import annotations

import os

import pytest
from mlflow.models.resources import DatabricksFunction

from databricks_mcp.mcp import DatabricksMCPClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MCP_INTEGRATION_TESTS") != "1",
    reason="MCP integration tests disabled. Set RUN_MCP_INTEGRATION_TESTS=1 to enable.",
)

# =============================================================================
# get_databricks_resources tests
# =============================================================================


@pytest.mark.integration
class TestGetDatabricksResources:
    """Verify get_databricks_resources() returns correct resource types from live tools."""

    def test_uc_function_returns_databricks_function_resources(self, mcp_client):
        resources = mcp_client.get_databricks_resources()
        assert len(resources) > 0, "Should return at least one resource"
        for resource in resources:
            assert isinstance(resource, DatabricksFunction), (
                f"Expected DatabricksFunction, got {type(resource)}"
            )

    def test_resource_names_are_dot_separated(self, mcp_client):
        resources = mcp_client.get_databricks_resources()
        assert len(resources) > 0
        for resource in resources:
            name = resource.function_name
            assert "." in name, f"Resource name should be dot-separated, got: {name}"
            assert "__" not in name, (
                f"Resource name should not contain '__', got: {name}"
            )

    def test_resource_count_matches_tools_count(self, mcp_client, cached_tools_list):
        resources = mcp_client.get_databricks_resources()
        assert len(resources) == len(cached_tools_list), (
            f"Resource count ({len(resources)}) should match tool count ({len(cached_tools_list)})"
        )


# =============================================================================
# Tool name normalization tests
# =============================================================================


@pytest.mark.integration
class TestToolNameNormalization:
    """Verify _normalize_tool_name converts real MCP tool names correctly."""

    def test_normalize_converts_real_tool_names(self, mcp_client, cached_tools_list):
        for tool in cached_tools_list:
            normalized = mcp_client._normalize_tool_name(tool.name)
            assert "." in normalized, (
                f"Normalized name should contain dots, got: {normalized}"
            )
            parts = normalized.split(".")
            assert len(parts) >= 2, (
                f"Normalized name should have at least 2 dot-separated parts, got: {normalized}"
            )
