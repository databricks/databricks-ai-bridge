"""
Integration tests for get_databricks_resources() and tool name normalization.

Validates that DatabricksMCPClient correctly extracts Databricks resource
metadata from live MCP tool names.
"""

from __future__ import annotations

import os

import pytest
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksVectorSearchIndex,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MCP_INTEGRATION_TESTS") != "1",
    reason="MCP integration tests disabled. Set RUN_MCP_INTEGRATION_TESTS=1 to enable.",
)


@pytest.mark.integration
class TestGetDatabricksResources:
    """Verify get_databricks_resources() returns correct resource types from live tools."""

    # -- UC Functions --

    def test_uc_function_resources_have_correct_type_and_names(self, schema_mcp_client):
        """UC resources should be DatabricksFunction with dot-separated names."""
        resources = schema_mcp_client.get_databricks_resources()
        assert len(resources) > 0, "Should return at least one resource"
        for resource in resources:
            assert isinstance(resource, DatabricksFunction), (
                f"Expected DatabricksFunction, got {type(resource)}"
            )
            assert "." in resource.name, (
                f"Resource name should be dot-separated, got: {resource.name}"
            )
            assert "__" not in resource.name, (
                f"Resource name should not contain '__', got: {resource.name}"
            )

    def test_uc_resource_count_matches_schema_tools(self, schema_mcp_client):
        resources = schema_mcp_client.get_databricks_resources()
        tools = schema_mcp_client.list_tools()
        assert len(resources) == len(tools), (
            f"Resource count ({len(resources)}) should match tool count ({len(tools)})"
        )

    # -- Vector Search --

    def test_vs_resources_have_correct_type_and_names(self, vs_schema_mcp_client):
        """VS resources should be DatabricksVectorSearchIndex with dot-separated names."""
        resources = vs_schema_mcp_client.get_databricks_resources()
        assert len(resources) > 0, "Should return at least one resource"
        for resource in resources:
            assert isinstance(resource, DatabricksVectorSearchIndex), (
                f"Expected DatabricksVectorSearchIndex, got {type(resource)}"
            )
            assert "." in resource.name, (
                f"Resource name should be dot-separated, got: {resource.name}"
            )
            assert "__" not in resource.name, (
                f"Resource name should not contain '__', got: {resource.name}"
            )

    def test_vs_resource_count_matches_schema_tools(self, vs_schema_mcp_client):
        resources = vs_schema_mcp_client.get_databricks_resources()
        tools = vs_schema_mcp_client.list_tools()
        assert len(resources) == len(tools), (
            f"Resource count ({len(resources)}) should match tool count ({len(tools)})"
        )

    # -- Genie --

    def test_genie_returns_genie_space_resource(self, genie_mcp_client):
        resources = genie_mcp_client.get_databricks_resources()
        assert len(resources) == 1, "Genie should return exactly one resource"
        assert isinstance(resources[0], DatabricksGenieSpace), (
            f"Expected DatabricksGenieSpace, got {type(resources[0])}"
        )

    def test_genie_resource_has_correct_space_id(self, genie_mcp_client, genie_space_id):
        resources = genie_mcp_client.get_databricks_resources()
        assert resources[0].name == genie_space_id, (
            f"Expected space ID '{genie_space_id}', got '{resources[0].name}'"
        )
