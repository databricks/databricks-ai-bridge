"""
Integration tests for OpenAI Vector Search components.

Tests VectorSearchRetrieverTool against live Databricks Vector Search indexes.

Prerequisites:
- Test indexes must be pre-created with sample data
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_VS_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set RUN_VS_INTEGRATION_TESTS=1 to enable.",
)

# Index configuration (must match root conftest)
CATALOG = "integration_testing"
SCHEMA = "databricks_ai_bridge_vs_test"
DELTA_SYNC_INDEX = f"{CATALOG}.{SCHEMA}.delta_sync_managed"


@pytest.fixture(scope="session")
def workspace_client():
    from databricks.sdk import WorkspaceClient

    wc = WorkspaceClient()
    # VectorSearchClient only supports PAT/OAuth-M2M/model-serving auth.
    # Convert other auth types (e.g. databricks-cli) to PAT so credentials
    # are forwarded correctly to the underlying VectorSearchClient.
    if wc.config.auth_type not in ("pat", "oauth-m2m", "model_serving_user_credentials"):
        headers = wc.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        if token:
            return WorkspaceClient(host=wc.config.host, token=token, auth_type="pat")
    return wc


# =============================================================================
# VectorSearchRetrieverTool Initialization Tests
# =============================================================================


@pytest.mark.integration
class TestOpenAIRetrieverToolInit:
    """Test OpenAI VectorSearchRetrieverTool initialization with live index."""

    def test_tool_init_produces_valid_tool_param(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )
        assert tool.tool is not None
        assert tool.tool["type"] == "function"
        assert "function" in tool.tool
        assert "name" in tool.tool["function"]

    def test_tool_schema_has_query_parameter(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )
        assert tool.tool is not None
        params = tool.tool["function"]["parameters"]
        assert isinstance(params, dict)
        assert "properties" in params
        properties = params["properties"]
        assert isinstance(properties, dict)
        assert "query" in properties

    def test_tool_schema_strict_removed(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )
        # strict and additionalProperties should be removed for filter compatibility
        assert tool.tool is not None
        assert "strict" not in tool.tool.get("function", {})
        assert "additionalProperties" not in tool.tool.get("function", {}).get("parameters", {})


# =============================================================================
# VectorSearchRetrieverTool Execution Tests
# =============================================================================


@pytest.mark.integration
class TestOpenAIRetrieverToolExecution:
    """Test OpenAI VectorSearchRetrieverTool execution with live index."""

    @pytest.fixture(scope="class")
    def tool(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        return VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )

    def test_execute_returns_list_of_dicts(self, tool):
        result = tool.execute(query="machine learning")
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, dict)

    def test_execute_result_has_expected_keys(self, tool):
        result = tool.execute(query="machine learning")
        assert len(result) > 0
        for item in result:
            assert "page_content" in item
            assert "metadata" in item

    def test_execute_returns_relevant_content(self, tool):
        result = tool.execute(query="machine learning artificial intelligence")
        assert len(result) > 0
        # At least one result should have non-empty content
        contents = [item["page_content"] for item in result]
        assert any(len(c) > 0 for c in contents)

    def test_execute_num_results_honored(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
            num_results=2,
        )
        result = tool.execute(query="technology")
        assert len(result) <= 2


# =============================================================================
# Filter Pass-Through Tests
# =============================================================================


@pytest.mark.integration
class TestOpenAIRetrieverWithFilter:
    """Test filter pass-through (parity with LangChain tests)."""

    def test_execute_with_static_filter(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
            filters={"category": "databricks"},
            num_results=10,
        )
        result = tool.execute(query="technology")
        assert isinstance(result, list)
        # Our test data has only 2 docs with category="databricks"
        assert len(result) > 0
        assert len(result) <= 2

    def test_execute_with_dynamic_filter(self, workspace_client):
        from databricks_ai_bridge.vector_search_retriever_tool import FilterItem

        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
            num_results=10,
        )
        result = tool.execute(
            query="technology",
            filters=[FilterItem(key="category", value="databricks")],
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 2


# =============================================================================
# Kwargs Pass-Through Tests
# =============================================================================


@pytest.mark.integration
class TestOpenAIKwargsPassThrough:
    """Verify kwargs forwarding and filtering in our execute() code."""

    def test_score_threshold_from_constructor(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
            score_threshold=0.99,  # type: ignore[unknown-argument]
        )
        result = tool.execute(query="machine learning")
        # High threshold may return fewer or no results, but should not error
        assert isinstance(result, list)

    def test_invalid_kwargs_filtered_out(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )
        # Our inspect.signature filtering should drop unknown kwargs
        result = tool.execute(
            query="machine learning",
            totally_fake_kwarg="ignored",
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_num_results_override_at_execute_time(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
            num_results=10,
        )
        result = tool.execute(query="machine learning", num_results=1)
        # Execute-time override takes precedence
        assert len(result) <= 1


# =============================================================================
# Auth Path Tests
# =============================================================================


@pytest.mark.integration
class TestOpenAIAuthPaths:
    """Verify auth credentials are correctly forwarded to VectorSearchClient."""

    def test_current_auth_produces_working_tool(self, workspace_client):
        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )
        result = tool.execute(query="machine learning")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_pat_auth_produces_working_tool(self, workspace_client):
        from databricks.sdk import WorkspaceClient

        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        # Extract a bearer token from the current auth (works for any auth type)
        headers = workspace_client.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        assert token, "Could not extract bearer token from workspace client"

        pat_wc = WorkspaceClient(host=workspace_client.config.host, token=token, auth_type="pat")
        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=pat_wc,
        )
        result = tool.execute(query="machine learning")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_oauth_m2m_auth_produces_working_tool(self):
        from databricks.sdk import WorkspaceClient

        from databricks_openai.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

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
        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=oauth_wc,
        )
        result = tool.execute(query="machine learning")
        assert isinstance(result, list)
        assert len(result) > 0
