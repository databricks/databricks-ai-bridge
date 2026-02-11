"""
Integration tests for LangChain Vector Search components.

Tests VectorSearchRetrieverTool and DatabricksVectorSearch against live
Databricks Vector Search indexes.

Prerequisites:
- Test indexes must be pre-created with sample data
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.",
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
class TestLangChainRetrieverToolInit:
    """Test VectorSearchRetrieverTool initialization with live index."""

    def test_tool_init_with_delta_sync_managed(self, workspace_client):
        from langchain_core.tools import BaseTool

        from databricks_langchain.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )
        assert isinstance(tool, BaseTool)
        assert tool.name != ""
        assert tool.description != ""

    def test_tool_init_with_custom_name_and_description(self, workspace_client):
        from databricks_langchain.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
            tool_name="my_search_tool",
            tool_description="A custom search tool for testing",
        )
        assert tool.name == "my_search_tool"
        assert tool.description == "A custom search tool for testing"

    def test_tool_resources_populated(self, workspace_client):
        from mlflow.models.resources import DatabricksVectorSearchIndex

        from databricks_langchain.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )
        assert tool.resources is not None
        assert len(tool.resources) > 0
        # Should include the vector search index resource
        vs_resources = [r for r in tool.resources if isinstance(r, DatabricksVectorSearchIndex)]
        assert len(vs_resources) > 0


# =============================================================================
# VectorSearchRetrieverTool Execution Tests
# =============================================================================


@pytest.mark.integration
class TestLangChainRetrieverToolExecution:
    """Test VectorSearchRetrieverTool execution with live index."""

    @pytest.fixture(scope="class")
    def tool(self, workspace_client):
        from databricks_langchain.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        return VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )

    def test_tool_invoke_returns_string(self, tool):
        result = tool.invoke("machine learning")
        assert isinstance(result, str)

    def test_tool_invoke_returns_nonempty_content(self, tool):
        result = tool.invoke("machine learning")
        assert len(result) > 0
        # Result contains Document representations with page_content
        assert "page_content" in result

    def test_tool_invoke_returns_relevant_results(self, tool):
        result = tool.invoke("machine learning artificial intelligence")
        assert len(result) > 0
        assert "page_content" in result

    def test_tool_invoke_with_num_results(self, workspace_client):
        from databricks_langchain.vector_search_retriever_tool import (
            VectorSearchRetrieverTool,
        )

        tool = VectorSearchRetrieverTool(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
            num_results=2,
        )
        result = tool.invoke("technology")
        assert isinstance(result, str)
        assert len(result) > 0


# =============================================================================
# DatabricksVectorSearch VectorStore Tests
# =============================================================================


@pytest.mark.integration
class TestLangChainDatabricksVectorSearch:
    """Test DatabricksVectorSearch vectorstore with live index."""

    @pytest.fixture(scope="class")
    def vectorstore(self, workspace_client):
        from databricks_langchain.vectorstores import DatabricksVectorSearch

        return DatabricksVectorSearch(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )

    def test_similarity_search_returns_documents(self, vectorstore):
        from langchain_core.documents import Document

        docs = vectorstore.similarity_search("machine learning", k=3)
        assert isinstance(docs, list)
        assert len(docs) <= 3
        for doc in docs:
            assert isinstance(doc, Document)
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0

    def test_similarity_search_with_score(self, vectorstore):
        results = vectorstore.similarity_search_with_score("Databricks analytics", k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for _doc, score in results:
            assert isinstance(score, (int, float))

    def test_similarity_search_with_filter(self, vectorstore):
        docs = vectorstore.similarity_search("technology", k=10, filter={"category": "databricks"})
        # Our test data has only 2 docs with category="databricks"
        assert len(docs) > 0
        assert len(docs) <= 2

    def test_similarity_search_with_columns_returns_metadata(self, workspace_client):
        from databricks_langchain.vectorstores import DatabricksVectorSearch

        vs = DatabricksVectorSearch(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
            columns=["title", "category"],
        )
        docs = vs.similarity_search("machine learning", k=1)
        assert len(docs) > 0
        metadata = docs[0].metadata
        assert "title" in metadata or "category" in metadata


# =============================================================================
# Kwargs Pass-Through Tests
# =============================================================================


@pytest.mark.integration
class TestLangChainKwargsPassThrough:
    """Verify our code correctly forwards/filters kwargs to the VS client."""

    @pytest.fixture(scope="class")
    def vectorstore(self, workspace_client):
        from databricks_langchain.vectorstores import DatabricksVectorSearch

        return DatabricksVectorSearch(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )

    def test_score_threshold_kwarg_accepted(self, vectorstore):
        results = vectorstore.similarity_search_with_score(
            "machine learning", k=3, score_threshold=0.99
        )
        # High threshold may return fewer or no results, but should not error
        assert isinstance(results, list)

    def test_invalid_kwargs_silently_dropped(self, vectorstore):
        docs = vectorstore.similarity_search(
            "machine learning",
            k=3,
            totally_fake_kwarg="ignored",
            another_invalid_kwarg=42,
        )
        # Our inspect.signature filtering should drop unknown kwargs
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_query_type_override(self, vectorstore):
        docs = vectorstore.similarity_search("machine learning", k=3, query_type="HYBRID")
        assert isinstance(docs, list)
        assert len(docs) > 0


# =============================================================================
# Auth Path Tests
# =============================================================================


@pytest.mark.integration
class TestLangChainAuthPaths:
    """Verify auth credentials are correctly forwarded to VectorSearchClient."""

    def test_current_auth_produces_working_vectorstore(self, workspace_client):
        from databricks_langchain.vectorstores import DatabricksVectorSearch

        vs = DatabricksVectorSearch(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=workspace_client,
        )
        docs = vs.similarity_search("machine learning", k=1)
        assert len(docs) > 0

    def test_pat_auth_produces_working_vectorstore(self, workspace_client):
        from databricks.sdk import WorkspaceClient

        from databricks_langchain.vectorstores import DatabricksVectorSearch

        # Extract a bearer token from the current auth (works for any auth type)
        headers = workspace_client.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        assert token, "Could not extract bearer token from workspace client"

        pat_wc = WorkspaceClient(host=workspace_client.config.host, token=token, auth_type="pat")
        vs = DatabricksVectorSearch(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=pat_wc,
        )
        docs = vs.similarity_search("machine learning", k=1)
        assert len(docs) > 0

    def test_oauth_m2m_auth_produces_working_vectorstore(self):
        from databricks.sdk import WorkspaceClient

        from databricks_langchain.vectorstores import DatabricksVectorSearch

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
        vs = DatabricksVectorSearch(
            index_name=DELTA_SYNC_INDEX,
            workspace_client=oauth_wc,
        )
        docs = vs.similarity_search("machine learning", k=1)
        assert len(docs) > 0
