"""
Contract validation tests for databricks-vectorsearch dependency.

These tests verify the API contract hasn't changed in ways that would break
databricks-ai-bridge. They check:
- Import paths exist
- Method signatures accept expected parameters
- Response structures contain expected keys

Run daily to catch breaking changes from upstream dependencies early.
"""

from __future__ import annotations

import inspect
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.",
)


# =============================================================================
# Import Contract Tests
# =============================================================================


@pytest.mark.contract
class TestImportContract:
    """Verify expected imports from databricks-vectorsearch exist."""

    def test_vector_search_client_import(self):
        from databricks.vector_search.client import VectorSearchClient

        assert VectorSearchClient is not None

    def test_vector_search_index_import(self):
        from databricks.vector_search.client import VectorSearchIndex

        assert VectorSearchIndex is not None

    def test_credential_strategy_import(self):
        from databricks.vector_search.utils import CredentialStrategy

        assert CredentialStrategy is not None

    def test_credential_strategy_model_serving_value(self):
        from databricks.vector_search.utils import CredentialStrategy

        assert hasattr(CredentialStrategy, "MODEL_SERVING_USER_CREDENTIALS")

    def test_reranker_import(self):
        from databricks.vector_search.reranker import Reranker

        assert Reranker is not None

    def test_databricks_reranker_import(self):
        from databricks.vector_search.reranker import DatabricksReranker

        assert DatabricksReranker is not None


# =============================================================================
# VectorSearchClient Contract Tests
# =============================================================================


@pytest.mark.contract
class TestVectorSearchClientContract:
    """Verify VectorSearchClient API contract."""

    def test_client_has_get_index_method(self, vector_search_client):
        assert hasattr(vector_search_client, "get_index")
        assert callable(vector_search_client.get_index)

    def test_get_index_returns_index_instance(self, vector_search_client, delta_sync_index_name):
        from databricks.vector_search.client import VectorSearchIndex

        index = vector_search_client.get_index(index_name=delta_sync_index_name)
        assert isinstance(index, VectorSearchIndex)


# =============================================================================
# Delta-Sync Index Contract Tests
# =============================================================================


@pytest.mark.contract
class TestDeltaSyncIndexContract:
    """Verify VectorSearchIndex API contract for delta-sync indexes."""

    def test_describe_returns_expected_keys(self, delta_sync_index):
        desc = delta_sync_index.describe()
        assert "name" in desc, "Missing 'name' in describe()"
        assert "primary_key" in desc, "Missing 'primary_key' in describe()"
        assert "index_type" in desc, "Missing 'index_type' in describe()"
        assert "delta_sync_index_spec" in desc, "Missing 'delta_sync_index_spec' in describe()"

    def test_describe_has_embedding_source_columns(self, delta_sync_index):
        desc = delta_sync_index.describe()
        spec = desc["delta_sync_index_spec"]
        assert "embedding_source_columns" in spec, (
            "Missing 'embedding_source_columns' in delta_sync_index_spec"
        )

    def test_similarity_search_method_exists(self, delta_sync_index):
        assert hasattr(delta_sync_index, "similarity_search")
        assert callable(delta_sync_index.similarity_search)

    def test_similarity_search_signature_has_expected_params(self, delta_sync_index):
        sig = inspect.signature(delta_sync_index.similarity_search)
        params = sig.parameters
        expected_params = [
            "query_text",
            "query_vector",
            "columns",
            "filters",
            "num_results",
            "query_type",
            "reranker",
        ]
        for param in expected_params:
            assert param in params, f"Missing parameter '{param}' in similarity_search"

    def test_similarity_search_response_structure(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="test query",
            num_results=1,
        )
        assert "manifest" in resp, "Missing 'manifest' in response"
        assert "columns" in resp["manifest"], "Missing 'columns' in manifest"
        assert "result" in resp, "Missing 'result' in response"
        assert "data_array" in resp["result"], "Missing 'data_array' in result"

        for col in resp["manifest"]["columns"]:
            assert "name" in col, "Missing 'name' in column manifest entry"


# =============================================================================
# Direct-Access Index Contract Tests
# =============================================================================


@pytest.mark.contract
class TestDirectAccessIndexContract:
    """Verify VectorSearchIndex API contract for direct-access indexes."""

    def test_describe_returns_expected_keys(self, direct_access_index):
        desc = direct_access_index.describe()
        assert "name" in desc, "Missing 'name' in describe()"
        assert "primary_key" in desc, "Missing 'primary_key' in describe()"
        assert "index_type" in desc, "Missing 'index_type' in describe()"
        assert "direct_access_index_spec" in desc, (
            "Missing 'direct_access_index_spec' in describe()"
        )

    def test_describe_has_schema_json(self, direct_access_index):
        desc = direct_access_index.describe()
        spec = desc["direct_access_index_spec"]
        assert "schema_json" in spec, "Missing 'schema_json' in direct_access_index_spec"

    def test_similarity_search_with_query_vector(self, direct_access_index, test_query_vector):
        resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id", "content"],
            num_results=1,
        )
        assert "manifest" in resp
        assert "result" in resp

    def test_similarity_search_response_structure(self, direct_access_index, test_query_vector):
        resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id", "content"],
            num_results=1,
        )
        assert "columns" in resp["manifest"]
        assert "data_array" in resp["result"]
        for col in resp["manifest"]["columns"]:
            assert "name" in col


# =============================================================================
# Reranker Contract Tests
# =============================================================================


@pytest.mark.contract
class TestRerankerContract:
    """Verify Reranker API contract."""

    def test_databricks_reranker_accepts_columns_to_rerank(self):
        from databricks.vector_search.reranker import DatabricksReranker

        reranker = DatabricksReranker(columns_to_rerank=["text"])
        assert reranker is not None

    def test_databricks_reranker_has_columns_to_rerank_attr(self):
        from databricks.vector_search.reranker import DatabricksReranker

        reranker = DatabricksReranker(columns_to_rerank=["text", "title"])
        assert hasattr(reranker, "columns_to_rerank") or hasattr(reranker, "_columns_to_rerank")
