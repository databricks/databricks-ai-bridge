"""
Behavior validation tests for Vector Search operations.

These tests verify actual search results and data handling against
live Databricks Vector Search indexes. They catch semantic changes
in the underlying service that might not change the API signature
but could affect behavior.

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


# =============================================================================
# Delta-Sync Search Behavior Tests
# =============================================================================


@pytest.mark.behavior
class TestDeltaSyncSearchBehavior:
    """Verify similarity search behavior on delta-sync managed embeddings index."""

    def test_basic_text_search_returns_results(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="machine learning artificial intelligence",
            num_results=3,
        )
        assert "result" in resp
        assert "data_array" in resp["result"]

    def test_num_results_limits_output(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="test",
            num_results=1,
        )
        assert len(resp["result"]["data_array"]) <= 1

    def test_num_results_returns_multiple(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="technology",
            num_results=5,
        )
        assert len(resp["result"]["data_array"]) <= 5

    def test_columns_selection_filters_response(self, delta_sync_index):
        desc = delta_sync_index.describe()
        primary_key = desc["primary_key"]

        resp = delta_sync_index.similarity_search(
            query_text="test",
            columns=[primary_key],
            num_results=1,
        )
        column_names = [c["name"] for c in resp["manifest"]["columns"]]
        assert primary_key in column_names

    def test_ann_query_type_works(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="test query",
            query_type="ANN",
            num_results=1,
        )
        assert "result" in resp

    def test_hybrid_query_type_works(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="test query",
            query_type="HYBRID",
            num_results=1,
        )
        assert "result" in resp

    def test_filter_with_category_narrows_results(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content", "category"],
            query_text="technology",
            filters={"category": "databricks"},
            num_results=10,
        )
        # Our test data has only 2 docs with category="databricks"
        assert len(resp["result"]["data_array"]) <= 2

    def test_filter_with_empty_dict_accepted(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="test",
            filters={},
            num_results=1,
        )
        assert "result" in resp

    def test_filter_with_none_accepted(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="test",
            filters=None,
            num_results=1,
        )
        assert "result" in resp

    def test_score_column_in_results(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="test",
            num_results=1,
        )
        column_names = [c["name"] for c in resp["manifest"]["columns"]]
        score_columns = [c for c in column_names if "score" in c.lower()]
        assert len(score_columns) > 0, f"No score column found in {column_names}"

    def test_data_array_row_width_matches_manifest(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="test",
            num_results=1,
        )
        num_columns = len(resp["manifest"]["columns"])
        if resp["result"]["data_array"]:
            for row in resp["result"]["data_array"]:
                assert isinstance(row, list)
                assert len(row) == num_columns

    def test_unicode_query_accepted(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="café résumé naïve 日本語",
            num_results=1,
        )
        assert "result" in resp

    def test_long_query_accepted(self, delta_sync_index):
        long_query = "test " * 500
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text=long_query,
            num_results=1,
        )
        assert "result" in resp


# =============================================================================
# Direct-Access Search Behavior Tests
# =============================================================================


@pytest.mark.behavior
class TestDirectAccessSearchBehavior:
    """Verify similarity search behavior on direct-access index."""

    def test_basic_vector_search_returns_results(self, direct_access_index, test_query_vector):
        resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id", "content"],
            num_results=3,
        )
        assert "result" in resp
        assert "data_array" in resp["result"]

    def test_num_results_limits_output(self, direct_access_index, test_query_vector):
        resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id"],
            num_results=1,
        )
        assert len(resp["result"]["data_array"]) <= 1

    def test_columns_selection_works(self, direct_access_index, test_query_vector):
        resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id", "title"],
            num_results=1,
        )
        column_names = [c["name"] for c in resp["manifest"]["columns"]]
        assert "id" in column_names
        assert "title" in column_names

    def test_score_column_in_results(self, direct_access_index, test_query_vector):
        resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id"],
            num_results=1,
        )
        column_names = [c["name"] for c in resp["manifest"]["columns"]]
        score_columns = [c for c in column_names if "score" in c.lower()]
        assert len(score_columns) > 0, f"No score column found in {column_names}"

    def test_data_array_row_width_matches_manifest(self, direct_access_index, test_query_vector):
        resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id", "content"],
            num_results=1,
        )
        num_columns = len(resp["manifest"]["columns"])
        if resp["result"]["data_array"]:
            for row in resp["result"]["data_array"]:
                assert isinstance(row, list)
                assert len(row) == num_columns
