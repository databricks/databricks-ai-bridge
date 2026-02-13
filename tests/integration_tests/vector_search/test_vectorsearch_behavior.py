"""
Minimal smoke tests for Vector Search operations.

These tests verify basic search and filter behavior against live Databricks
Vector Search indexes. They focus on our bridge code's correctness rather
than exercising the VS SDK's own behavior.

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


@pytest.mark.behavior
class TestVectorSearchBasicBehavior:
    """Minimal smoke tests: one call per index type + filter verification."""

    def test_delta_sync_text_search(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="machine learning artificial intelligence",
            num_results=3,
        )
        assert "result" in resp
        assert "data_array" in resp["result"]
        assert "manifest" in resp

        data_array = resp["result"]["data_array"]
        assert len(data_array) > 0
        assert len(data_array) <= 3

        num_columns = len(resp["manifest"]["columns"])
        for row in data_array:
            assert len(row) == num_columns

        # Score column exists and contains valid floats (regression: not empty string)
        column_names = [c["name"] for c in resp["manifest"]["columns"]]
        score_cols = [i for i, c in enumerate(column_names) if "score" in c.lower()]
        assert len(score_cols) > 0, f"No score column found in {column_names}"
        score_idx = score_cols[0]
        for row in data_array:
            score = row[score_idx]
            assert score != "", "Score must not be empty string (regression)"
            assert isinstance(score, (int, float)), f"Score must be numeric, got {type(score)}"

    def test_delta_sync_filter_passes_through(self, delta_sync_index):
        resp = delta_sync_index.similarity_search(
            columns=["id", "content", "category"],
            query_text="technology",
            filters={"category": "databricks"},
            num_results=10,
        )
        data_array = resp["result"]["data_array"]
        # Our test data has only 2 docs with category="databricks"
        assert len(data_array) > 0
        assert len(data_array) <= 2

    def test_direct_access_vector_search(self, direct_access_index, test_query_vector):
        resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id", "content"],
            num_results=3,
        )
        assert "result" in resp
        assert "data_array" in resp["result"]
        assert "manifest" in resp

        data_array = resp["result"]["data_array"]
        assert len(data_array) > 0
        assert len(data_array) <= 3

        num_columns = len(resp["manifest"]["columns"])
        for row in data_array:
            assert len(row) == num_columns

        # Score column contains valid floats
        column_names = [c["name"] for c in resp["manifest"]["columns"]]
        score_cols = [i for i, c in enumerate(column_names) if "score" in c.lower()]
        assert len(score_cols) > 0, f"No score column found in {column_names}"
        score_idx = score_cols[0]
        for row in data_array:
            score = row[score_idx]
            assert score != "", "Score must not be empty string (regression)"
            assert isinstance(score, (int, float)), f"Score must be numeric, got {type(score)}"

    def test_direct_access_column_selection(self, direct_access_index, test_query_vector):
        resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id", "title"],
            num_results=1,
        )
        column_names = [c["name"] for c in resp["manifest"]["columns"]]
        assert "id" in column_names
        assert "title" in column_names
