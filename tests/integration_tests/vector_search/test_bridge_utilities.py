"""
Integration tests for core bridge utilities.

These tests verify that the bridge layer (IndexDetails, parse_vector_search_response,
validate_and_get_text_column, validate_and_get_return_columns) works correctly
against real Vector Search API responses. This catches cases where the SDK
response format changes in ways that break our parsing code.
"""

from __future__ import annotations

import os

import pytest

from databricks_ai_bridge.utils.vector_search import (
    RetrieverSchema,
    parse_vector_search_response,
    validate_and_get_return_columns,
    validate_and_get_text_column,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_VS_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set RUN_VS_INTEGRATION_TESTS=1 to enable.",
)


# =============================================================================
# IndexDetails Tests - Delta-Sync Index
# =============================================================================


@pytest.mark.integration
class TestIndexDetailsDeltaSync:
    """Verify IndexDetails works correctly with a live delta-sync index."""

    def test_name_matches(self, delta_sync_index_details, delta_sync_index_name):
        assert delta_sync_index_details.name == delta_sync_index_name

    def test_is_delta_sync_index(self, delta_sync_index_details):
        assert delta_sync_index_details.is_delta_sync_index() is True

    def test_is_not_direct_access(self, delta_sync_index_details):
        assert delta_sync_index_details.is_direct_access_index() is False

    def test_is_managed_embeddings(self, delta_sync_index_details):
        assert delta_sync_index_details.is_databricks_managed_embeddings() is True

    def test_has_primary_key(self, delta_sync_index_details):
        assert delta_sync_index_details.primary_key == "id"

    def test_has_embedding_source_column(self, delta_sync_index_details):
        source_col = delta_sync_index_details.embedding_source_column
        assert source_col.get("name") == "content"

    def test_index_spec_has_source_table(self, delta_sync_index_details):
        spec = delta_sync_index_details.index_spec
        assert "source_table" in spec

    def test_schema_is_none(self, delta_sync_index_details):
        # schema property is only populated for direct-access indexes
        assert delta_sync_index_details.schema is None


# =============================================================================
# IndexDetails Tests - Direct-Access Index
# =============================================================================


@pytest.mark.integration
class TestIndexDetailsDirectAccess:
    """Verify IndexDetails works correctly with a live direct-access index."""

    def test_name_matches(self, direct_access_index_details, direct_access_index_name):
        assert direct_access_index_details.name == direct_access_index_name

    def test_is_direct_access_index(self, direct_access_index_details):
        assert direct_access_index_details.is_direct_access_index() is True

    def test_is_not_delta_sync(self, direct_access_index_details):
        assert direct_access_index_details.is_delta_sync_index() is False

    def test_not_managed_embeddings(self, direct_access_index_details):
        assert direct_access_index_details.is_databricks_managed_embeddings() is False

    def test_has_primary_key(self, direct_access_index_details):
        assert direct_access_index_details.primary_key == "id"

    def test_has_embedding_vector_column(self, direct_access_index_details):
        vec_col = direct_access_index_details.embedding_vector_column
        assert vec_col.get("name") == "content_vector"
        assert vec_col.get("embedding_dimension") == 1024

    def test_has_schema(self, direct_access_index_details):
        schema = direct_access_index_details.schema
        assert schema is not None
        assert "id" in schema
        assert "content" in schema
        assert "content_vector" in schema


# =============================================================================
# parse_vector_search_response Tests
# =============================================================================


@pytest.mark.integration
class TestParseVectorSearchResponse:
    """Test parse_vector_search_response against real search responses."""

    def test_parse_delta_sync_response(self, delta_sync_index):
        search_resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="machine learning",
            num_results=3,
        )

        retriever_schema = RetrieverSchema(text_column="content")
        results = parse_vector_search_response(
            search_resp=search_resp,
            retriever_schema=retriever_schema,
            document_class=dict,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(doc, dict)
            assert "page_content" in doc
            assert "metadata" in doc
            assert isinstance(doc["page_content"], str)
            assert len(doc["page_content"]) > 0
            assert isinstance(score, (int, float))
            assert score != "", "Score must not be empty string (regression)"
            assert score >= 0, f"Score should be non-negative, got {score}"

    def test_parse_with_doc_uri_and_primary_key(self, delta_sync_index):
        search_resp = delta_sync_index.similarity_search(
            query_text="machine learning",
            columns=["id", "content", "source"],
            num_results=1,
        )

        retriever_schema = RetrieverSchema(
            text_column="content",
            doc_uri="source",
            primary_key="id",
        )
        results = parse_vector_search_response(
            search_resp=search_resp,
            retriever_schema=retriever_schema,
            document_class=dict,
        )

        assert len(results) > 0
        doc, _ = results[0]
        assert "doc_uri" in doc["metadata"]
        assert "chunk_id" in doc["metadata"]

    def test_parse_with_other_columns(self, delta_sync_index):
        search_resp = delta_sync_index.similarity_search(
            query_text="machine learning",
            columns=["id", "content", "title", "category"],
            num_results=1,
        )

        retriever_schema = RetrieverSchema(
            text_column="content",
            primary_key="id",
            other_columns=["title", "category"],
        )
        results = parse_vector_search_response(
            search_resp=search_resp,
            retriever_schema=retriever_schema,
            document_class=dict,
        )

        assert len(results) > 0
        doc, _ = results[0]
        assert "title" in doc["metadata"]
        assert "category" in doc["metadata"]

    def test_parse_with_include_score(self, delta_sync_index):
        search_resp = delta_sync_index.similarity_search(
            columns=["id", "content"],
            query_text="machine learning",
            num_results=1,
        )

        retriever_schema = RetrieverSchema(text_column="content")
        results = parse_vector_search_response(
            search_resp=search_resp,
            retriever_schema=retriever_schema,
            document_class=dict,
            include_score=True,
        )

        assert len(results) > 0
        doc, _ = results[0]
        assert "score" in doc["metadata"]
        score = doc["metadata"]["score"]
        assert score != "", "Score must not be empty string (regression)"
        assert isinstance(score, (int, float))
        assert score >= 0, f"Score should be non-negative, got {score}"

    def test_parse_direct_access_response(self, direct_access_index, test_query_vector):
        search_resp = direct_access_index.similarity_search(
            query_vector=test_query_vector,
            columns=["id", "content", "title"],
            num_results=3,
        )

        retriever_schema = RetrieverSchema(
            text_column="content",
            primary_key="id",
        )
        results = parse_vector_search_response(
            search_resp=search_resp,
            retriever_schema=retriever_schema,
            document_class=dict,
        )

        assert isinstance(results, list)
        for doc, score in results:
            assert "page_content" in doc
            assert "metadata" in doc
            assert isinstance(score, (int, float))
            assert score != "", "Score must not be empty string (regression)"
            assert score >= 0, f"Score should be non-negative, got {score}"


# =============================================================================
# validate_and_get_text_column Tests
# =============================================================================


@pytest.mark.integration
class TestValidateAndGetTextColumn:
    """Test text column validation against live index details."""

    def test_managed_embeddings_returns_source_column(self, delta_sync_index_details):
        result = validate_and_get_text_column(None, delta_sync_index_details)
        assert result == "content"

    def test_managed_embeddings_raises_if_text_column_set(self, delta_sync_index_details):
        with pytest.raises(ValueError, match="Do not pass the `text_column` parameter"):
            validate_and_get_text_column("content", delta_sync_index_details)

    def test_direct_access_requires_text_column(self, direct_access_index_details):
        with pytest.raises(ValueError, match="text_column.*required"):
            validate_and_get_text_column(None, direct_access_index_details)

    def test_direct_access_returns_provided_text_column(self, direct_access_index_details):
        result = validate_and_get_text_column("content", direct_access_index_details)
        assert result == "content"


# =============================================================================
# validate_and_get_return_columns Tests
# =============================================================================


@pytest.mark.integration
class TestValidateAndGetReturnColumns:
    """Test column validation against live index details."""

    def test_adds_primary_key_if_missing(self, delta_sync_index_details):
        columns = ["content"]
        result = validate_and_get_return_columns(
            columns.copy(), "content", delta_sync_index_details
        )
        assert "id" in result

    def test_adds_text_column_if_missing(self, delta_sync_index_details):
        columns = ["id"]
        result = validate_and_get_return_columns(
            columns.copy(), "content", delta_sync_index_details
        )
        assert "content" in result

    def test_direct_access_validates_against_schema(self, direct_access_index_details):
        with pytest.raises(ValueError, match="not in the index schema"):
            validate_and_get_return_columns(
                ["nonexistent_column"],
                "content",
                direct_access_index_details,
            )

    def test_direct_access_valid_columns_pass(self, direct_access_index_details):
        result = validate_and_get_return_columns(
            ["title", "category"],
            "content",
            direct_access_index_details,
        )
        assert "title" in result
        assert "category" in result
        assert "id" in result  # primary key auto-added
        assert "content" in result  # text column auto-added
