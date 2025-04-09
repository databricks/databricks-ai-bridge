import pytest

from databricks_ai_bridge.utils.vector_search import (
    IndexDetails,
    RetrieverSchema,
    parse_vector_search_response,
    validate_and_get_return_columns,
    validate_and_get_text_column,
)

import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# -- Mock setup --

@dataclass
class MockDocument:
    page_content: str
    metadata: Dict[str, Any]

@dataclass
class IndexDetails:
    primary_key: str

# -- Fixtures --

@pytest.fixture
def search_resp():
    return {
        "manifest": {
            "columns": [
                {"name": "id"},
                {"name": "text"},
                {"name": "uri"},
                {"name": "chunk"},
                {"name": "extra"},
                {"name": "score"},
            ]
        },
        "result": {
            "data_array": [
                [1, "This is text A", "doc://a", "chunk-1", "x", 0.9],
                [2, "This is text B", "doc://b", "chunk-2", "y", 0.8],
            ]
        }
    }

@pytest.fixture
def retriever_schema():
    return RetrieverSchema(
        text_column="text",
        doc_uri="uri",
        chunk_id="chunk",
        other_columns=["extra"]
    )

@pytest.fixture
def index_details():
    return IndexDetails(primary_key="id")


def test_parses_basic_response(search_resp, index_details, retriever_schema):
    results = parse_vector_search_response(
        search_resp=search_resp,
        index_details=index_details,
        retriever_schema=retriever_schema,
        document_class=MockDocument,
    )

    assert len(results) == 2

    doc1, score1 = results[0]
    assert doc1.page_content == "This is text A"
    assert doc1.metadata["doc_uri"] == "doc://a"
    assert doc1.metadata["chunk_id"] == "chunk-1"
    assert doc1.metadata["extra"] == "x"
    assert doc1.metadata["id"] == 1
    assert score1 == 0.9

    doc2, score2 = results[1]
    assert doc2.page_content == "This is text B"
    assert doc2.metadata["doc_uri"] == "doc://b"
    assert doc2.metadata["chunk_id"] == "chunk-2"
    assert doc2.metadata["extra"] == "y"
    assert doc2.metadata["id"] == 2
    assert score2 == 0.8


def test_ignores_specified_columns(search_resp, index_details, retriever_schema):
    results = parse_vector_search_response(
        search_resp=search_resp,
        index_details=index_details,
        retriever_schema=retriever_schema,
        ignore_cols=["extra"],
        document_class=MockDocument,
    )

    doc, _ = results[0]
    assert "extra" not in doc.metadata


def test_handles_empty_results(index_details, retriever_schema):
    empty_resp = {"manifest": {"columns": []}, "result": {"data_array": []}}
    results = parse_vector_search_response(
        search_resp=empty_resp,
        index_details=index_details,
        retriever_schema=retriever_schema,
        document_class=MockDocument,
    )
    assert results == []


def test_missing_optional_fields_handled_gracefully(search_resp):
    retriever_schema = RetrieverSchema(text_column="text")  # no doc_uri, chunk_id, or other_columns
    index_details = IndexDetails(primary_key="id")

    results = parse_vector_search_response(
        search_resp=search_resp,
        index_details=index_details,
        retriever_schema=retriever_schema,
        document_class=MockDocument,
    )

    doc, _ = results[0]
    assert doc.page_content == "This is text A"
    assert doc.metadata["id"] == 1
    assert "doc_uri" not in doc.metadata
    assert "chunk_id" not in doc.metadata
