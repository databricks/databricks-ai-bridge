from unittest.mock import MagicMock, patch

import pytest

from databricks_dspy.retrievers.databricks_rm import DatabricksRM


@pytest.fixture
def mock_vector_search_response():
    """Mock response from Databricks vector search index query."""
    return {
        "result": {
            "data_array": [
                ["doc1", "This is document 1", 0.95, {"category": "tech"}],
                ["doc2", "This is document 2", 0.90, {"category": "science"}],
                ["doc3", "This is document 3", 0.85, {"category": "tech"}],
            ]
        },
        "manifest": {
            "columns": [
                {"name": "id"},
                {"name": "text"},
                {"name": "score"},
                {"name": "metadata"},
            ]
        },
    }


@pytest.fixture
def mock_vector_search_response_with_uri():
    """Mock response with URI column from Databricks vector search index query."""
    return {
        "result": {
            "data_array": [
                ["doc1", "This is document 1", 0.95, "http://doc1.com", {"category": "tech"}],
                ["doc2", "This is document 2", 0.90, "http://doc2.com", {"category": "science"}],
            ]
        },
        "manifest": {
            "columns": [
                {"name": "id"},
                {"name": "text"},
                {"name": "score"},
                {"name": "uri"},
                {"name": "metadata"},
            ]
        },
    }


@patch("databricks.sdk.WorkspaceClient")
def test_databricks_rm_forward_string_query(mock_workspace_client, mock_vector_search_response):
    """Test forward method with string query and ANN search."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.vector_search_indexes.query_index.return_value.as_dict.return_value = (
        mock_vector_search_response
    )

    rm = DatabricksRM(
        databricks_index_name="test_index",
        databricks_token="test_token",
        databricks_endpoint="https://test.databricks.com",
    )

    result = rm("test query", query_type="ANN")

    # Verify API call
    call_args = mock_client.vector_search_indexes.query_index.call_args[1]
    assert call_args["index_name"] == "test_index"
    assert call_args["query_type"] == "ANN"
    assert call_args["query_text"] == "test query"
    assert call_args["query_vector"] is None
    assert set(call_args["columns"]) == {"id", "text"}
    assert call_args["filters_json"] is None
    assert call_args["num_results"] == 3

    # Verify result format
    assert hasattr(result, "docs")
    assert hasattr(result, "doc_ids")
    assert len(result.docs) == 3
    assert result.docs[0] == "This is document 1"
    assert result.doc_ids[0] == "doc1"


@patch("databricks.sdk.WorkspaceClient")
def test_databricks_rm_forward_vector_query(mock_workspace_client, mock_vector_search_response):
    """Test forward method with vector query and HYBRID search."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.vector_search_indexes.query_index.return_value.as_dict.return_value = (
        mock_vector_search_response
    )

    rm = DatabricksRM(databricks_index_name="test_index")
    query_vector = [0.1, 0.2, 0.3]

    rm(query_vector, query_type="HYBRID")

    # Verify API call
    call_args = mock_client.vector_search_indexes.query_index.call_args[1]
    assert call_args["index_name"] == "test_index"
    assert call_args["query_type"] == "HYBRID"
    assert call_args["query_text"] is None
    assert call_args["query_vector"] == query_vector
    assert set(call_args["columns"]) == {"id", "text"}


@patch("databricks.sdk.WorkspaceClient")
def test_databricks_rm_agent_framework_format(
    mock_workspace_client, mock_vector_search_response_with_uri
):
    """Test forward method returning agent framework format."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.vector_search_indexes.query_index.return_value.as_dict.return_value = (
        mock_vector_search_response_with_uri
    )

    with patch("mlflow.models.set_retriever_schema"):
        rm = DatabricksRM(
            databricks_index_name="test_index",
            docs_uri_column_name="uri",
            use_with_databricks_agent_framework=True,
        )

    result = rm.forward("test query")

    # Should return list of Document dictionaries
    assert isinstance(result, list)
    assert len(result) == 2

    doc = result[0]
    assert doc["page_content"] == "This is document 1"
    assert doc["metadata"]["doc_id"] == "doc1"
    assert doc["metadata"]["doc_uri"] == "http://doc1.com"
    assert doc["type"] == "Document"


def test_databricks_rm_initialization():
    """Test initialization with token authentication."""
    rm = DatabricksRM(
        databricks_index_name="test_index",
        databricks_endpoint="https://test.databricks.com",
        databricks_token="test_token",
        k=5,
    )

    assert rm.databricks_index_name == "test_index"
    assert rm.databricks_endpoint == "https://test.databricks.com"
    assert rm.databricks_token == "test_token"
    assert rm.k == 5
    assert rm.docs_id_column_name == "id"
    assert rm.text_column_name == "text"
    assert not rm.use_with_databricks_agent_framework


@patch("databricks.sdk.WorkspaceClient")
def test_databricks_rm_service_principal_auth(mock_workspace_client, mock_vector_search_response):
    """Test querying with service principal authentication."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.vector_search_indexes.query_index.return_value.as_dict.return_value = (
        mock_vector_search_response
    )

    rm = DatabricksRM(
        databricks_index_name="test_index",
        databricks_client_id="test_client_id",
        databricks_client_secret="test_client_secret",
    )

    rm("test query")

    # Verify WorkspaceClient was created with service principal auth
    mock_workspace_client.assert_called_once_with(
        client_id="test_client_id",
        client_secret="test_client_secret",
    )


def test_databricks_rm_invalid_query_type():
    """Test forward method with invalid query type."""
    rm = DatabricksRM(databricks_index_name="test_index")

    with pytest.raises(ValueError, match="Invalid query_type: INVALID"):
        rm("test query", query_type="INVALID")


@patch("databricks.sdk.WorkspaceClient")
def test_databricks_rm_missing_column_error(mock_workspace_client):
    """Test error when ID column is missing from index."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client

    # Response missing the ID column
    mock_response = {
        "result": {"data_array": []},
        "manifest": {"columns": [{"name": "text"}, {"name": "score"}]},
    }
    mock_client.vector_search_indexes.query_index.return_value.as_dict.return_value = mock_response

    rm = DatabricksRM(
        databricks_index_name="test_index",
        docs_id_column_name="id",
    )

    with pytest.raises(ValueError, match="docs_id_column_name: 'id' is not in the index columns"):
        rm("test query")


@patch("databricks.sdk.WorkspaceClient")
def test_databricks_rm_result_sorting(mock_workspace_client):
    """Test that results are sorted by score in descending order."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client

    # Results in random order
    mock_response = {
        "result": {
            "data_array": [
                ["doc2", "Document 2", 0.75],  # Lower score
                ["doc1", "Document 1", 0.95],  # Higher score
                ["doc3", "Document 3", 0.85],  # Middle score
            ]
        },
        "manifest": {"columns": [{"name": "id"}, {"name": "text"}, {"name": "score"}]},
    }
    mock_client.vector_search_indexes.query_index.return_value.as_dict.return_value = mock_response

    rm = DatabricksRM(databricks_index_name="test_index", k=3)

    result = rm("test query")

    # Should be sorted by score (highest first)
    assert result.doc_ids == ["doc1", "doc3", "doc2"]
    assert result.docs == ["Document 1", "Document 3", "Document 2"]
