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


def test_databricks_rm_forward_string_query(mock_vector_search_response):
    """Test forward method with string query and ANN search."""
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.return_value = {"userName": "test_user"}
        mock_ws.vector_search_indexes.query_index.return_value.as_dict.return_value = (
            mock_vector_search_response
        )
        MockWSClient.return_value = mock_ws

        rm = DatabricksRM(
            databricks_index_name="test_index",
            databricks_token="test_token",
            databricks_endpoint="https://test.databricks.com",
        )

        result = rm("test query", query_type="ANN")

        # Verify API call
        call_args = mock_ws.vector_search_indexes.query_index.call_args[1]
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


def test_databricks_rm_forward_vector_query(mock_vector_search_response):
    """Test forward method with vector query and HYBRID search."""
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.return_value = {"userName": "test_user"}
        mock_ws.vector_search_indexes.query_index.return_value.as_dict.return_value = (
            mock_vector_search_response
        )
        MockWSClient.return_value = mock_ws

        rm = DatabricksRM(databricks_index_name="test_index")
        query_vector = [0.1, 0.2, 0.3]

        rm(query_vector, query_type="HYBRID")

        # Verify API call
        call_args = mock_ws.vector_search_indexes.query_index.call_args[1]
        assert call_args["index_name"] == "test_index"
        assert call_args["query_type"] == "HYBRID"
        assert call_args["query_text"] is None
        assert call_args["query_vector"] == query_vector
        assert set(call_args["columns"]) == {"id", "text"}


def test_databricks_rm_agent_framework_format(mock_vector_search_response_with_uri):
    """Test forward method returning agent framework format."""
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.return_value = {"userName": "test_user"}
        mock_ws.vector_search_indexes.query_index.return_value.as_dict.return_value = (
            mock_vector_search_response_with_uri
        )
        MockWSClient.return_value = mock_ws

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
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.return_value = {"userName": "test_user"}
        MockWSClient.return_value = mock_ws

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

        # Verify WorkspaceClient was created with token auth
        MockWSClient.assert_called_once_with(
            host="https://test.databricks.com",
            token="test_token",
        )
        # Verify credentials validation was performed
        mock_ws.current_user.me.assert_called_once()


def test_databricks_rm_initialization_with_custom_workspace_client():
    """Test initialization with custom workspace_client."""
    mock_workspace_client = MagicMock()
    mock_workspace_client.current_user.me.return_value = {"userName": "test_user"}

    rm = DatabricksRM(
        databricks_index_name="test_index",
        workspace_client=mock_workspace_client,
        k=5,
    )

    assert rm.databricks_index_name == "test_index"
    assert rm.workspace_client == mock_workspace_client
    assert rm.k == 5
    assert rm.docs_id_column_name == "id"
    assert rm.text_column_name == "text"
    assert not rm.use_with_databricks_agent_framework

    # Verify credentials validation was performed
    mock_workspace_client.current_user.me.assert_called_once()


def test_databricks_rm_query_with_custom_workspace_client():
    """Test that custom workspace_client is used for queries."""
    mock_workspace_client = MagicMock()
    mock_workspace_client.current_user.me.return_value = {"userName": "test_user"}

    mock_response = {
        "result": {
            "data_array": [
                ["doc1", "This is document 1", 0.95],
            ]
        },
        "manifest": {
            "columns": [
                {"name": "id"},
                {"name": "text"},
                {"name": "score"},
            ]
        },
    }
    mock_workspace_client.vector_search_indexes.query_index.return_value.as_dict.return_value = (
        mock_response
    )

    rm = DatabricksRM(
        databricks_index_name="test_index",
        workspace_client=mock_workspace_client,
    )

    result = rm("test query")

    # Verify that the custom workspace_client was used for the query
    mock_workspace_client.vector_search_indexes.query_index.assert_called_once()
    call_args = mock_workspace_client.vector_search_indexes.query_index.call_args[1]
    assert call_args["index_name"] == "test_index"
    assert call_args["query_text"] == "test query"

    # Verify results
    assert len(result.docs) == 1
    assert result.docs[0] == "This is document 1"
    assert result.doc_ids[0] == "doc1"


def test_databricks_rm_service_principal_auth(mock_vector_search_response):
    """Test querying with service principal authentication."""
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.return_value = {"userName": "test_user"}
        mock_ws.vector_search_indexes.query_index.return_value.as_dict.return_value = (
            mock_vector_search_response
        )
        MockWSClient.return_value = mock_ws

        rm = DatabricksRM(
            databricks_index_name="test_index",
            databricks_client_id="test_client_id",
            databricks_client_secret="test_client_secret",
        )

        rm("test query")

        # Verify WorkspaceClient was created with service principal auth
        MockWSClient.assert_called_once_with(
            client_id="test_client_id",
            client_secret="test_client_secret",
        )


def test_databricks_rm_invalid_query_type():
    """Test forward method with invalid query type."""
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.return_value = {"userName": "test_user"}
        MockWSClient.return_value = mock_ws

        rm = DatabricksRM(databricks_index_name="test_index")

        with pytest.raises(ValueError, match="Invalid query_type: INVALID"):
            rm("test query", query_type="INVALID")


def test_databricks_rm_missing_column_error():
    """Test error when ID column is missing from index."""
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.return_value = {"userName": "test_user"}

        # Response missing the ID column
        mock_response = {
            "result": {"data_array": []},
            "manifest": {"columns": [{"name": "text"}, {"name": "score"}]},
        }
        mock_ws.vector_search_indexes.query_index.return_value.as_dict.return_value = mock_response
        MockWSClient.return_value = mock_ws

        rm = DatabricksRM(
            databricks_index_name="test_index",
            docs_id_column_name="id",
        )

        with pytest.raises(
            ValueError, match="docs_id_column_name: 'id' is not in the index columns"
        ):
            rm("test query")


def test_databricks_rm_result_sorting():
    """Test that results are sorted by score in descending order."""
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.return_value = {"userName": "test_user"}

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
        mock_ws.vector_search_indexes.query_index.return_value.as_dict.return_value = mock_response
        MockWSClient.return_value = mock_ws

        rm = DatabricksRM(databricks_index_name="test_index", k=3)

        result = rm("test query")

        # Should be sorted by score (highest first)
        assert result.doc_ids == ["doc1", "doc3", "doc2"]
        assert result.docs == ["Document 1", "Document 3", "Document 2"]


def test_databricks_rm_invalid_credentials_error():
    """Test error when workspace client credentials validation fails."""
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.side_effect = Exception("Invalid credentials")
        MockWSClient.return_value = mock_ws

        with pytest.raises(RuntimeError, match="Failed to validate databricks credentials"):
            DatabricksRM(
                databricks_index_name="test_index",
                databricks_token="invalid_token",
                databricks_endpoint="https://test.databricks.com",
            )

        # Verify credentials validation was attempted
        mock_ws.current_user.me.assert_called_once()


def test_databricks_rm_custom_workspace_client_invalid_credentials():
    """Test error when custom workspace client credentials validation fails."""
    mock_workspace_client = MagicMock()
    mock_workspace_client.current_user.me.side_effect = Exception("Invalid credentials")

    with pytest.raises(RuntimeError, match="Failed to validate databricks credentials"):
        DatabricksRM(
            databricks_index_name="test_index",
            workspace_client=mock_workspace_client,
        )

    # Verify credentials validation was attempted
    mock_workspace_client.current_user.me.assert_called_once()


def test_databricks_rm_fallback_to_default_auth():
    """Test fallback to default auth when no credentials are provided."""
    with patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient") as MockWSClient:
        mock_ws = MagicMock()
        mock_ws.current_user.me.return_value = {"userName": "test_user"}
        MockWSClient.return_value = mock_ws

        rm = DatabricksRM(databricks_index_name="test_index")

        assert rm.databricks_index_name == "test_index"

        # Verify WorkspaceClient was created with no auth params (default auth)
        MockWSClient.assert_called_once_with()
        # Verify credentials validation was performed
        mock_ws.current_user.me.assert_called_once()
