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


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_forward_string_query(mock_workspace_client, mock_vector_search_response):
    """Test forward method with string query and ANN search."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.return_value = {"userName": "test_user"}
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


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_forward_vector_query(mock_workspace_client, mock_vector_search_response):
    """Test forward method with vector query and HYBRID search."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.return_value = {"userName": "test_user"}
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


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_agent_framework_format(
    mock_workspace_client, mock_vector_search_response_with_uri
):
    """Test forward method returning agent framework format."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.return_value = {"userName": "test_user"}
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
    """Test initialization with token authentication - no workspace client created yet."""
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
    # Workspace client should be None until first access
    assert rm._workspace_client is None


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_lazy_workspace_client_creation(mock_workspace_client):
    """Test that workspace client is created lazily on first access."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.return_value = {"userName": "test_user"}

    rm = DatabricksRM(
        databricks_index_name="test_index",
        databricks_endpoint="https://test.databricks.com",
        databricks_token="test_token",
    )

    # WorkspaceClient should not be created during initialization (lazy initialization)
    mock_workspace_client.assert_not_called()

    # Access workspace_client property to trigger lazy creation
    client = rm.workspace_client

    # Verify WorkspaceClient was created with token auth
    mock_workspace_client.assert_called_once_with(
        host="https://test.databricks.com",
        token="test_token",
    )
    # Verify credentials validation was performed
    mock_client.current_user.me.assert_called_once()
    assert client == mock_client


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
    assert rm._workspace_client == mock_workspace_client  # Stored internally
    assert rm.k == 5
    assert rm.docs_id_column_name == "id"
    assert rm.text_column_name == "text"
    assert not rm.use_with_databricks_agent_framework

    # No credentials validation during initialization (lazy initialization)
    mock_workspace_client.current_user.me.assert_not_called()

    # Then validation occurs when accessing the property
    client = rm.workspace_client
    assert client == mock_workspace_client
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


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_service_principal_auth(mock_workspace_client, mock_vector_search_response):
    """Test querying with service principal authentication."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.return_value = {"userName": "test_user"}
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


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_invalid_query_type(mock_workspace_client):
    """Test forward method with invalid query type."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.return_value = {"userName": "test_user"}

    rm = DatabricksRM(databricks_index_name="test_index")

    with pytest.raises(ValueError, match="Invalid query_type: INVALID"):
        rm("test query", query_type="INVALID")


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_missing_column_error(mock_workspace_client):
    """Test error when ID column is missing from index."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.return_value = {"userName": "test_user"}

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


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_result_sorting(mock_workspace_client):
    """Test that results are sorted by score in descending order."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.return_value = {"userName": "test_user"}

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

    # Verify that current_user.me() was called on the workspace client
    mock_client.current_user.me.assert_called_once()

    # Should be sorted by score (highest first)
    assert result.doc_ids == ["doc1", "doc3", "doc2"]
    assert result.docs == ["Document 1", "Document 3", "Document 2"]


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_invalid_credentials_error(mock_workspace_client):
    """Test error when workspace client credentials validation fails on first access."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.side_effect = Exception("Invalid credentials")

    rm = DatabricksRM(
        databricks_index_name="test_index",
        databricks_token="invalid_token",
        databricks_endpoint="https://test.databricks.com",
    )

    # Validation fails on first access to workspace_client property
    with pytest.raises(RuntimeError, match="Failed to validate databricks credentials"):
        _ = rm.workspace_client

    # Verify credentials validation was attempted
    mock_client.current_user.me.assert_called_once()


def test_databricks_rm_custom_workspace_client_invalid_credentials():
    """Test error when custom workspace client credentials validation fails on first access."""
    mock_workspace_client = MagicMock()
    mock_workspace_client.current_user.me.side_effect = Exception("Invalid credentials")

    rm = DatabricksRM(
        databricks_index_name="test_index",
        workspace_client=mock_workspace_client,
    )

    # Validation fails on first access to workspace_client property
    with pytest.raises(RuntimeError, match="Failed to validate databricks credentials"):
        _ = rm.workspace_client

    # Verify credentials validation was attempted
    mock_workspace_client.current_user.me.assert_called_once()


@patch("databricks_dspy.retrievers.databricks_rm.WorkspaceClient")
def test_databricks_rm_fallback_to_default_auth(mock_workspace_client):
    """Test fallback to default authentication when no credentials provided."""
    mock_client = MagicMock()
    mock_workspace_client.return_value = mock_client
    mock_client.current_user.me.return_value = {"userName": "test_user"}

    rm = DatabricksRM(databricks_index_name="test_index")

    assert rm.databricks_index_name == "test_index"

    # Access workspace_client property to trigger lazy creation
    client = rm.workspace_client

    # Verify WorkspaceClient was created with no auth params (default auth)
    mock_workspace_client.assert_called_once_with()
    # Verify credentials validation was performed
    mock_client.current_user.me.assert_called_once()
    assert client == mock_client
