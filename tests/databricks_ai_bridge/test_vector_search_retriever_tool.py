import json
from unittest.mock import MagicMock

import pytest
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

from databricks_ai_bridge.test_utils.vector_search import mock_workspace_client  # noqa: F401
from databricks_ai_bridge.utils.vector_search import IndexDetails
from databricks_ai_bridge.vector_search_retriever_tool import (
    FilterItem,
    VectorSearchRetrieverToolMixin,
)


class DummyVectorSearchRetrieverTool(VectorSearchRetrieverToolMixin):
    pass


index_name = "catalog.schema.index"


def make_mock_index_details(is_databricks_managed_embeddings=False, embedding_source_column=None):
    mock = MagicMock(spec=IndexDetails)
    mock.is_databricks_managed_embeddings = is_databricks_managed_embeddings
    mock.embedding_source_column = embedding_source_column or {}
    return mock


@pytest.mark.parametrize(
    "embedding_endpoint,index_details,resources",
    [
        (None, make_mock_index_details(False, {}), [DatabricksVectorSearchIndex(index_name)]),
        (
            "embedding_endpoint",
            make_mock_index_details(False, {}),
            [
                DatabricksVectorSearchIndex(index_name),
                DatabricksServingEndpoint("embedding_endpoint"),
            ],
        ),
        (
            None,
            make_mock_index_details(True, {"embedding_model_endpoint_name": "embedding_endpoint"}),
            [
                DatabricksVectorSearchIndex(index_name),
                DatabricksServingEndpoint("embedding_endpoint"),
            ],
        ),  # The following cases should not happen, but ensuring that they have reasonable behavior
        (
            "embedding_endpoint",
            make_mock_index_details(True, {"embedding_model_endpoint_name": "embedding_endpoint"}),
            [
                DatabricksVectorSearchIndex(index_name),
                DatabricksServingEndpoint("embedding_endpoint"),
            ],
        ),
        (
            "embedding_endpoint_1",
            make_mock_index_details(
                True, {"embedding_model_endpoint_name": "embedding_endpoint_2"}
            ),
            [
                DatabricksVectorSearchIndex(index_name),
                DatabricksServingEndpoint("embedding_endpoint_1"),
                DatabricksServingEndpoint("embedding_endpoint_2"),
            ],
        ),
        (None, make_mock_index_details(True, {}), [DatabricksVectorSearchIndex(index_name)]),
    ],
)
def test_get_resources(embedding_endpoint, index_details, resources):
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)
    assert tool._get_resources(index_name, embedding_endpoint, index_details) == resources


def test_describe_columns():
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)
    assert tool._describe_columns() == (
        "The vector search index includes the following columns:\n"
        "city_id (INT): No description provided\n"
        "city (STRING): Name of the city\n"
        "country (STRING): Name of the country\n"
        "description (STRING): Detailed description of the city"
    )


# =============================================================================
# Tests for _normalize_filters
# =============================================================================


def test_normalize_filters_with_filter_items():
    """Test that FilterItem list is normalized to dict."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    filters = [
        FilterItem(key="category", value="electronics"),
        FilterItem(key="price >=", value=100),
    ]

    result = tool._normalize_filters(filters)

    assert result == {"category": "electronics", "price >=": 100}


# =============================================================================
# Tests for _parse_mcp_response_to_dicts
# =============================================================================


def test_parse_mcp_response_to_dicts_json_array():
    """Test that JSON array response is parsed correctly into dicts."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    json_response = json.dumps(
        [
            {"id": "doc1", "text": "content1", "score": 0.9},
            {"id": "doc2", "text": "content2", "score": 0.8},
        ]
    )

    dicts = tool._parse_mcp_response_to_dicts(json_response)

    assert len(dicts) == 2
    assert dicts[0]["page_content"] == "content1"
    assert dicts[0]["metadata"] == {"id": "doc1", "score": 0.9}
    assert dicts[1]["page_content"] == "content2"
    assert dicts[1]["metadata"] == {"id": "doc2", "score": 0.8}


def test_parse_mcp_response_to_dicts_non_json_strict():
    """Test that non-JSON response raises ValueError when strict=True."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    plain_text_response = "This is a plain text response"

    with pytest.raises(ValueError, match="Unable to parse MCP response"):
        tool._parse_mcp_response_to_dicts(plain_text_response, strict=True)


def test_parse_mcp_response_to_dicts_non_json_non_strict():
    """Test that non-JSON response is treated as single document when strict=False."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    plain_text_response = "This is a plain text response"

    dicts = tool._parse_mcp_response_to_dicts(plain_text_response, strict=False)

    assert len(dicts) == 1
    assert dicts[0]["page_content"] == plain_text_response
    assert dicts[0]["metadata"] == {}


def test_parse_mcp_response_to_dicts_non_list_json_strict():
    """Test that non-list JSON raises ValueError when strict=True."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    json_response = json.dumps({"message": "single object response"})

    with pytest.raises(ValueError, match="Expected JSON array, got"):
        tool._parse_mcp_response_to_dicts(json_response, strict=True)


def test_parse_mcp_response_to_dicts_non_list_json_non_strict():
    """Test that non-list JSON is converted to single document when strict=False."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    json_response = json.dumps({"message": "single object response"})

    dicts = tool._parse_mcp_response_to_dicts(json_response, strict=False)

    assert len(dicts) == 1
    assert dicts[0]["page_content"] == "{'message': 'single object response'}"
    assert dicts[0]["metadata"] == {}


def test_parse_mcp_response_to_dicts_empty_list():
    """Test parsing empty list response."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    json_response = json.dumps([])

    dicts = tool._parse_mcp_response_to_dicts(json_response)

    assert dicts == []


def test_parse_mcp_response_to_dicts_custom_text_column():
    """Test that custom text column is used for page_content."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    json_response = json.dumps(
        [
            {"id": "doc1", "content": "custom content", "score": 0.9},
        ]
    )

    dicts = tool._parse_mcp_response_to_dicts(json_response, text_column="content")

    assert len(dicts) == 1
    assert dicts[0]["page_content"] == "custom content"
    assert dicts[0]["metadata"] == {"id": "doc1", "score": 0.9}


# =============================================================================
# Tests for _build_mcp_params
# =============================================================================


def test_build_mcp_params_basic():
    """Test basic MCP params building."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    params = tool._build_mcp_params(None)

    assert params["num_results"] == tool.num_results
    assert params["query_type"] == tool.query_type
    assert params["include_score"] == "false"
    assert "filters" not in params


def test_build_mcp_params_with_filters():
    """Test MCP params building with filters."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)

    filters = [FilterItem(key="category", value="electronics")]
    params = tool._build_mcp_params(filters)

    assert json.loads(params["filters"]) == {"category": "electronics"}


def test_build_mcp_params_combines_filters():
    """Test MCP params building combines predefined and runtime filters."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name, filters={"status": "active"})

    runtime_filters = [FilterItem(key="category", value="electronics")]
    params = tool._build_mcp_params(runtime_filters)

    expected_filters = {"status": "active", "category": "electronics"}
    assert json.loads(params["filters"]) == expected_filters


def test_build_mcp_params_kwargs_override_defaults():
    """Test that kwargs override default values."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name, num_results=10, query_type="ANN")

    params = tool._build_mcp_params(None, num_results=5, query_type="HYBRID")

    assert params["num_results"] == 5
    assert params["query_type"] == "HYBRID"


def test_build_mcp_params_with_columns():
    """Test MCP params building with columns."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name, columns=["id", "text", "score"])

    params = tool._build_mcp_params(None)

    assert params["columns"] == "id,text,score"


def test_build_mcp_params_with_include_score():
    """Test MCP params building with include_score=True."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name, include_score=True)

    params = tool._build_mcp_params(None)

    assert params["include_score"] == "true"


def test_build_mcp_params_k_alias_for_num_results():
    """Test that 'k' kwarg is treated as alias for num_results."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name, num_results=10)

    params = tool._build_mcp_params(None, k=3)

    assert params["num_results"] == 3


def test_build_mcp_params_with_reranker():
    """Test MCP params building with reranker."""
    from databricks.vector_search.reranker import DatabricksReranker

    reranker = DatabricksReranker(columns_to_rerank=["text", "title"])
    tool = DummyVectorSearchRetrieverTool(index_name=index_name, reranker=reranker)

    params = tool._build_mcp_params(None)

    assert params["columns_to_rerank"] == "text,title"


# =============================================================================
# Tests for _parse_index_name
# =============================================================================


def test_parse_index_name_invalid():
    """Test parsing invalid index name raises ValueError."""
    tool = DummyVectorSearchRetrieverTool(index_name="invalid_index_name")

    with pytest.raises(ValueError, match="Invalid index name format"):
        tool._parse_index_name()


# =============================================================================
# Tests for validate_filter_configuration
# =============================================================================


def test_cannot_use_both_dynamic_filter_and_predefined_filters():
    """Test that using both dynamic_filter and predefined filters raises an error."""
    # Try to initialize tool with both dynamic_filter=True and predefined filters
    with pytest.raises(
        ValueError, match="Cannot use both dynamic_filter=True and predefined filters"
    ):
        DummyVectorSearchRetrieverTool(
            index_name=index_name,
            filters={"status": "active", "category": "electronics"},
            dynamic_filter=True,
        )


# =============================================================================
# Tests for _get_tool_name
# =============================================================================


def test_get_tool_name_replaces_dots():
    """Test that dots in index name are replaced with underscores."""
    tool = DummyVectorSearchRetrieverTool(index_name="catalog.schema.my_index")
    assert tool._get_tool_name() == "catalog__schema__my_index"


def test_get_tool_name_uses_custom_name():
    """Test that custom tool_name is used when provided."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name, tool_name="custom_tool")
    assert tool._get_tool_name() == "custom_tool"


def test_get_tool_name_truncates_long_names():
    """Test that long tool names are truncated to 64 characters."""
    long_index = (
        "catalog.schema.really_really_really_long_tool_name_that_should_be_truncated_to_64_chars"
    )
    tool = DummyVectorSearchRetrieverTool(index_name=long_index)
    result = tool._get_tool_name()
    assert len(result) <= 64


# =============================================================================
# Tests for _validate_mcp_tools
# =============================================================================


def test_validate_mcp_tools_empty_list():
    """Test that empty tools list raises ValueError."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)
    with pytest.raises(ValueError, match="No MCP tools found for index"):
        tool._validate_mcp_tools([])


def test_validate_mcp_tools_multiple_tools():
    """Test that multiple tools raises ValueError."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)
    with pytest.raises(ValueError, match="Expected exactly 1 MCP tool"):
        tool._validate_mcp_tools([MagicMock(), MagicMock()])


def test_validate_mcp_tools_single_tool():
    """Test that single tool passes validation."""
    tool = DummyVectorSearchRetrieverTool(index_name=index_name)
    # Should not raise
    tool._validate_mcp_tools([MagicMock()])


# =============================================================================
# Tests for _get_filter_param_description
# =============================================================================


def test_get_filter_param_description_includes_column_metadata():
    """Test that _get_filter_param_description includes column metadata when available."""
    from unittest.mock import Mock, patch

    mock_column1 = Mock()
    mock_column1.name = "category"
    mock_column1.type_name.name = "STRING"

    mock_column2 = Mock()
    mock_column2.name = "price"
    mock_column2.type_name.name = "FLOAT"

    mock_column3 = Mock()
    mock_column3.name = "__internal_column"  # Should be excluded
    mock_column3.type_name.name = "STRING"

    mock_table_info = Mock()
    mock_table_info.columns = [mock_column1, mock_column2, mock_column3]

    with patch("databricks.sdk.WorkspaceClient") as mock_ws_client_class:
        mock_ws_client = Mock()
        mock_ws_client.tables.get.return_value = mock_table_info
        mock_ws_client_class.return_value = mock_ws_client

        tool = DummyVectorSearchRetrieverTool(index_name=index_name)
        description = tool._get_filter_param_description()

        # Should include available columns in description
        assert "Available columns for filtering: category (STRING), price (FLOAT)" in description

        # Should include comprehensive filter syntax
        assert "Inclusion:" in description
        assert "Exclusion:" in description
        assert "Comparisons:" in description
        assert "Pattern match:" in description
        assert "OR logic:" in description


def test_get_filter_param_description_fails_on_table_metadata_error():
    """Test that _get_filter_param_description fails with clear error when table metadata cannot be retrieved."""
    from unittest.mock import patch

    with patch("databricks.sdk.WorkspaceClient") as mock_ws_client_class:
        mock_ws_client = MagicMock()
        mock_ws_client.tables.get.side_effect = Exception("Permission denied")
        mock_ws_client_class.return_value = mock_ws_client

        tool = DummyVectorSearchRetrieverTool(index_name=index_name)

        with pytest.raises(
            ValueError,
            match="Failed to retrieve table metadata for index.*Permission denied",
        ):
            tool._get_filter_param_description()


def test_get_filter_param_description_fails_on_empty_columns():
    """Test that _get_filter_param_description fails when table has no valid columns."""
    from unittest.mock import patch

    with patch("databricks.sdk.WorkspaceClient") as mock_ws_client_class:
        mock_ws_client = MagicMock()
        mock_table = MagicMock()
        mock_column = MagicMock()
        mock_column.name = "__internal_column"
        mock_column.type_name = MagicMock()
        mock_column.type_name.name = "STRING"
        mock_table.columns = [mock_column]
        mock_ws_client.tables.get.return_value = mock_table
        mock_ws_client_class.return_value = mock_ws_client

        tool = DummyVectorSearchRetrieverTool(index_name=index_name)

        with pytest.raises(
            ValueError,
            match="No valid columns found in table metadata for index",
        ):
            tool._get_filter_param_description()
