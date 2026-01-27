import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import mlflow
import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import ModelServingUserCredentials
from databricks.vector_search.utils import CredentialStrategy
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DELTA_SYNC_INDEX,
    DELTA_SYNC_INDEX_EMBEDDING_MODEL_ENDPOINT_NAME,
    DIRECT_ACCESS_INDEX,
    INPUT_TEXTS,
    _get_index,
    mock_vs_client,
    mock_workspace_client,
)
from databricks_ai_bridge.vector_search_retriever_tool import FilterItem
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
)
from tests.utils.chat_models import llm, mock_client  # noqa: F401
from tests.utils.vector_search import (
    EMBEDDING_MODEL,
    embeddings,  # noqa: F401
)
from tests.utils.vector_search import (
    mock_client as mock_embeddings_client,  # noqa: F401
)


def _create_mcp_response_json(texts: List[str] = None) -> str:
    """Create a mock MCP response in JSON format."""
    texts = texts or INPUT_TEXTS
    return json.dumps(
        [
            {"id": str(uuid.uuid4()), "text": text, "score": 0.85 - (i * 0.1)}
            for i, text in enumerate(texts)
        ]
    )


def assert_mcp_tool_called_with(mock_tool, expected_args: Dict[str, Any]):
    """Assert MCP tool was called with expected args, handling JSON-stringified filters."""
    mock_tool.invoke.assert_called_once()
    call_args = mock_tool.invoke.call_args[0][0]
    for key, value in expected_args.items():
        if key == "filters":
            assert json.loads(call_args["filters"]) == value
        else:
            assert call_args[key] == value


@pytest.fixture
def mock_mcp_infrastructure():
    """Mock MCP infrastructure for tests that need it."""
    # Create mock MCP tool that returns JSON response
    mock_tool = MagicMock()
    mock_tool.invoke = MagicMock(return_value=_create_mcp_response_json())

    # Create mock MCP client
    mock_client_instance = MagicMock()
    mock_client_instance.get_tools = AsyncMock(return_value=[mock_tool])

    # Create mock MCP server
    mock_server_instance = MagicMock()

    with (
        patch(
            "databricks_langchain.vector_search_retriever_tool.DatabricksMultiServerMCPClient"
        ) as mock_client_class,
        patch(
            "databricks_langchain.vector_search_retriever_tool.DatabricksMCPServer"
        ) as mock_server_class,
    ):
        mock_client_class.return_value = mock_client_instance
        mock_server_class.from_vector_search.return_value = mock_server_instance
        yield {
            "client_class": mock_client_class,
            "client_instance": mock_client_instance,
            "server_class": mock_server_class,
            "server_instance": mock_server_instance,
            "tool": mock_tool,
        }


@pytest.fixture(params=["mcp", "direct_api"])
def execution_path(request, mock_mcp_infrastructure):
    """Parametrized fixture that sets up mocks for MCP or Direct API path."""
    if request.param == "mcp":
        yield {
            "path": "mcp",
            "index_name": DELTA_SYNC_INDEX,
            "mock_tool": mock_mcp_infrastructure["tool"],
            "mock_mcp": mock_mcp_infrastructure,
        }
    else:
        # For direct API, use an index that requires self-managed embeddings
        yield {
            "path": "direct_api",
            "index_name": DIRECT_ACCESS_INDEX,
            "mock_tool": None,
            "mock_mcp": mock_mcp_infrastructure,
        }


def setup_tool_for_path(execution_path, tool):
    """Set up mock for the tool based on execution path."""
    if execution_path["path"] == "direct_api":
        tool._vector_store.similarity_search = MagicMock(return_value=[])


def init_vector_search_tool(
    index_name: str,
    columns: Optional[List[str]] = None,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    embedding: Optional[Embeddings] = None,
    text_column: Optional[str] = None,
    doc_uri: Optional[str] = None,
    primary_key: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> VectorSearchRetrieverTool:
    kwargs.update(
        {
            "index_name": index_name,
            "columns": columns,
            "tool_name": tool_name,
            "tool_description": tool_description,
            "embedding": embedding,
            "text_column": text_column,
            "doc_uri": doc_uri,
            "primary_key": primary_key,
            "filters": filters,
        }
    )
    if index_name != DELTA_SYNC_INDEX:
        kwargs.update(
            {
                "embedding": EMBEDDING_MODEL,
                "text_column": "text",
            }
        )
    return VectorSearchRetrieverTool(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_init(index_name: str) -> None:
    vector_search_tool = init_vector_search_tool(index_name)
    assert isinstance(vector_search_tool, BaseTool)
    assert "'additionalProperties': true" not in str(vector_search_tool.args)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_chat_model_bind_tools(llm: ChatDatabricks, index_name: str) -> None:
    from langchain_core.messages import AIMessage

    vector_search_tool = init_vector_search_tool(index_name)
    llm_with_tools = llm.bind_tools([vector_search_tool])
    response = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
    assert isinstance(response, AIMessage)


def test_filters_are_passed_through(execution_path) -> None:
    """Test filters are passed through correctly on both paths."""
    tool = init_vector_search_tool(execution_path["index_name"])
    setup_tool_for_path(execution_path, tool)

    tool.invoke(
        {
            "query": "what cities are in Germany",
            "filters": [FilterItem(key="country", value="Germany")],
        }
    )

    if execution_path["path"] == "mcp":
        assert_mcp_tool_called_with(
            execution_path["mock_tool"],
            {"query": "what cities are in Germany", "filters": {"country": "Germany"}},
        )
    else:
        tool._vector_store.similarity_search.assert_called_once_with(
            query="what cities are in Germany",
            k=tool.num_results,
            filter={"country": "Germany"},
            query_type=tool.query_type,
        )


def test_filters_are_combined(execution_path) -> None:
    """Test filters are combined correctly (predefined + runtime) on both paths."""
    tool = init_vector_search_tool(execution_path["index_name"], filters={"city LIKE": "Berlin"})
    setup_tool_for_path(execution_path, tool)

    tool.invoke(
        {
            "query": "what cities are in Germany",
            "filters": [FilterItem(key="country", value="Germany")],
        }
    )

    expected_filters = {"city LIKE": "Berlin", "country": "Germany"}
    if execution_path["path"] == "mcp":
        assert_mcp_tool_called_with(
            execution_path["mock_tool"],
            {"query": "what cities are in Germany", "filters": expected_filters},
        )
    else:
        tool._vector_store.similarity_search.assert_called_once_with(
            query="what cities are in Germany",
            k=tool.num_results,
            filter=expected_filters,
            query_type=tool.query_type,
        )


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
@pytest.mark.parametrize("columns", [None, ["id", "text"]])
@pytest.mark.parametrize("tool_name", [None, "test_tool"])
@pytest.mark.parametrize("tool_description", [None, "Test tool for vector search"])
@pytest.mark.parametrize("embedding", [None, EMBEDDING_MODEL])
@pytest.mark.parametrize("text_column", [None, "text"])
def test_vector_search_retriever_tool_combinations(
    mock_mcp_infrastructure,
    index_name: str,
    columns: Optional[List[str]],
    tool_name: Optional[str],
    tool_description: Optional[str],
    embedding: Optional[Any],
    text_column: Optional[str],
) -> None:
    if index_name == DELTA_SYNC_INDEX:
        embedding = None
        text_column = None

    vector_search_tool = init_vector_search_tool(
        index_name=index_name,
        columns=columns,
        tool_name=tool_name,
        tool_description=tool_description,
        embedding=embedding,
        text_column=text_column,
    )
    assert isinstance(vector_search_tool, BaseTool)
    result = vector_search_tool.invoke("Databricks Agent Framework")
    assert result is not None


def test_vector_search_retriever_tool_doc_uri_primary_key(mock_mcp_infrastructure) -> None:
    """Test that doc_uri and primary_key work correctly with MCP path."""
    vector_search_tool = init_vector_search_tool(
        index_name=DELTA_SYNC_INDEX,
        doc_uri="uri",
        primary_key="id",
    )
    assert isinstance(vector_search_tool, BaseTool)
    result = vector_search_tool.invoke("Databricks Agent Framework")
    # With MCP path, results are parsed from mock JSON response
    assert result is not None
    assert len(result) > 0
    assert all(isinstance(doc, Document) for doc in result)
    # Verify Documents have expected structure from mock response
    assert all(doc.page_content for doc in result)
    assert all("id" in doc.metadata for doc in result)


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_vector_search_retriever_tool_description_generation(index_name: str) -> None:
    vector_search_tool = init_vector_search_tool(index_name)
    assert vector_search_tool.name != ""
    assert vector_search_tool.description != ""
    assert vector_search_tool.name == index_name.replace(".", "__")
    assert (
        "A vector search-based retrieval tool for querying indexed embeddings."
        in vector_search_tool.description
    )
    assert vector_search_tool.args_schema.model_fields["query"] is not None
    assert vector_search_tool.args_schema.model_fields["query"].description == (
        "The string used to query the index with and identify the most similar "
        "vectors and return the associated documents."
    )


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
@pytest.mark.parametrize("tool_name", [None, "test_tool"])
def test_vs_tool_tracing(
    mock_mcp_infrastructure, index_name: str, tool_name: Optional[str]
) -> None:
    vector_search_tool = init_vector_search_tool(index_name, tool_name=tool_name)
    vector_search_tool._run("Databricks Agent Framework")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    spans = trace.search_spans(name=tool_name or index_name, span_type=SpanType.RETRIEVER)
    assert len(spans) == 1
    inputs = json.loads(trace.to_dict()["data"]["spans"][0]["attributes"]["mlflow.spanInputs"])
    assert inputs["query"] == "Databricks Agent Framework"
    outputs = json.loads(trace.to_dict()["data"]["spans"][0]["attributes"]["mlflow.spanOutputs"])
    # Verify outputs are Documents with page_content
    assert len(outputs) > 0
    assert all("page_content" in d for d in outputs)
    assert all(d["page_content"] for d in outputs)  # page_content is not empty


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_vector_search_retriever_tool_resources(
    mock_embeddings_client,
    embeddings,
    index_name: str,
) -> None:
    text_column = "text"
    if index_name == DELTA_SYNC_INDEX:
        embeddings = None
        text_column = None

    vector_search_tool = VectorSearchRetrieverTool(
        index_name=index_name, embedding=embeddings, text_column=text_column
    )
    expected_resources = (
        [DatabricksVectorSearchIndex(index_name=index_name)]
        + ([DatabricksServingEndpoint(endpoint_name=embeddings.endpoint)] if embeddings else [])
        + (
            [
                DatabricksServingEndpoint(
                    endpoint_name=DELTA_SYNC_INDEX_EMBEDDING_MODEL_ENDPOINT_NAME
                )
            ]
            if index_name == DELTA_SYNC_INDEX
            else []
        )
    )
    assert [res.to_dict() for res in vector_search_tool.resources] == [
        res.to_dict() for res in expected_resources
    ]


@pytest.mark.parametrize("tool_name", [None, "valid_tool_name", "test_tool"])
def test_tool_name_validation_valid(tool_name: Optional[str]) -> None:
    index_name = "catalog.schema.index"
    tool = init_vector_search_tool(index_name, tool_name=tool_name)
    assert tool.tool_name == tool_name
    if tool_name:
        assert tool.name == tool_name


@pytest.mark.parametrize("tool_name", ["test.tool.name", "tool&name"])
def test_tool_name_validation_invalid(tool_name: str) -> None:
    index_name = "catalog.schema.index"
    with pytest.raises(ValueError):
        init_vector_search_tool(index_name, tool_name=tool_name)


@pytest.mark.parametrize(
    "index_name,name",
    [
        ("catalog.schema.index", "catalog__schema__index"),
        ("cata_log.schema_.index", "cata_log__schema___index"),
    ],
)
def test_index_name_to_tool_name(index_name: str, name: str) -> None:
    vector_search_tool = init_vector_search_tool(index_name)
    assert vector_search_tool.name == name


def test_vector_search_client_model_serving_environment():
    with patch("os.path.isfile", return_value=True):
        # Simulate Model Serving Environment
        os.environ["IS_IN_DB_MODEL_SERVING_ENV"] = "true"

        # Fake credential token
        current_thread = threading.current_thread()
        thread_data = current_thread.__dict__
        thread_data["invokers_token"] = "abc"

        w = WorkspaceClient(
            host="testDogfod.com", credentials_strategy=ModelServingUserCredentials()
        )

        with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
            mock_instance = mockVSClient.return_value
            mock_instance.get_index.side_effect = _get_index
            with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
                vsTool = VectorSearchRetrieverTool(
                    index_name="test.delta_sync.index",
                    tool_description="desc",
                    workspace_client=w,
                )
                mockVSClient.assert_called_once_with(
                    disable_notice=True,
                    credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS,
                )


def test_vector_search_client_non_model_serving_environment():
    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        mock_instance = mockVSClient.return_value
        mock_instance.get_index.side_effect = _get_index
        vsTool = VectorSearchRetrieverTool(
            index_name="test.delta_sync.index",
            tool_description="desc",
        )
        mockVSClient.assert_called_once_with(disable_notice=True)


def test_vector_search_client_with_pat_workspace_client():
    w = WorkspaceClient(host="testDogfod.com", token="fakeToken")
    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
            mock_instance = mockVSClient.return_value
            mock_instance.get_index.side_effect = _get_index
            VectorSearchRetrieverTool(
                index_name="test.delta_sync.index",
                tool_description="desc",
                workspace_client=w,
            )
            mockVSClient.assert_called_once_with(
                disable_notice=True, personal_access_token="fakeToken"
            )


def test_vector_search_client_with_sp_workspace_client():
    # Create a proper mock workspace client that passes isinstance check
    w = create_autospec(WorkspaceClient, instance=True)
    w.config.auth_type = "oauth-m2m"
    w.config.host = "testDogfod.com"
    w.config.client_id = "fakeClientId"
    w.config.client_secret = "fakeClientSecret"

    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
            mock_instance = mockVSClient.return_value
            mock_instance.get_index.side_effect = _get_index
            VectorSearchRetrieverTool(
                index_name="test.delta_sync.index",
                tool_description="desc",
                workspace_client=w,
            )
            mockVSClient.assert_called_once_with(
                disable_notice=True,
                workspace_url="testDogfod.com",
                service_principal_client_id="fakeClientId",
                service_principal_client_secret="fakeClientSecret",
            )


def test_kwargs_are_passed_through(execution_path) -> None:
    """Test kwargs are passed through correctly on both paths."""
    tool = init_vector_search_tool(execution_path["index_name"], score_threshold=0.5)
    setup_tool_for_path(execution_path, tool)

    tool.invoke({"query": "what cities are in Germany"})

    if execution_path["path"] == "mcp":
        assert_mcp_tool_called_with(
            execution_path["mock_tool"],
            {"query": "what cities are in Germany", "score_threshold": 0.5},
        )
    else:
        tool._vector_store.similarity_search.assert_called_once_with(
            query="what cities are in Germany",
            k=tool.num_results,
            filter={},
            query_type=tool.query_type,
            score_threshold=0.5,
        )


def test_kwargs_override_both_num_results_and_query_type(execution_path) -> None:
    """Test kwargs can override num_results and query_type on both paths."""
    tool = init_vector_search_tool(execution_path["index_name"], num_results=10, query_type="ANN")
    setup_tool_for_path(execution_path, tool)

    tool.invoke({"query": "what cities are in Germany", "k": 3, "query_type": "HYBRID"})

    if execution_path["path"] == "mcp":
        assert_mcp_tool_called_with(
            execution_path["mock_tool"],
            {"query": "what cities are in Germany", "num_results": 3, "query_type": "HYBRID"},
        )
    else:
        tool._vector_store.similarity_search.assert_called_once_with(
            query="what cities are in Germany",
            k=3,
            filter={},
            query_type="HYBRID",
        )


def test_enhanced_filter_description_with_column_metadata() -> None:
    """Test that the tool args_schema includes enhanced filter descriptions with column metadata."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, dynamic_filter=True)

    # The LangChain implementation calls index.describe() to get column information
    # and includes them in the filter description
    args_schema = vector_search_tool.args_schema
    filter_field = args_schema.model_fields["filters"]

    # Check that the filter description is enhanced with available columns
    # Note: The actual columns will depend on the mocked index.describe() response
    assert (
        "Available columns for filtering:" in filter_field.description
        or "Optional filters" in filter_field.description
    )

    # Should include comprehensive filter syntax
    assert "Inclusion:" in filter_field.description
    assert "Exclusion:" in filter_field.description
    assert "Comparisons:" in filter_field.description
    assert "Pattern match:" in filter_field.description
    assert "OR logic:" in filter_field.description

    # Should include examples
    assert "Examples:" in filter_field.description
    assert "Filter by category:" in filter_field.description
    assert "Filter by price range:" in filter_field.description


def test_enhanced_filter_description_fails_on_table_metadata_error() -> None:
    """Test that tool initialization fails with clear error when table metadata cannot be retrieved."""
    # Mock WorkspaceClient to raise an exception when accessing table metadata
    with patch("databricks.sdk.WorkspaceClient") as mock_ws_client_class:
        mock_ws_client = MagicMock()
        mock_ws_client.tables.get.side_effect = Exception("Permission denied")
        mock_ws_client_class.return_value = mock_ws_client

        # Try to initialize tool with dynamic_filter=True
        # This should fail because we can't get table metadata
        with pytest.raises(
            ValueError,
            match="Failed to retrieve table metadata for index.*Permission denied",
        ):
            init_vector_search_tool(DELTA_SYNC_INDEX, dynamic_filter=True)


def test_enhanced_filter_description_fails_on_empty_columns() -> None:
    """Test that tool initialization fails when table has no valid columns."""
    # Mock WorkspaceClient to return a table with no valid columns (all start with __)
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

        # Try to initialize tool with dynamic_filter=True
        # This should fail because there are no valid columns
        with pytest.raises(
            ValueError,
            match="No valid columns found in table metadata for index",
        ):
            init_vector_search_tool(DELTA_SYNC_INDEX, dynamic_filter=True)


def test_cannot_use_both_dynamic_filter_and_predefined_filters() -> None:
    """Test that using both dynamic_filter and predefined filters raises an error."""
    # Try to initialize tool with both dynamic_filter=True and predefined filters
    with pytest.raises(
        ValueError, match="Cannot use both dynamic_filter=True and predefined filters"
    ):
        init_vector_search_tool(
            DELTA_SYNC_INDEX,
            filters={"status": "active", "category": "electronics"},
            dynamic_filter=True,
        )


def test_predefined_filters_work_without_dynamic_filter(execution_path) -> None:
    """Test that predefined filters work correctly when dynamic_filter is False on both paths."""
    tool = init_vector_search_tool(
        execution_path["index_name"], filters={"status": "active", "category": "electronics"}
    )
    setup_tool_for_path(execution_path, tool)

    # The filters parameter should NOT be exposed since dynamic_filter=False
    args_schema = tool.args_schema
    assert "filters" not in args_schema.model_fields

    tool.invoke({"query": "what electronics are available"})

    expected_filters = {"status": "active", "category": "electronics"}
    if execution_path["path"] == "mcp":
        assert_mcp_tool_called_with(
            execution_path["mock_tool"],
            {"query": "what electronics are available", "filters": expected_filters},
        )
    else:
        tool._vector_store.similarity_search.assert_called_once_with(
            query="what electronics are available",
            k=tool.num_results,
            filter=expected_filters,
            query_type=tool.query_type,
        )


def test_filter_item_serialization(execution_path) -> None:
    """Test that FilterItem objects are properly converted to dictionaries on both paths."""
    tool = init_vector_search_tool(execution_path["index_name"])
    setup_tool_for_path(execution_path, tool)

    # Test various filter types
    filters = [
        FilterItem(key="category", value="electronics"),
        FilterItem(key="price >=", value=100),
        FilterItem(key="status NOT", value="discontinued"),
        FilterItem(key="tags", value=["wireless", "bluetooth"]),
    ]

    tool.invoke({"query": "find products", "filters": filters})

    expected_filters = {
        "category": "electronics",
        "price >=": 100,
        "status NOT": "discontinued",
        "tags": ["wireless", "bluetooth"],
    }

    if execution_path["path"] == "mcp":
        assert_mcp_tool_called_with(
            execution_path["mock_tool"],
            {"query": "find products", "filters": expected_filters},
        )
    else:
        tool._vector_store.similarity_search.assert_called_once_with(
            query="find products",
            k=tool.num_results,
            filter=expected_filters,
            query_type=tool.query_type,
        )


# =============================================================================
# MCP Path Specific Tests
# =============================================================================


def test_mcp_path_is_used_for_databricks_managed_embeddings(mock_mcp_infrastructure) -> None:
    """Test that MCP path is used for Databricks-managed embeddings indexes."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)

    # Invoke the tool (should use MCP path for DELTA_SYNC_INDEX which has managed embeddings)
    result = vector_search_tool._run("test query")

    # Verify MCP server was created with correct parameters
    mock_mcp_infrastructure["server_class"].from_vector_search.assert_called_once()
    call_kwargs = mock_mcp_infrastructure["server_class"].from_vector_search.call_args[1]
    assert call_kwargs["catalog"] == "test"
    assert call_kwargs["schema"] == "delta_sync"
    assert call_kwargs["index_name"] == "index"

    # Verify MCP client was used
    mock_mcp_infrastructure["client_class"].assert_called_once()

    # Verify MCP tool was invoked with expected query
    mock_mcp_infrastructure["tool"].invoke.assert_called_once_with(
        {
            "query": "test query",
            "num_results": vector_search_tool.num_results,
            "query_type": vector_search_tool.query_type,
            "include_score": "false",
        }
    )


def test_direct_api_path_is_used_for_self_managed_embeddings(mock_mcp_infrastructure) -> None:
    """Test that direct API path is used for self-managed embeddings indexes."""
    # Use an index that requires self-managed embeddings
    index_name = "test.direct_access.index"
    vector_search_tool = init_vector_search_tool(index_name)
    vector_search_tool._vector_store.similarity_search = MagicMock(return_value=[])

    # Invoke the tool (should use direct API path)
    result = vector_search_tool._run("test query")

    # Verify similarity_search was called directly
    vector_search_tool._vector_store.similarity_search.assert_called_once()

    # Verify MCP was NOT used for self-managed embeddings
    mock_mcp_infrastructure["tool"].invoke.assert_not_called()


def test_mcp_tool_is_cached(mock_mcp_infrastructure) -> None:
    """Test that MCP tool is cached and not recreated on subsequent calls."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)

    # Call _run multiple times
    vector_search_tool._run("query 1")
    vector_search_tool._run("query 2")
    vector_search_tool._run("query 3")

    # MCP server should only be created once
    assert mock_mcp_infrastructure["server_class"].from_vector_search.call_count == 1

    # MCP client should only be created once
    assert mock_mcp_infrastructure["client_class"].call_count == 1

    # But MCP tool should be invoked 3 times
    assert mock_mcp_infrastructure["tool"].invoke.call_count == 3
