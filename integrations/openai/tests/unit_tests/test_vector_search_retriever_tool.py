import json
import os
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, create_autospec, patch

import mlflow
import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import ModelServingUserCredentials
from databricks.vector_search.reranker import DatabricksReranker, Reranker
from databricks.vector_search.utils import CredentialStrategy
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DELTA_SYNC_INDEX,
    DELTA_SYNC_INDEX_EMBEDDING_MODEL_ENDPOINT_NAME,
    DIRECT_ACCESS_INDEX,
    INPUT_TEXTS,
    mock_vs_client,
    mock_workspace_client,
)
from databricks_ai_bridge.vector_search_retriever_tool import FilterItem
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call_param import Function
from pydantic import BaseModel

from databricks_openai import VectorSearchRetrieverTool


@pytest.fixture(autouse=True)
def mock_openai_client():
    mock_client = MagicMock()
    mock_client.api_key = "fake_api_key"
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3, 0.4])]
    mock_client.embeddings.create.return_value = mock_response
    with patch("openai.OpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_mcp_toolkit():
    """Mock McpServerToolkit for testing MCP path."""
    import uuid

    # Create mock response in MCP format (flat JSON with all columns)
    def create_mcp_response(**kwargs):
        mcp_response_data = [
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "score": 0.85 - (i * 0.1),  # Decreasing scores like real results
            }
            for i, text in enumerate(INPUT_TEXTS)
        ]
        return json.dumps(mcp_response_data)

    # Create mock tool with execute method
    mock_tool = MagicMock()
    mock_tool.execute = MagicMock(side_effect=create_mcp_response)

    # Create mock toolkit instance
    mock_toolkit_instance = MagicMock()
    mock_toolkit_instance.get_tools.return_value = [mock_tool]

    # Mock the from_vector_search factory method
    with patch(
        "databricks_openai.vector_search_retriever_tool.McpServerToolkit"
    ) as mock_toolkit_class:
        mock_toolkit_class.from_vector_search.return_value = mock_toolkit_instance
        yield mock_toolkit_instance


@pytest.fixture(params=["mcp", "direct_api"])
def execution_path(request, mock_mcp_toolkit):
    """Parametrized fixture that sets up mocks for MCP or Direct API path."""
    if request.param == "mcp":
        yield {
            "path": "mcp",
            "index_name": DELTA_SYNC_INDEX,
            "mock_tool": mock_mcp_toolkit.get_tools.return_value[0],
        }
    else:
        # For direct API, we need to mock _index on the tool after creation
        yield {
            "path": "direct_api",
            "index_name": DIRECT_ACCESS_INDEX,
            "mock_tool": None,
        }


def setup_tool_for_path(execution_path, tool):
    """Set up mock for the tool based on execution path."""
    from databricks.vector_search.client import VectorSearchIndex

    if execution_path["path"] == "direct_api":
        tool._index = create_autospec(VectorSearchIndex, instance=True)


def get_chat_completion_response(tool_name: str, index_name: str):
    return ChatCompletion(
        id="chatcmpl-AlSTQf3qIjeEOdoagPXUYhuWZkwme",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=None,
                    refusal=None,
                    role="assistant",
                    audio=None,
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_VtmBTsVM2zQ3yL5GzddMgWb0",
                            function=Function(
                                arguments='{"query":"Databricks Agent Framework"}',
                                name=tool_name
                                or index_name.replace(
                                    ".", "__"
                                ),  # see get_tool_name() in VectorSearchRetrieverTool
                            ),
                            type="function",
                        )
                    ],
                ),
            )
        ],
        created=1735874232,
        model="gpt-4o-mini-2024-07-18",
        object="chat.completion",
    )


def init_vector_search_tool(
    index_name: str,
    columns: Optional[List[str]] = None,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    text_column: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    reranker: Optional[Reranker] = None,
    **kwargs: Any,
) -> VectorSearchRetrieverTool:
    kwargs.update(
        {
            "index_name": index_name,
            "columns": columns,
            "tool_name": tool_name,
            "tool_description": tool_description,
            "text_column": text_column,
            "embedding_model_name": embedding_model_name,
            "filters": filters,
            "reranker": reranker,
        }
    )
    if index_name != DELTA_SYNC_INDEX:
        kwargs.update(
            {
                "text_column": "text",
                "embedding_model_name": "text-embedding-3-small",
            }
        )
    return VectorSearchRetrieverTool(**kwargs)  # type: ignore[arg-type]


class SelfManagedEmbeddingsTest:
    def __init__(self, text_column=None, embedding_model_name=None, open_ai_client=None):
        self.text_column = text_column
        self.embedding_model_name = embedding_model_name
        self.open_ai_client = open_ai_client


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
@pytest.mark.parametrize("columns", [None, ["id", "text"]])
@pytest.mark.parametrize("tool_name", [None, "test_tool"])
@pytest.mark.parametrize("tool_description", [None, "Test tool for vector search"])
def test_vector_search_retriever_tool_init(
    mock_mcp_toolkit,
    index_name: str,
    columns: Optional[List[str]],
    tool_name: Optional[str],
    tool_description: Optional[str],
) -> None:
    if index_name == DELTA_SYNC_INDEX:
        self_managed_embeddings_test = SelfManagedEmbeddingsTest()
    else:
        from openai import OpenAI

        self_managed_embeddings_test = SelfManagedEmbeddingsTest(
            "text", "text-embedding-3-small", OpenAI(api_key="your-api-key")
        )

    vector_search_tool = init_vector_search_tool(
        index_name=index_name,
        columns=columns,
        tool_name=tool_name,
        tool_description=tool_description,
        text_column=self_managed_embeddings_test.text_column,
        embedding_model_name=self_managed_embeddings_test.embedding_model_name,
    )
    assert isinstance(vector_search_tool, BaseModel)

    expected_resources = (
        [DatabricksVectorSearchIndex(index_name=index_name)]
        + (
            [DatabricksServingEndpoint(endpoint_name="text-embedding-3-small")]
            if self_managed_embeddings_test.embedding_model_name
            else []
        )
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

    # simulate call to openai.chat.completions.create
    chat_completion_resp = get_chat_completion_response(tool_name, index_name)
    tool_call = chat_completion_resp.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    docs = vector_search_tool.execute(query=args["query"])
    assert docs is not None
    assert len(docs) == len(INPUT_TEXTS)

    assert sorted([d["page_content"] for d in docs]) == sorted(INPUT_TEXTS)
    assert all(["id" in d["metadata"] for d in docs])

    # Ensure tracing works properly
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    spans = trace.search_spans(name=tool_name or index_name, span_type=SpanType.RETRIEVER)
    assert len(spans) == 1
    inputs = json.loads(trace.to_dict()["data"]["spans"][0]["attributes"]["mlflow.spanInputs"])
    assert inputs["query"] == "Databricks Agent Framework"
    outputs = json.loads(trace.to_dict()["data"]["spans"][0]["attributes"]["mlflow.spanOutputs"])
    assert all([d["page_content"] in INPUT_TEXTS for d in outputs])

    # Ensure that there aren't additional properties (not compatible with llama)
    assert "'additionalProperties': True" not in str(vector_search_tool.tool)


@pytest.mark.parametrize("columns", [None, ["id", "text"]])
@pytest.mark.parametrize("tool_name", [None, "test_tool"])
@pytest.mark.parametrize("tool_description", [None, "Test tool for vector search"])
def test_open_ai_client_from_env(
    columns: Optional[List[str]], tool_name: Optional[str], tool_description: Optional[str]
) -> None:
    self_managed_embeddings_test = SelfManagedEmbeddingsTest("text", "text-embedding-3-small", None)
    os.environ["OPENAI_API_KEY"] = "your-api-key"

    vector_search_tool = init_vector_search_tool(
        index_name=DIRECT_ACCESS_INDEX,
        columns=columns,
        tool_name=tool_name,
        tool_description=tool_description,
        text_column=self_managed_embeddings_test.text_column,
        embedding_model_name=self_managed_embeddings_test.embedding_model_name,
    )
    assert isinstance(vector_search_tool, BaseModel)
    # simulate call to openai.chat.completions.create
    chat_completion_resp = get_chat_completion_response(tool_name, DIRECT_ACCESS_INDEX)
    tool_call = chat_completion_resp.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    docs = vector_search_tool.execute(
        query=args["query"], openai_client=self_managed_embeddings_test.open_ai_client
    )
    assert docs is not None
    assert len(docs) == len(INPUT_TEXTS)
    assert sorted([d["page_content"] for d in docs]) == sorted(INPUT_TEXTS)
    assert all(["id" in d["metadata"] for d in docs])


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
def test_vector_search_retriever_index_name_rewrite(
    index_name: str,
) -> None:
    if index_name == DELTA_SYNC_INDEX:
        self_managed_embeddings_test = SelfManagedEmbeddingsTest()
    else:
        from openai import OpenAI

        self_managed_embeddings_test = SelfManagedEmbeddingsTest(
            "text", "text-embedding-3-small", OpenAI(api_key="your-api-key")
        )

    vector_search_tool = init_vector_search_tool(
        index_name=index_name,
        text_column=self_managed_embeddings_test.text_column,
        embedding_model_name=self_managed_embeddings_test.embedding_model_name,
    )
    assert vector_search_tool.tool["function"]["name"] == index_name.replace(".", "__")


@pytest.mark.parametrize(
    "index_name",
    ["catalog.schema.really_really_really_long_tool_name_that_should_be_truncated_to_64_chars"],
)
def test_vector_search_retriever_long_index_name(
    index_name: str,
) -> None:
    vector_search_tool = init_vector_search_tool(index_name=index_name)
    assert len(vector_search_tool.tool["function"]["name"]) <= 64


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
            with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
                vsTool = VectorSearchRetrieverTool(
                    index_name="catalog.schema.my_index_name",
                    text_column="abc",
                    embedding_model_name="text-embedding-3-small",
                    tool_description="desc",
                    workspace_client=w,
                )
                mockVSClient.assert_called_once_with(
                    disable_notice=True,
                    credential_strategy=CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS,
                )


def test_vector_search_client_non_model_serving_environment():
    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        vsTool = VectorSearchRetrieverTool(
            index_name="catalog.schema.my_index_name",
            text_column="abc",
            embedding_model_name="text-embedding-3-small",
            tool_description="desc",
        )
        mockVSClient.assert_called_once_with(disable_notice=True)


def test_vector_search_client_with_pat_workspace_client():
    w = WorkspaceClient(host="testDogfod.com", token="fakeToken")
    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
            VectorSearchRetrieverTool(
                index_name="catalog.schema.my_index_name",
                text_column="abc",
                embedding_model_name="text-embedding-3-small",
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
            VectorSearchRetrieverTool(
                index_name="catalog.schema.my_index_name",
                text_column="abc",
                embedding_model_name="text-embedding-3-small",
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
    vector_search_tool = init_vector_search_tool(execution_path["index_name"], score_threshold=0.5)
    setup_tool_for_path(execution_path, vector_search_tool)

    # extra_param is ignored because it's not supported
    vector_search_tool.execute(
        query="what cities are in Germany", debug_level=2, extra_param="something random"
    )

    if execution_path["path"] == "mcp":
        mock_tool = execution_path["mock_tool"]
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs

        assert call_kwargs["query"] == "what cities are in Germany"

        meta = call_kwargs["_meta"]
        assert meta["num_results"] == vector_search_tool.num_results
        assert meta["columns"] == ",".join(vector_search_tool.columns)
        assert meta["score_threshold"] == 0.5
        assert meta["query_type"] == vector_search_tool.query_type
        assert "filters" not in meta
        assert "columns_to_rerank" not in meta
        assert "debug_level" not in meta
        assert "extra_param" not in meta
    else:
        vector_search_tool._index.similarity_search.assert_called_once()
        call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs

        assert call_kwargs["num_results"] == vector_search_tool.num_results
        assert call_kwargs["columns"] == vector_search_tool.columns
        assert call_kwargs["score_threshold"] == 0.5
        assert call_kwargs["query_type"] == vector_search_tool.query_type
        assert call_kwargs["filters"] == {}
        # debug_level should be passed through for direct API (if supported by signature)
        assert call_kwargs.get("debug_level") == 2


def test_filters_are_passed_through(execution_path) -> None:
    vector_search_tool = init_vector_search_tool(execution_path["index_name"])
    setup_tool_for_path(execution_path, vector_search_tool)

    vector_search_tool.execute(
        query="what cities are in Germany",
        filters=[FilterItem(key="country", value="Germany")],
    )

    if execution_path["path"] == "mcp":
        mock_tool = execution_path["mock_tool"]
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs
        meta = call_kwargs["_meta"]

        assert call_kwargs["query"] == "what cities are in Germany"
        assert json.loads(meta["filters"]) == {"country": "Germany"}
        assert meta["num_results"] == vector_search_tool.num_results
        assert meta["query_type"] == vector_search_tool.query_type
        assert meta["columns"] == ",".join(vector_search_tool.columns)
    else:
        vector_search_tool._index.similarity_search.assert_called_once()
        call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs

        assert call_kwargs["filters"] == {"country": "Germany"}
        assert call_kwargs["num_results"] == vector_search_tool.num_results
        assert call_kwargs["query_type"] == vector_search_tool.query_type
        assert call_kwargs["columns"] == vector_search_tool.columns


def test_filters_are_combined(execution_path) -> None:
    vector_search_tool = init_vector_search_tool(
        execution_path["index_name"], filters={"city LIKE": "Berlin"}
    )
    setup_tool_for_path(execution_path, vector_search_tool)

    vector_search_tool.execute(
        query="what cities are in Germany", filters=[FilterItem(key="country", value="Germany")]
    )

    expected_combined_filters = {"country": "Germany", "city LIKE": "Berlin"}

    if execution_path["path"] == "mcp":
        mock_tool = execution_path["mock_tool"]
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs

        assert call_kwargs["query"] == "what cities are in Germany"

        meta = call_kwargs["_meta"]
        assert json.loads(meta["filters"]) == expected_combined_filters
        assert meta["num_results"] == vector_search_tool.num_results
        assert meta["query_type"] == vector_search_tool.query_type
        assert meta["columns"] == ",".join(vector_search_tool.columns)
    else:
        vector_search_tool._index.similarity_search.assert_called_once()
        call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs

        assert call_kwargs["filters"] == expected_combined_filters
        assert call_kwargs["num_results"] == vector_search_tool.num_results
        assert call_kwargs["query_type"] == vector_search_tool.query_type
        assert call_kwargs["columns"] == vector_search_tool.columns


def test_kwargs_override_both_num_results_and_query_type(execution_path) -> None:
    vector_search_tool = init_vector_search_tool(
        execution_path["index_name"], num_results=10, query_type="ANN"
    )
    setup_tool_for_path(execution_path, vector_search_tool)

    vector_search_tool.execute(
        query="what cities are in Germany", num_results=3, query_type="HYBRID"
    )

    if execution_path["path"] == "mcp":
        mock_tool = execution_path["mock_tool"]
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs

        assert call_kwargs["query"] == "what cities are in Germany"

        meta = call_kwargs["_meta"]
        # Should use overridden values, not the defaults from constructor
        assert meta["num_results"] == 3
        assert meta["query_type"] == "HYBRID"
        assert meta["columns"] == ",".join(vector_search_tool.columns)
        assert "filters" not in meta
    else:
        vector_search_tool._index.similarity_search.assert_called_once()
        call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs

        # Should use overridden values, not the defaults from constructor
        assert call_kwargs["num_results"] == 3
        assert call_kwargs["query_type"] == "HYBRID"
        assert call_kwargs["columns"] == vector_search_tool.columns
        assert call_kwargs["filters"] == {}


def test_filters_as_dict_mcp_path(mock_mcp_toolkit) -> None:
    """Test that filters can be passed as dict (not just List[FilterItem]) in MCP path."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)
    mock_tool = mock_mcp_toolkit.get_tools.return_value[0]

    # Pass filters as dict instead of List[FilterItem]
    vector_search_tool.execute(
        query="test query",
        filters={"country": "Germany", "status": "active"},
    )

    mock_tool.execute.assert_called_once()
    call_kwargs = mock_tool.execute.call_args.kwargs
    meta = call_kwargs["_meta"]

    # Filters should be serialized correctly
    assert json.loads(meta["filters"]) == {"country": "Germany", "status": "active"}


def test_include_score_always_sent_in_meta(mock_mcp_toolkit) -> None:
    """Test that include_score is always sent explicitly (true or false) to override backend defaults."""
    mock_tool = mock_mcp_toolkit.get_tools.return_value[0]

    # Test with include_score=True
    tool_with_score = init_vector_search_tool(DELTA_SYNC_INDEX, include_score=True)
    tool_with_score.execute(query="test")

    call_kwargs = mock_tool.execute.call_args.kwargs
    assert call_kwargs["_meta"]["include_score"] == "true"

    mock_tool.reset_mock()

    # Test with include_score=False (default)
    tool_without_score = init_vector_search_tool(DELTA_SYNC_INDEX, include_score=False)
    tool_without_score.execute(query="test")

    call_kwargs = mock_tool.execute.call_args.kwargs
    assert call_kwargs["_meta"]["include_score"] == "false"


def test_get_filter_param_description_with_column_metadata() -> None:
    """Test that _get_filter_param_description includes column metadata when available."""
    # Mock table info with column metadata
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

        vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)

        # Test the _get_filter_param_description method directly
        description = vector_search_tool._get_filter_param_description()

        # Should include available columns in description
        assert "Available columns for filtering: category (STRING), price (FLOAT)" in description

        # Should include comprehensive filter syntax
        assert "Inclusion:" in description
        assert "Exclusion:" in description
        assert "Comparisons:" in description
        assert "Pattern match:" in description
        assert "OR logic:" in description

        # Should include examples
        assert "Examples:" in description
        assert "Filter by category:" in description
        assert "Filter by price range:" in description


def test_enhanced_filter_description_used_in_tool_schema() -> None:
    """Test that the tool schema includes comprehensive filter descriptions."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, dynamic_filter=True)

    # Check that the tool schema includes enhanced filter description
    tool_schema = vector_search_tool.tool
    filter_param = tool_schema["function"]["parameters"]["properties"]["filters"]

    # Check that it includes the comprehensive filter syntax
    assert "Inclusion:" in filter_param["description"]
    assert "Exclusion:" in filter_param["description"]
    assert "Comparisons:" in filter_param["description"]
    assert "Pattern match:" in filter_param["description"]
    assert "OR logic:" in filter_param["description"]

    # Check that it includes useful filter information
    assert "array of key-value pairs" in filter_param["description"]
    assert "column" in filter_param["description"]


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
    """Test that predefined filters work correctly when dynamic_filter is False."""
    predefined_filters = {"status": "active", "category": "electronics"}
    # Initialize tool with only predefined filters (dynamic_filter=False by default)
    vector_search_tool = init_vector_search_tool(
        execution_path["index_name"], filters=predefined_filters
    )
    setup_tool_for_path(execution_path, vector_search_tool)

    # The filters parameter should NOT be exposed since dynamic_filter=False
    tool_schema = vector_search_tool.tool
    assert "filters" not in tool_schema["function"]["parameters"]["properties"]

    vector_search_tool.execute(query="what electronics are available")

    if execution_path["path"] == "mcp":
        mock_tool = execution_path["mock_tool"]
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs

        assert call_kwargs["query"] == "what electronics are available"

        meta = call_kwargs["_meta"]
        assert meta["columns"] == ",".join(vector_search_tool.columns)
        assert json.loads(meta["filters"]) == predefined_filters
        assert meta["num_results"] == vector_search_tool.num_results
        assert meta["query_type"] == vector_search_tool.query_type
    else:
        vector_search_tool._index.similarity_search.assert_called_once()
        call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs

        assert call_kwargs["filters"] == predefined_filters
        assert call_kwargs["columns"] == vector_search_tool.columns
        assert call_kwargs["num_results"] == vector_search_tool.num_results
        assert call_kwargs["query_type"] == vector_search_tool.query_type


def test_filter_item_serialization(execution_path) -> None:
    """Test that FilterItem objects are properly converted to dictionaries."""
    vector_search_tool = init_vector_search_tool(execution_path["index_name"])
    setup_tool_for_path(execution_path, vector_search_tool)

    # Test various filter types
    filters = [
        FilterItem(key="category", value="electronics"),
        FilterItem(key="price >=", value=100),
        FilterItem(key="status NOT", value="discontinued"),
        FilterItem(key="tags", value=["wireless", "bluetooth"]),
    ]

    vector_search_tool.execute("find products", filters=filters)

    expected_filters = {
        "category": "electronics",
        "price >=": 100,
        "status NOT": "discontinued",
        "tags": ["wireless", "bluetooth"],
    }

    if execution_path["path"] == "mcp":
        mock_tool = execution_path["mock_tool"]
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs

        assert call_kwargs["query"] == "find products"

        meta = call_kwargs["_meta"]
        # Filters should be serialized as JSON
        assert json.loads(meta["filters"]) == expected_filters
        assert meta["num_results"] == vector_search_tool.num_results
        assert meta["query_type"] == vector_search_tool.query_type
        assert meta["columns"] == ",".join(vector_search_tool.columns)
    else:
        vector_search_tool._index.similarity_search.assert_called_once()
        call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs

        assert call_kwargs["filters"] == expected_filters
        assert call_kwargs["num_results"] == vector_search_tool.num_results
        assert call_kwargs["query_type"] == vector_search_tool.query_type
        assert call_kwargs["columns"] == vector_search_tool.columns


def test_reranker_is_passed_through(execution_path) -> None:
    reranker = DatabricksReranker(columns_to_rerank=["country"])
    vector_search_tool = init_vector_search_tool(execution_path["index_name"], reranker=reranker)
    setup_tool_for_path(execution_path, vector_search_tool)

    vector_search_tool.execute(
        query="what cities are in Germany", filters=[FilterItem(key="country", value="Germany")]
    )

    if execution_path["path"] == "mcp":
        mock_tool = execution_path["mock_tool"]
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs

        assert call_kwargs["query"] == "what cities are in Germany"

        meta = call_kwargs["_meta"]
        assert meta["columns_to_rerank"] == "country"
        assert json.loads(meta["filters"]) == {"country": "Germany"}
        assert meta["num_results"] == vector_search_tool.num_results
        assert meta["query_type"] == vector_search_tool.query_type
    else:
        vector_search_tool._index.similarity_search.assert_called_once()
        call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs

        assert call_kwargs["reranker"] == reranker
        assert call_kwargs["filters"] == {"country": "Germany"}
        assert call_kwargs["num_results"] == vector_search_tool.num_results
        assert call_kwargs["query_type"] == vector_search_tool.query_type


def test_reranker_is_overriden(execution_path) -> None:
    original_reranker = DatabricksReranker(columns_to_rerank=["country"])
    vector_search_tool = init_vector_search_tool(
        execution_path["index_name"], reranker=original_reranker
    )
    setup_tool_for_path(execution_path, vector_search_tool)

    overridden_reranker = DatabricksReranker(columns_to_rerank=["country2"])
    vector_search_tool.execute(
        query="what cities are in Germany",
        filters=[FilterItem(key="country", value="Germany")],
        reranker=overridden_reranker,
    )

    if execution_path["path"] == "mcp":
        mock_tool = execution_path["mock_tool"]
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs

        assert call_kwargs["query"] == "what cities are in Germany"

        meta = call_kwargs["_meta"]
        # Should use overridden reranker columns
        assert meta["columns_to_rerank"] == "country2"
        assert json.loads(meta["filters"]) == {"country": "Germany"}
        assert meta["num_results"] == vector_search_tool.num_results
        assert meta["query_type"] == vector_search_tool.query_type
    else:
        vector_search_tool._index.similarity_search.assert_called_once()
        call_kwargs = vector_search_tool._index.similarity_search.call_args.kwargs

        # Should use overridden reranker
        assert call_kwargs["reranker"] == overridden_reranker
        assert call_kwargs["filters"] == {"country": "Germany"}
        assert call_kwargs["num_results"] == vector_search_tool.num_results
        assert call_kwargs["query_type"] == vector_search_tool.query_type


# ============================================================================
# Response Format Normalization Tests
# ============================================================================


class TestMCPResponseNormalization:
    """Test that MCP responses are normalized to match Direct API format."""

    def test_normalize_mcp_result_basic(self) -> None:
        """Test basic normalization of a single MCP result."""
        vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)

        mcp_result = {
            "id": "doc-123",
            "text": "This is the document content",
            "score": 0.95,
        }

        normalized = vector_search_tool._normalize_mcp_result(mcp_result)

        assert normalized["page_content"] == "This is the document content"
        assert normalized["metadata"]["id"] == "doc-123"
        assert normalized["metadata"]["score"] == 0.95
        assert "text" not in normalized["metadata"]  # text column moved to page_content

    def test_normalize_mcp_result_missing_text_column(self) -> None:
        """Test normalization handles missing text column gracefully."""
        vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)

        mcp_result = {
            "id": "doc-789",
            "score": 0.75,
            # "text" column is missing
        }

        normalized = vector_search_tool._normalize_mcp_result(mcp_result)

        assert normalized["page_content"] == ""  # Empty string when text column missing
        assert normalized["metadata"]["id"] == "doc-789"
        assert normalized["metadata"]["score"] == 0.75

    def test_parse_mcp_response_empty_list(self) -> None:
        """Test parsing empty MCP response."""
        vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)

        mcp_response = json.dumps([])

        results = vector_search_tool._parse_mcp_response(mcp_response)

        assert results == []

    def test_parse_mcp_response_invalid_json(self) -> None:
        """Test parsing invalid JSON raises ValueError."""
        vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)

        with pytest.raises(ValueError, match="Unable to parse MCP response"):
            vector_search_tool._parse_mcp_response("not valid json {")

    def test_parse_mcp_response_not_a_list(self) -> None:
        """Test parsing non-list JSON raises ValueError."""
        vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)

        # MCP should return a list, not a dict
        mcp_response = json.dumps({"error": "something went wrong"})

        with pytest.raises(ValueError, match="Expected MCP vector search to return a JSON array"):
            vector_search_tool._parse_mcp_response(mcp_response)
