import json
import os
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, create_autospec, patch

import mlflow
import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import ModelServingUserCredentials
from databricks.vector_search.client import VectorSearchIndex
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
    assert [d["page_content"] in INPUT_TEXTS for d in outputs]

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
        mockVSClient.assert_called_once_with(disable_notice=True, credential_strategy=None)

    w = WorkspaceClient(host="testDogfod.com", token="fakeToken")
    with patch("databricks.vector_search.client.VectorSearchClient") as mockVSClient:
        with patch("databricks.sdk.service.serving.ServingEndpointsAPI.get", return_value=None):
            vsTool = VectorSearchRetrieverTool(
                index_name="catalog.schema.my_index_name",
                text_column="abc",
                embedding_model_name="text-embedding-3-small",
                tool_description="desc",
                workspace_client=w,
            )
            mockVSClient.assert_called_once_with(disable_notice=True, credential_strategy=None)


def test_kwargs_are_passed_through() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, score_threshold=0.5)
    vector_search_tool._index = create_autospec(VectorSearchIndex, instance=True)

    # extra_param is ignored because it isn't part of the signature for similarity_search
    vector_search_tool.execute(
        query="what cities are in Germany", debug_level=2, extra_param="something random"
    )
    vector_search_tool._index.similarity_search.assert_called_once_with(
        columns=vector_search_tool.columns,
        query_text="what cities are in Germany",
        num_results=vector_search_tool.num_results,
        query_type=vector_search_tool.query_type,
        query_vector=None,
        filters={},
        score_threshold=0.5,
        debug_level=2,
    )


def test_filters_are_passed_through() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)
    vector_search_tool._index.similarity_search = MagicMock()

    vector_search_tool.execute(
        {"query": "what cities are in Germany"},
        filters=[FilterItem(key="country", value="Germany")],
    )
    vector_search_tool._index.similarity_search.assert_called_once_with(
        columns=vector_search_tool.columns,
        query_text={"query": "what cities are in Germany"},
        filters={"country": "Germany"},
        num_results=vector_search_tool.num_results,
        query_type=vector_search_tool.query_type,
        query_vector=None,
    )


def test_filters_are_combined() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, filters={"city LIKE": "Berlin"})
    vector_search_tool._index.similarity_search = MagicMock()

    vector_search_tool.execute(
        query="what cities are in Germany", filters=[FilterItem(key="country", value="Germany")]
    )
    vector_search_tool._index.similarity_search.assert_called_once_with(
        columns=vector_search_tool.columns,
        query_text="what cities are in Germany",
        filters={"city LIKE": "Berlin", "country": "Germany"},
        num_results=vector_search_tool.num_results,
        query_type=vector_search_tool.query_type,
        query_vector=None,
    )


def test_kwargs_override_both_num_results_and_query_type() -> None:
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX, num_results=10, query_type="ANN")
    vector_search_tool._index = create_autospec(VectorSearchIndex, instance=True)

    vector_search_tool.execute(
        query="what cities are in Germany", num_results=3, query_type="HYBRID"
    )
    vector_search_tool._index.similarity_search.assert_called_once_with(
        columns=vector_search_tool.columns,
        query_text="what cities are in Germany",
        filters={},
        num_results=3,  # Should use overridden value
        query_type="HYBRID",  # Should use overridden value
        query_vector=None,
    )


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


def test_predefined_filters_work_without_dynamic_filter() -> None:
    """Test that predefined filters work correctly when dynamic_filter is False."""
    # Initialize tool with only predefined filters (dynamic_filter=False by default)
    vector_search_tool = init_vector_search_tool(
        DELTA_SYNC_INDEX, filters={"status": "active", "category": "electronics"}
    )

    # The filters parameter should NOT be exposed since dynamic_filter=False
    tool_schema = vector_search_tool.tool
    assert "filters" not in tool_schema["function"]["parameters"]["properties"]

    # Test that predefined filters are used
    vector_search_tool._index.similarity_search = MagicMock()

    vector_search_tool.execute(query="what electronics are available")

    vector_search_tool._index.similarity_search.assert_called_once_with(
        columns=vector_search_tool.columns,
        query_text="what electronics are available",
        filters={"status": "active", "category": "electronics"},  # Only predefined filters
        num_results=vector_search_tool.num_results,
        query_type=vector_search_tool.query_type,
        query_vector=None,
    )


def test_filter_item_serialization() -> None:
    """Test that FilterItem objects are properly converted to dictionaries."""
    vector_search_tool = init_vector_search_tool(DELTA_SYNC_INDEX)
    vector_search_tool._index.similarity_search = MagicMock()

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

    vector_search_tool._index.similarity_search.assert_called_once_with(
        columns=vector_search_tool.columns,
        query_text="find products",
        filters=expected_filters,
        num_results=vector_search_tool.num_results,
        query_type=vector_search_tool.query_type,
        query_vector=None,
    )
