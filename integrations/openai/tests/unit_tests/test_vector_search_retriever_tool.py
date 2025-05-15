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
    trace = mlflow.get_last_active_trace()
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
