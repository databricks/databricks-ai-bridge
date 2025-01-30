import json
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import mlflow
import pytest
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DELTA_SYNC_INDEX,
    DIRECT_ACCESS_INDEX,
    INPUT_TEXTS,
    mock_vs_client,
    mock_workspace_client,
)
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


@pytest.fixture(autouse=True)
def mock_workspace_client():
    with patch("databricks_openai.vector_search_retriever_tool") as MockWorkspaceClient:
        mock_client = MockWorkspaceClient.return_value
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
) -> VectorSearchRetrieverTool:
    kwargs: Dict[str, Any] = {
        "index_name": index_name,
        "columns": columns,
        "tool_name": tool_name,
        "tool_description": tool_description,
        "text_column": text_column,
        "embedding_model_name": embedding_model_name,
    }
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

    expected_resources = [DatabricksVectorSearchIndex(index_name=index_name)] + (
        [DatabricksServingEndpoint(endpoint_name="text-embedding-3-small")]
        if self_managed_embeddings_test.embedding_model_name
        else []
    )
    assert vector_search_tool.resources == expected_resources

    # simulate call to openai.chat.completions.create
    chat_completion_resp = get_chat_completion_response(tool_name, index_name)
    tool_call = chat_completion_resp.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    docs = vector_search_tool.execute(
        query=args["query"], openai_client=self_managed_embeddings_test.open_ai_client
    )
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
def test_vector_search_retriever_long_idex_name_rewrite(
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


@pytest.mark.parametrize("index_name", ALL_INDEX_NAMES)
@pytest.mark.parametrize(
    "tool_name", [None, "really_really_really_long_tool_name_that_should_be_truncated_to_64_chars"]
)
def test_vector_search_retriever_long_tool_name(
    index_name: str,
    tool_name: Optional[str],
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
        tool_name=tool_name,
        text_column=self_managed_embeddings_test.text_column,
        embedding_model_name=self_managed_embeddings_test.embedding_model_name,
    )
    assert len(vector_search_tool.tool["function"]["name"]) <= 64
