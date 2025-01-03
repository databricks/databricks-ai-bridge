from typing import Any, Dict, Generator, List, Optional

import pytest

from unittest import mock
from unittest.mock import MagicMock, patch
from databricks_ai_bridge.test_utils.vector_search import (  # noqa: F401
    ALL_INDEX_NAMES,
    DELTA_SYNC_INDEX,
    DIRECT_ACCESS_INDEX,
    mock_vs_client,
    mock_workspace_client,
)
from databricks_openai import VectorSearchRetrieverTool
from pydantic import BaseModel
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call_param import Function
import os

@pytest.fixture(autouse=True)
def mock_openai_client():
    mock_client = MagicMock()
    mock_client.api_key = "fake_api_key"
    mock_client.embeddings.create.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]
    }
    with patch("openai.OpenAI", return_value=mock_client):
        yield mock_client

def get_chat_completion_response(tool_name: str, index_name: str):
    return ChatCompletion(
        id='chatcmpl-AlSTQf3qIjeEOdoagPXUYhuWZkwme',
        choices=[
            Choice(
                finish_reason='tool_calls',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=None,
                    refusal=None,
                    role='assistant',
                    audio=None,
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='call_VtmBTsVM2zQ3yL5GzddMgWb0',
                            function=Function(
                                arguments='{"query":"Databricks Agent Framework"}',
                                name=tool_name or index_name.split(".")[-1]  # see rewrite_index_name() in VectorSearchRetrieverTool
                            ),
                            type='function'
                        )
                    ]
                )
            )
        ],
        created=1735874232,
        model='gpt-4o-mini-2024-07-18',
        object='chat.completion',
    )

def init_vector_search_tool(
        index_name: str,
        columns: Optional[List[str]] = None,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        text_column: Optional[str] = None,
) -> VectorSearchRetrieverTool:
    kwargs: Dict[str, Any] = {
        "index_name": index_name,
        "columns": columns,
        "tool_name": tool_name,
        "tool_description": tool_description,
        "text_column": text_column,
    }
    if index_name != DELTA_SYNC_INDEX:
        kwargs.update(
            {
                "text_column": "text",
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
        tool_description: Optional[str]
) -> None:
    if index_name == DELTA_SYNC_INDEX:
        self_managed_embeddings_test = SelfManagedEmbeddingsTest()
    else:
        from openai import OpenAI
        self_managed_embeddings_test = SelfManagedEmbeddingsTest("text", "text-embedding-3-small", OpenAI(api_key="your-api-key"))

    vector_search_tool = init_vector_search_tool(
        index_name=index_name,
        columns=columns,
        tool_name=tool_name,
        tool_description=tool_description,
        text_column=self_managed_embeddings_test.text_column,
    )
    assert isinstance(vector_search_tool, BaseModel)
    # simulate call to openai.chat.completions.create
    chat_completion_resp = get_chat_completion_response(tool_name, index_name)
    response = vector_search_tool.execute_retriever_calls(
        chat_completion_resp,
        embedding_model_name=self_managed_embeddings_test.embedding_model_name,
        openai_client=self_managed_embeddings_test.open_ai_client
    )
    assert response is not None


@pytest.mark.parametrize("columns", [None, ["id", "text"]])
@pytest.mark.parametrize("tool_name", [None, "test_tool"])
@pytest.mark.parametrize("tool_description", [None, "Test tool for vector search"])
def test_open_ai_client_from_env(
        columns: Optional[List[str]],
        tool_name: Optional[str],
        tool_description: Optional[str]
) -> None:
    self_managed_embeddings_test = SelfManagedEmbeddingsTest("text", "text-embedding-3-small", None)
    os.environ["OPENAI_API_KEY"] = "your-api-key"

    vector_search_tool = init_vector_search_tool(
        index_name=DIRECT_ACCESS_INDEX,
        columns=columns,
        tool_name=tool_name,
        tool_description=tool_description,
        text_column=self_managed_embeddings_test.text_column,
    )
    assert isinstance(vector_search_tool, BaseModel)
    # simulate call to openai.chat.completions.create
    chat_completion_resp = get_chat_completion_response(tool_name, DIRECT_ACCESS_INDEX)
    response = vector_search_tool.execute_retriever_calls(
        chat_completion_resp,
        embedding_model_name=self_managed_embeddings_test.embedding_model_name,
        openai_client=self_managed_embeddings_test.open_ai_client
    )
    assert response is not None
