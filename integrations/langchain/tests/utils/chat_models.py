from typing import Generator
from unittest import mock

import pytest

from databricks_langchain import ChatDatabricks

_MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl_id",
    "object": "chat.completion",
    "created": 1721875529,
    "model": "meta-llama-3.1-70b-instruct-072424",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "To calculate the result of 36939 multiplied by 8922.4, "
                "I get:\n\n36939 x 8922.4 = 329,511,111.6",
            },
            "finish_reason": "stop",
            "logprobs": None,
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 36, "total_tokens": 66},
}

_MOCK_STREAM_RESPONSE = [
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "36939"},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "x"},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 22, "total_tokens": 52},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "8922.4"},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 24, "total_tokens": 54},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": " = "},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 28, "total_tokens": 58},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "329,511,111.6"},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 30, "total_tokens": 60},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 36, "total_tokens": 66},
    },
]


@pytest.fixture(autouse=True)
def mock_client() -> Generator:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
    from openai.types.chat.chat_completion_chunk import ChoiceDelta
    from openai.types.completion_usage import CompletionUsage

    def mock_openai_stream():
        for chunk_data in _MOCK_STREAM_RESPONSE:
            choice_data = chunk_data["choices"][0]
            delta_data = choice_data["delta"]
            usage_data = chunk_data.get("usage")

            delta = ChoiceDelta(
                role=delta_data.get("role"), content=delta_data.get("content", ""), tool_calls=None
            )
            choice = ChunkChoice(
                index=0, delta=delta, finish_reason=choice_data.get("finish_reason"), logprobs=None
            )
            usage = CompletionUsage(**usage_data) if usage_data else None
            yield ChatCompletionChunk(
                id=chunk_data["id"],
                choices=[choice],
                created=chunk_data["created"],
                model=chunk_data["model"],
                object="chat.completion.chunk",
                usage=usage,
            )

    def create_mock_response():
        expected_content = _MOCK_CHAT_RESPONSE["choices"][0]["message"]["content"]
        message = ChatCompletionMessage(role="assistant", content=expected_content, tool_calls=None)
        choice = Choice(index=0, message=message, finish_reason="stop", logprobs=None)
        usage = CompletionUsage(**_MOCK_CHAT_RESPONSE["usage"])
        return ChatCompletion(
            id=_MOCK_CHAT_RESPONSE["id"],
            choices=[choice],
            created=_MOCK_CHAT_RESPONSE["created"],
            model=_MOCK_CHAT_RESPONSE["model"],
            object="chat.completion",
            usage=usage,
        )

    # Mock OpenAI client
    openai_client = mock.MagicMock()

    def mock_create_completion(**kwargs):
        if kwargs.get("stream"):
            return mock_openai_stream()
        else:
            return create_mock_response()

    openai_client.chat.completions.create.side_effect = mock_create_completion

    with (
        mock.patch("databricks_langchain.utils.get_openai_client", return_value=openai_client),
        mock.patch(
            "databricks_langchain.chat_models.get_openai_client", return_value=openai_client
        ),
    ):
        yield


@pytest.fixture
def llm() -> ChatDatabricks:
    return ChatDatabricks(model="databricks-meta-llama-3-3-70b-instruct")
