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
    client = mock.MagicMock()
    client.predict.return_value = _MOCK_CHAT_RESPONSE
    client.predict_stream.return_value = _MOCK_STREAM_RESPONSE
    with mock.patch("mlflow.deployments.get_deploy_client", return_value=client):
        yield


@pytest.fixture
def llm() -> ChatDatabricks:
    return ChatDatabricks(model="databricks-meta-llama-3-3-70b-instruct", target_uri="databricks")
