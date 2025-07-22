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
    # Mock MLflow deployments client
    mlflow_client = mock.MagicMock()
    mlflow_client.predict.return_value = _MOCK_CHAT_RESPONSE
    mlflow_client.predict_stream.return_value = _MOCK_STREAM_RESPONSE
    
    # Mock OpenAI client response objects
    class MockOpenAIMessage:
        def __init__(self, role="assistant", content="", tool_calls=None):
            self.role = role
            self.content = content
            self.tool_calls = tool_calls or []
    
    class MockOpenAIChoice:
        def __init__(self, message, finish_reason="stop"):
            self.message = message
            self.finish_reason = finish_reason
    
    class MockOpenAIUsage:
        def __init__(self, prompt_tokens=30, completion_tokens=36, total_tokens=66):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens
    
    class MockOpenAIResponse:
        def __init__(self):
            # Ensure the content matches exactly
            expected_content = _MOCK_CHAT_RESPONSE["choices"][0]["message"]["content"]
            self.choices = [
                MockOpenAIChoice(
                    MockOpenAIMessage(content=expected_content)
                )
            ]
            self.usage = MockOpenAIUsage()
            self.model = _MOCK_CHAT_RESPONSE["model"]
    
    # Mock OpenAI streaming response
    class MockOpenAIDelta:
        def __init__(self, role=None, content=None, tool_calls=None):
            self.role = role
            self.content = content
            self.tool_calls = tool_calls
    
    class MockOpenAIStreamChoice:
        def __init__(self, delta, finish_reason=None):
            self.delta = delta
            self.finish_reason = finish_reason
    
    class MockOpenAIStreamChunk:
        def __init__(self, choices, usage=None):
            self.choices = choices
            self.usage = usage
    
    def mock_openai_stream():
        for chunk_data in _MOCK_STREAM_RESPONSE:
            choice_data = chunk_data["choices"][0]
            delta_data = choice_data["delta"]
            usage_data = chunk_data.get("usage")
            
            delta = MockOpenAIDelta(
                role=delta_data.get("role"),
                content=delta_data.get("content", "")
            )
            choice = MockOpenAIStreamChoice(
                delta=delta,
                finish_reason=choice_data.get("finish_reason")
            )
            usage = MockOpenAIUsage(**usage_data) if usage_data else None
            yield MockOpenAIStreamChunk([choice], usage)
    
    # Mock OpenAI client
    openai_client = mock.MagicMock()
    
    def mock_create_completion(**kwargs):
        if kwargs.get("stream"):
            return mock_openai_stream()
        else:
            return MockOpenAIResponse()
    
    openai_client.chat.completions.create.side_effect = mock_create_completion
    
    with mock.patch("mlflow.deployments.get_deploy_client", return_value=mlflow_client), \
         mock.patch("databricks_langchain.utils.get_openai_client", return_value=openai_client), \
         mock.patch("databricks_langchain.chat_models.get_openai_client", return_value=openai_client):
        yield


@pytest.fixture
def llm() -> ChatDatabricks:
    return ChatDatabricks(model="databricks-meta-llama-3-3-70b-instruct", target_uri="databricks")


@pytest.fixture
def llm_openai() -> ChatDatabricks:
    return ChatDatabricks(
        model="databricks-meta-llama-3-3-70b-instruct", 
        target_uri="databricks",
        use_openai_client=True
    )
