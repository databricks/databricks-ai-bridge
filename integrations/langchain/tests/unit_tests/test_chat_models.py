"""Test chat model integration."""

import json
from unittest.mock import Mock, patch

import mlflow  # type: ignore # noqa: F401
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableMap
from pydantic import BaseModel, Field

from databricks_langchain.chat_models import (
    ChatDatabricks,
    _convert_dict_to_message,
    _convert_dict_to_message_chunk,
    _convert_lc_messages_to_responses_api,
    _convert_message_to_dict,
    _convert_responses_api_chunk_to_lc_chunk,
)
from tests.utils.chat_models import (  # noqa: F401
    _MOCK_CHAT_RESPONSE,
    _MOCK_STREAM_RESPONSE,
    llm,
    mock_client,
)


def test_dict(llm: ChatDatabricks) -> None:
    d = llm.dict()
    assert d["_type"] == "chat-databricks"
    assert d["model"] == "databricks-meta-llama-3-3-70b-instruct"
    assert d["target_uri"] == "databricks"


def test_dict_with_endpoint() -> None:
    llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", target_uri="databricks")
    d = llm.dict()
    assert d["_type"] == "chat-databricks"
    assert d["model"] == "databricks-meta-llama-3-3-70b-instruct"
    assert d["target_uri"] == "databricks"

    llm = ChatDatabricks(
        model="databricks-meta-llama-3-3-70b-instruct",
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        target_uri="databricks",
    )
    d = llm.dict()
    assert d["_type"] == "chat-databricks"
    assert d["model"] == "databricks-meta-llama-3-3-70b-instruct"
    assert d["target_uri"] == "databricks"


def test_chat_model_predict(llm: ChatDatabricks) -> None:
    res = llm.invoke(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "36939 * 8922.4"},
        ]
    )
    assert res.content == _MOCK_CHAT_RESPONSE["choices"][0]["message"]["content"]  # type: ignore[index]


def test_chat_model_stream(llm: ChatDatabricks) -> None:
    res = llm.stream(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "36939 * 8922.4"},
        ]
    )
    for chunk, expected in zip(res, _MOCK_STREAM_RESPONSE):
        assert chunk.content == expected["choices"][0]["delta"]["content"]  # type: ignore[index]


def test_chat_model_stream_with_usage(llm: ChatDatabricks) -> None:
    def _assert_usage(chunk, expected):
        usage = chunk.usage_metadata
        assert usage is not None
        assert usage["input_tokens"] == expected["usage"]["prompt_tokens"]
        assert usage["output_tokens"] == expected["usage"]["completion_tokens"]
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]

    # Method 1: Pass stream_usage=True to the constructor
    res = llm.stream(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "36939 * 8922.4"},
        ],
        stream_usage=True,
    )
    for chunk, expected in zip(res, _MOCK_STREAM_RESPONSE):
        assert chunk.content == expected["choices"][0]["delta"]["content"]  # type: ignore[index]
        _assert_usage(chunk, expected)

    # Method 2: Pass stream_usage=True to the constructor
    llm_with_usage = ChatDatabricks(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        target_uri="databricks",
        stream_usage=True,
    )
    res = llm_with_usage.stream(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "36939 * 8922.4"},
        ],
    )
    for chunk, expected in zip(res, _MOCK_STREAM_RESPONSE):
        assert chunk.content == expected["choices"][0]["delta"]["content"]  # type: ignore[index]
        _assert_usage(chunk, expected)


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


class GetPopulation(BaseModel):
    """Get the current population in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def test_chat_model_bind_tools(llm: ChatDatabricks) -> None:
    llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
    response = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
    assert isinstance(response, AIMessage)


@pytest.mark.parametrize(
    ("tool_choice", "expected_output"),
    [
        ("auto", "auto"),
        ("none", "none"),
        ("required", "required"),
        # "any" should be replaced with "required"
        ("any", "required"),
        ("GetWeather", {"type": "function", "function": {"name": "GetWeather"}}),
        (
            {"type": "function", "function": {"name": "GetWeather"}},
            {"type": "function", "function": {"name": "GetWeather"}},
        ),
    ],
)
def test_chat_model_bind_tools_with_choices(
    llm: ChatDatabricks, tool_choice, expected_output
) -> None:
    llm_with_tool = llm.bind_tools([GetWeather], tool_choice=tool_choice)
    assert llm_with_tool.kwargs["tool_choice"] == expected_output


def test_chat_model_bind_tolls_with_invalid_choices(llm: ChatDatabricks) -> None:
    with pytest.raises(ValueError, match="Unrecognized tool_choice type"):
        llm.bind_tools([GetWeather], tool_choice=123)

    # Non-existing tool
    with pytest.raises(ValueError, match="Tool choice"):
        llm.bind_tools(
            [GetWeather],
            tool_choice={"type": "function", "function": {"name": "NonExistingTool"}},
        )


# Pydantic-based schema
class AnswerWithJustification(BaseModel):
    """An answer to the user question along with justification for the answer."""

    answer: str = Field(description="The answer to the user question.")
    justification: str = Field(description="The justification for the answer.")


# Raw JSON schema
JSON_SCHEMA = {
    "title": "AnswerWithJustification",
    "description": "An answer to the user question along with justification for the answer.",
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "title": "Answer",
            "description": "The answer to the user question.",
        },
        "justification": {
            "type": "string",
            "title": "Justification",
            "description": "The justification for the answer.",
        },
    },
    "required": ["answer", "justification"],
}


@pytest.mark.parametrize("schema", [AnswerWithJustification, JSON_SCHEMA, None])
@pytest.mark.parametrize("method", ["function_calling", "json_mode", "json_schema"])
def test_chat_model_with_structured_output(llm, schema, method: str):
    if schema is None and method in ["function_calling", "json_schema"]:
        pytest.skip("Cannot use function_calling without schema")

    structured_llm = llm.with_structured_output(schema, method=method)

    bind = structured_llm.first.kwargs
    if method == "function_calling":
        assert bind["tool_choice"]["function"]["name"] == "AnswerWithJustification"
    elif method == "json_schema":
        assert bind["response_format"]["json_schema"]["schema"] == JSON_SCHEMA
    else:
        assert bind["response_format"] == {"type": "json_object"}

    structured_llm = llm.with_structured_output(schema, include_raw=True, method=method)
    assert isinstance(structured_llm.first, RunnableMap)


### Test data conversion functions ###


@pytest.mark.parametrize(
    ("role", "expected_output"),
    [
        ("user", HumanMessage("foo")),
        ("system", SystemMessage("foo")),
        ("assistant", AIMessage("foo")),
        ("tool", ToolMessage(content="foo", tool_call_id="call_123")),
        ("any_role", ChatMessage(content="foo", role="any_role")),
    ],
)
def test_convert_message(role: str, expected_output: BaseMessage) -> None:
    if role == "tool":
        message = {"role": role, "content": "foo", "tool_call_id": "call_123"}
    else:
        message = {"role": role, "content": "foo"}
    result = _convert_dict_to_message(message)
    assert result == expected_output

    # convert back
    dict_result = _convert_message_to_dict(result)
    assert dict_result == message


def test_convert_message_not_propagate_id() -> None:
    # The AIMessage returned by the model endpoint can contain "id" field,
    # but it is not always supported for requests. Therefore, we should not
    # propagate it to the request payload.
    message = AIMessage(content="foo", id="some-id")
    result = _convert_message_to_dict(message)
    assert "id" not in result


def test_convert_message_with_tool_calls() -> None:
    ID = "call_fb5f5e1a-bac0-4422-95e9-d06e6022ad12"
    tool_calls = [
        {
            "id": ID,
            "type": "function",
            "function": {
                "name": "main__test__python_exec",
                "arguments": '{"code": "result = 36939 * 8922.4"}',
            },
        }
    ]
    message_with_tools = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
        "id": ID,
    }
    result = _convert_dict_to_message(message_with_tools)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": tool_calls},
        id=ID,
        tool_calls=[
            {
                "name": tool_calls[0]["function"]["name"],  # type: ignore[index]
                "args": json.loads(tool_calls[0]["function"]["arguments"]),  # type: ignore[index]
                "id": ID,
                "type": "tool_call",
            }
        ],
    )
    assert result == expected_output

    # convert back
    dict_result = _convert_message_to_dict(result)
    message_with_tools.pop("id")  # id is not propagated
    assert dict_result == message_with_tools


def test_convert_tool_message() -> None:
    tool_message = ToolMessage(content="result", tool_call_id="call_123")
    result = _convert_message_to_dict(tool_message)
    expected = {"role": "tool", "content": "result", "tool_call_id": "call_123"}
    assert result == expected

    # convert back
    converted_back = _convert_dict_to_message(result)
    assert converted_back.content == tool_message.content
    assert converted_back.tool_call_id == tool_message.tool_call_id


@pytest.mark.parametrize(
    ("role", "expected_output"),
    [
        ("user", HumanMessageChunk(content="foo")),
        ("system", SystemMessageChunk(content="foo")),
        ("assistant", AIMessageChunk(content="foo")),
        ("any_role", ChatMessageChunk(content="foo", role="any_role")),
    ],
)
def test_convert_message_chunk(role: str, expected_output: BaseMessage) -> None:
    delta = {"role": role, "content": "foo"}
    result = _convert_dict_to_message_chunk(delta, "default_role")
    assert result == expected_output

    # convert back
    dict_result = _convert_message_to_dict(result)
    assert dict_result == delta


def test_convert_message_chunk_with_tool_calls() -> None:
    delta_with_tools = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"index": 0, "function": {"arguments": " }"}}],
    }
    result = _convert_dict_to_message_chunk(delta_with_tools, "role")
    expected_output = AIMessageChunk(
        content="",
        additional_kwargs={"tool_calls": delta_with_tools["tool_calls"]},
        id=None,
        tool_call_chunks=[ToolCallChunk(name=None, args=" }", id=None, index=0)],
    )
    assert result == expected_output


def test_convert_tool_message_chunk() -> None:
    delta = {
        "role": "tool",
        "content": "foo",
        "tool_call_id": "tool_call_id",
        "id": "some_id",
    }
    result = _convert_dict_to_message_chunk(delta, "default_role")
    expected_output = ToolMessageChunk(content="foo", id="some_id", tool_call_id="tool_call_id")
    assert result == expected_output

    # convert back
    dict_result = _convert_message_to_dict(result)
    delta.pop("id")  # id is not propagated
    assert dict_result == delta


def test_convert_message_to_dict_function() -> None:
    with pytest.raises(ValueError, match="Function messages are not supported"):
        _convert_message_to_dict(FunctionMessage(content="", name="name"))


def test_convert_response_to_chat_result_llm_output(llm: ChatDatabricks) -> None:
    """Test that _convert_response_to_chat_result correctly sets llm_output."""

    result = llm._convert_response_to_chat_result(_MOCK_CHAT_RESPONSE)

    expected_content = _MOCK_CHAT_RESPONSE["choices"][0]["message"]["content"]
    expected = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content=expected_content),
                generation_info={},
            ),
        ],
        llm_output={
            "id": _MOCK_CHAT_RESPONSE["id"],
            "object": _MOCK_CHAT_RESPONSE["object"],
            "created": _MOCK_CHAT_RESPONSE["created"],
            "model": _MOCK_CHAT_RESPONSE["model"],
            "model_name": _MOCK_CHAT_RESPONSE["model"],
            "usage": _MOCK_CHAT_RESPONSE["usage"],
        },
    )

    assert result == expected


def test_convert_lc_messages_to_responses_api_basic():
    """Test _convert_lc_messages_to_responses_api with basic messages."""
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there!"),
    ]

    result = _convert_lc_messages_to_responses_api(messages)
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {
            "type": "message",
            "role": "assistant",
            "id": None,
            "content": [{"type": "output_text", "text": "Hi there!"}],
        },
    ]
    assert result == expected


def test_convert_lc_messages_to_responses_api_with_tool_calls():
    """Test _convert_lc_messages_to_responses_api with tool calls."""
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
        AIMessage(
            content="I'll check the weather.",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"location": "SF"},
                    "id": "call_123",
                    "type": "tool_call",
                }
            ],
            id="msg_123",
        ),
        ToolMessage(content="Sunny, 72°F", tool_call_id="call_123"),
    ]

    result = _convert_lc_messages_to_responses_api(messages)
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_123",
            "content": [
                {"type": "output_text", "text": "I'll check the weather."},
            ],
        },
        {
            "type": "function_call",
            "name": "get_weather",
            "call_id": "call_123",
            "arguments": '{"location": "SF"}',
            "id": "msg_123",
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "Sunny, 72°F",
        },
    ]
    assert result == expected


def test_convert_lc_messages_to_responses_api_with_complex_content():
    """Test _convert_lc_messages_to_responses_api with complex content."""
    messages = [
        AIMessage(
            content=[
                {"type": "text", "text": "Here's the answer:", "annotations": [{"key": "value"}]},
                {"type": "refusal", "refusal": "I cannot do that."},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Let me think..."}],
                },
            ],
            id="msg_456",
        )
    ]

    result = _convert_lc_messages_to_responses_api(messages)
    expected = [
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_456",
            "content": [
                {
                    "type": "output_text",
                    "text": "Here's the answer:",
                    "annotations": [{"key": "value"}],
                },
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_456",
            "content": [
                {
                    "type": "refusal",
                    "refusal": "I cannot do that.",
                },
            ],
        },
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Let me think..."}],
            "id": "msg_456",
        },
    ]
    assert result == expected


def test_convert_responses_api_chunk_to_lc_chunk_text_delta():
    """Test _convert_responses_api_chunk_to_lc_chunk with text delta."""
    chunk = {"type": "response.output_text.delta", "item_id": "item_123", "delta": "Hello"}

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)

    assert isinstance(result, AIMessageChunk)
    assert result.content == [{"type": "text", "text": "Hello"}]
    assert result.id == "item_123"


def test_convert_responses_api_chunk_to_lc_chunk_function_call():
    """Test _convert_responses_api_chunk_to_lc_chunk with function call."""
    chunk = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "id": "item_123",
            "call_id": "call_456",
            "name": "get_weather",
            "arguments": '{"location": "SF"}',
        },
    }

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)
    expected = AIMessageChunk(
        id="call_456",
        content=[
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "SF"}',
                "id": "item_123",
                "call_id": "call_456",
            },
        ],
        tool_call_chunks=[
            ToolCallChunk(name="get_weather", args='{"location": "SF"}', id="call_456")
        ],
    )
    assert result == expected


def test_convert_responses_api_chunk_to_lc_chunk_function_call_output():
    """Test _convert_responses_api_chunk_to_lc_chunk with function call output."""
    chunk = {
        "type": "response.output_item.done",
        "item": {"type": "function_call_output", "call_id": "call_456", "output": "Sunny, 72°F"},
    }

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)

    assert isinstance(result, ToolMessageChunk)
    assert result.content == "Sunny, 72°F"
    assert result.tool_call_id == "call_456"


def test_convert_responses_api_chunk_to_lc_chunk_message():
    """Test _convert_responses_api_chunk_to_lc_chunk with message."""
    chunk = {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "id": "msg_123",
            "content": [
                {"type": "output_text", "text": "Hello!", "annotations": [{"key": "value"}]},
                {"type": "refusal", "refusal": "I cannot help with that."},
            ],
        },
    }

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)
    expected = AIMessageChunk(
        id="msg_123",
        content=[
            {"type": "text", "text": "Hello!", "annotations": [{"key": "value"}]},
            {"type": "refusal", "refusal": "I cannot help with that."},
        ],
    )
    assert result == expected


def test_convert_responses_api_chunk_to_lc_chunk_skip_duplicate():
    """Test _convert_responses_api_chunk_to_lc_chunk skips duplicate text."""
    previous_chunk = {"type": "response.output_text.delta", "item_id": "item_123", "delta": "Hello"}

    chunk = {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "id": "item_123",
            "content": [{"type": "output_text", "text": "Hello"}],
        },
    }

    result = _convert_responses_api_chunk_to_lc_chunk(chunk, previous_chunk)
    assert result is None


def test_convert_responses_api_chunk_to_lc_chunk_skip_duplicate_with_annotations():
    """Test _convert_responses_api_chunk_to_lc_chunk skips duplicate text."""
    previous_chunk = {"type": "response.output_text.delta", "item_id": "item_123", "delta": "Hello"}

    chunk = {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "id": "item_123",
            "content": [
                {
                    "type": "output_text",
                    "text": "Hello",
                    "annotations": [
                        {"type": "url_citation", "title": "title", "url": "google.com"}
                    ],
                }
            ],
        },
    }

    result = _convert_responses_api_chunk_to_lc_chunk(chunk, previous_chunk)
    expected = AIMessageChunk(
        content=[
            {"annotations": [{"type": "url_citation", "title": "title", "url": "google.com"}]}
        ],
        additional_kwargs={},
        response_metadata={},
        id="item_123",
    )
    assert result == expected


def test_convert_responses_api_chunk_to_lc_chunk_error():
    """Test _convert_responses_api_chunk_to_lc_chunk with error."""
    chunk = {"type": "error", "error": "Something went wrong"}

    with pytest.raises(ValueError, match="Something went wrong"):
        _convert_responses_api_chunk_to_lc_chunk(chunk)


def test_convert_responses_api_chunk_to_lc_chunk_unknown_type():
    """Test _convert_responses_api_chunk_to_lc_chunk with unknown type."""
    chunk = {"type": "unknown_type", "data": "some data"}

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)
    assert result is None


def test_convert_responses_api_chunk_to_lc_chunk_special_items():
    """Test _convert_responses_api_chunk_to_lc_chunk with special item types."""
    chunk = {
        "type": "response.output_item.done",
        "item": {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Let me think about this..."}],
        },
    }

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)

    assert isinstance(result, AIMessageChunk)
    assert result.content == [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Let me think about this..."}],
        }
    ]


### Test ChatDatabricks response conversion methods ###


def test_convert_responses_api_response_to_chat_result():
    """Test _convert_responses_api_response_to_chat_result method."""
    llm = ChatDatabricks(model="test-model", use_responses_api=True)

    response = {
        "id": "response_123",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello!",
                        "id": "text_123",
                        "annotations": [{"key": "value"}],
                    }
                ],
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "SF"}',
                "call_id": "call_123",
            },
        ],
    }

    result = llm._convert_responses_api_response_to_chat_result(response)
    expected = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    id="response_123",
                    content=[
                        {
                            "type": "text",
                            "text": "Hello!",
                            "annotations": [{"key": "value"}],
                            "id": "text_123",
                        },
                        {
                            "type": "function_call",
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}',
                            "call_id": "call_123",
                        },
                    ],
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"location": "SF"},
                            "id": "call_123",
                        }
                    ],
                )
            ),
        ]
    )
    assert result == expected


def test_convert_responses_api_response_to_chat_result_with_error():
    """Test _convert_responses_api_response_to_chat_result with error."""
    llm = ChatDatabricks(model="test-model", use_responses_api=True)

    response = {"error": "Something went wrong"}

    with pytest.raises(ValueError, match="Something went wrong"):
        llm._convert_responses_api_response_to_chat_result(response)


def test_convert_chatagent_response_to_chat_result():
    """Test _convert_chatagent_response_to_chat_result method."""
    llm = ChatDatabricks(model="test-model")

    response = {"messages": [{"role": "assistant", "content": "Hello from ChatAgent!"}]}

    result = llm._convert_chatagent_response_to_chat_result(response)
    expected = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content=[{"role": "assistant", "content": "Hello from ChatAgent!"}]
                )
            ),
        ]
    )
    assert result == expected


### Test ChatDatabricks initialization and configuration ###


def test_chat_databricks_init_with_use_responses_api():
    """Test ChatDatabricks initialization with use_responses_api."""
    llm = ChatDatabricks(model="test-model", use_responses_api=True)
    assert llm.use_responses_api is True


def test_chat_databricks_init_with_extra_params():
    """Test ChatDatabricks initialization with extra_params."""
    extra_params = {"custom_param": "value"}
    llm = ChatDatabricks(model="test-model", extra_params=extra_params)
    assert llm.extra_params == extra_params


def test_chat_databricks_init_sets_client():
    """Test ChatDatabricks initialization sets client."""
    with patch("databricks_langchain.chat_models.get_deployment_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        llm = ChatDatabricks(model="test-model", target_uri="custom-uri")

        mock_get_client.assert_called_once_with("custom-uri")
        assert llm.client == mock_client


### Test ChatDatabricks _prepare_inputs method ###


def test_prepare_inputs_basic():
    """Test _prepare_inputs method with basic parameters."""
    llm = ChatDatabricks(model="test-model", temperature=0.7, max_tokens=100, stop=["stop"], n=2)

    messages = [HumanMessage(content="Hello")]
    result = llm._prepare_inputs(messages)

    expected = {
        "temperature": 0.7,
        "max_tokens": 100,
        "stop": ["stop"],
        "n": 2,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    assert result == expected


def test_prepare_inputs_with_responses_api():
    """Test _prepare_inputs method with responses API."""
    llm = ChatDatabricks(model="test-model", use_responses_api=True, temperature=0.5)

    messages = [HumanMessage(content="Hello")]
    result = llm._prepare_inputs(messages)
    expected = {
        "temperature": 0.5,
        "input": [{"role": "user", "content": "Hello"}],
        "n": 1,
    }

    assert result == expected


def test_prepare_inputs_override_stop():
    """Test _prepare_inputs method with stop parameter override."""
    llm = ChatDatabricks(model="test-model", stop=["default_stop"])

    messages = [HumanMessage(content="Hello")]
    result = llm._prepare_inputs(messages, stop=["override_stop"])

    # The implementation uses "self.stop or stop" which means if self.stop is truthy, it uses self.stop
    # This is the current behavior based on line 319: if stop := self.stop or stop:
    assert result["stop"] == ["default_stop"]


def test_prepare_inputs_with_kwargs():
    """Test _prepare_inputs method with additional kwargs."""
    llm = ChatDatabricks(model="test-model")

    messages = [HumanMessage(content="Hello")]
    result = llm._prepare_inputs(messages, custom_param="value")

    assert result["custom_param"] == "value"


def test_prepare_inputs_with_extra_params():
    """Test _prepare_inputs method with extra_params."""
    llm = ChatDatabricks(model="test-model", extra_params={"param1": "value1"})

    messages = [HumanMessage(content="Hello")]
    result = llm._prepare_inputs(messages, param2="value2")

    assert result["param1"] == "value1"
    assert result["param2"] == "value2"
