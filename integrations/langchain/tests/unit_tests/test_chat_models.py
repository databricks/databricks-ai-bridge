"""Test chat model integration."""

import json
from unittest import mock

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
    ToolMessageChunk,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.runnables import RunnableMap
from pydantic import BaseModel, Field

from databricks_langchain.chat_models import (
    ChatDatabricks,
    _convert_dict_to_message,
    _convert_dict_to_message_chunk,
    _convert_message_to_dict,
    _convert_responses_chunk_to_generation_chunk,
    _convert_responses_to_chat_result,
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
        ("any_role", ChatMessage(content="foo", role="any_role")),
    ],
)
def test_convert_message(role: str, expected_output: BaseMessage) -> None:
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


def test_convert_message_chunk_developer_role() -> None:
    """Test that developer role is handled correctly as SystemMessageChunk with special kwargs."""
    delta = {"role": "developer", "content": "System instructions", "id": "msg_123"}
    result = _convert_dict_to_message_chunk(delta, "default_role")
    expected_output = SystemMessageChunk(
        content="System instructions", 
        id="msg_123", 
        additional_kwargs={"__openai_role__": "developer"}
    )
    assert result == expected_output


def test_convert_message_chunk_with_function_call() -> None:
    """Test that function calls are properly handled in assistant message chunks."""
    delta = {
        "role": "assistant",
        "content": "",
        "function_call": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
        "id": "msg_456"
    }
    result = _convert_dict_to_message_chunk(delta, "default_role")
    expected_output = AIMessageChunk(
        content="",
        id="msg_456",
        additional_kwargs={"function_call": {"name": "get_weather", "arguments": '{"location": "NYC"}'}},
        tool_call_chunks=[]
    )
    assert result == expected_output


def test_convert_message_chunk_with_function_call_none_name() -> None:
    """Test that function calls with None name are handled correctly."""
    delta = {
        "role": "assistant", 
        "content": "",
        "function_call": {"name": None, "arguments": "{}"},
        "id": "msg_789"
    }
    result = _convert_dict_to_message_chunk(delta, "default_role")
    expected_output = AIMessageChunk(
        content="",
        id="msg_789", 
        additional_kwargs={"function_call": {"name": "", "arguments": "{}"}},
        tool_call_chunks=[]
    )
    assert result == expected_output


def test_convert_message_chunk_id_propagation() -> None:
    """Test that IDs are properly propagated to all message chunk types."""
    test_cases = [
        ("user", "user_123", HumanMessageChunk(content="test", id="user_123")),
        ("system", "sys_123", SystemMessageChunk(content="test", id="sys_123")),
        ("assistant", "ai_123", AIMessageChunk(content="test", id="ai_123", tool_call_chunks=[])),
        ("any_role", "chat_123", ChatMessageChunk(content="test", role="any_role", id="chat_123")),
    ]
    
    for role, test_id, expected_output in test_cases:
        delta = {"role": role, "content": "test", "id": test_id}
        result = _convert_dict_to_message_chunk(delta, "default_role")
        assert result == expected_output


def test_convert_message_chunk_tool_calls_with_default_index() -> None:
    """Test that tool calls with missing index get default value of 0."""
    delta = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "test_func", "arguments": "{}"}, "id": "call_123"}],
        "id": "msg_tool"
    }
    result = _convert_dict_to_message_chunk(delta, "default_role")
    expected_output = AIMessageChunk(
        content="",
        id="msg_tool",
        additional_kwargs={"tool_calls": delta["tool_calls"]},
        tool_call_chunks=[ToolCallChunk(name="test_func", args="{}", id="call_123", index=0)]
    )
    assert result == expected_output


def test_convert_message_to_dict_function() -> None:
    with pytest.raises(ValueError, match="Function messages are not supported"):
        _convert_message_to_dict(FunctionMessage(content="", name="name"))


def test_convert_response_to_chat_result_llm_output(llm: ChatDatabricks) -> None:
    """Test that _convert_response_to_chat_result correctly sets llm_output."""

    result = llm._convert_response_to_chat_result(_MOCK_CHAT_RESPONSE)

    # Verify that llm_output contains the full response metadata
    assert "model_name" in result.llm_output
    assert "usage" in result.llm_output
    assert result.llm_output["model_name"] == _MOCK_CHAT_RESPONSE["model"]

    # Verify that usage information is included directly in llm_output
    assert result.llm_output["usage"] == _MOCK_CHAT_RESPONSE["usage"]

    # Verify that choices, content, role, and type are excluded from llm_output
    assert "choices" not in result.llm_output
    assert "content" not in result.llm_output
    assert "role" not in result.llm_output
    assert "type" not in result.llm_output


### ResponsesAgent API Tests ###

# Mock ResponsesAgent streaming response data
_MOCK_RESPONSES_STREAM = [
    {
        "type": "response.output_text.delta",
        "delta": "To calculate"
    },
    {
        "type": "response.output_text.delta", 
        "delta": " the result:"
    },
    {
        "type": "response.output_text.delta",
        "delta": " 36939 * 8922.4 = 329,511,111.6"
    },
    {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "output_text": {
                "text": "To calculate the result: 36939 * 8922.4 = 329,511,111.6"
            },
            "annotations": {
                "finish_reason": "stop"
            }
        }
    }
]

# Mock ResponsesAgent non-streaming response
_MOCK_RESPONSES_RESPONSE = {
    "id": "response_id",
    "object": "response",
    "created": 1721875529,
    "model": "responses-agent-model",
    "output": [
        {
            "type": "message",
            "output_text": {
                "text": "To calculate the result: 36939 * 8922.4 = 329,511,111.6"
            },
            "annotations": {
                "finish_reason": "stop"
            }
        }
    ]
}

# Mock ResponsesAgent function call streaming response
_MOCK_RESPONSES_FUNCTION_STREAM = [
    {
        "type": "response.output_text.delta",
        "delta": "I'll help you get the weather."
    },
    {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "function_call": {
                "id": "call_123",
                "name": "GetWeather",
                "arguments": '{"location": "San Francisco, CA"}'
            }
        }
    }
]

# Mock ResponsesAgent function call non-streaming response
_MOCK_RESPONSES_FUNCTION_RESPONSE = {
    "id": "response_id",
    "object": "response", 
    "created": 1721875529,
    "model": "responses-agent-model",
    "output": [
        {
            "type": "function_call",
            "function_call": {
                "id": "call_123",
                "name": "GetWeather",
                "arguments": {"location": "San Francisco, CA"}
            }
        }
    ]
}


def test_convert_responses_chunk_to_generation_chunk_text_delta():
    """Test conversion of response.output_text.delta chunks."""
    chunk = {
        "type": "response.output_text.delta",
        "delta": "Hello world"
    }
    
    result = _convert_responses_chunk_to_generation_chunk(chunk)
    
    assert result is not None
    assert result.message.content == "Hello world"
    assert isinstance(result.message, AIMessageChunk)


def test_convert_responses_chunk_to_generation_chunk_message_done():
    """Test conversion of response.output_item.done chunks with message type."""
    chunk = {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "output_text": {
                "text": "Complete response text"
            },
            "annotations": {
                "finish_reason": "stop"
            }
        }
    }
    
    result = _convert_responses_chunk_to_generation_chunk(chunk)
    
    assert result is not None
    assert result.message.content == "Complete response text"
    assert result.message.additional_kwargs["annotations"]["finish_reason"] == "stop"
    assert result.generation_info["finish_reason"] == "stop"
    assert isinstance(result.message, AIMessageChunk)


def test_convert_responses_chunk_to_generation_chunk_function_call():
    """Test conversion of response.output_item.done chunks with function_call type."""
    chunk = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "function_call": {
                "id": "call_123",
                "name": "GetWeather",
                "arguments": '{"location": "San Francisco, CA"}'
            }
        }
    }
    
    result = _convert_responses_chunk_to_generation_chunk(chunk)
    
    assert result is not None
    assert result.message.content == ""
    assert len(result.message.tool_call_chunks) == 1
    
    tool_call = result.message.tool_call_chunks[0]
    assert tool_call["name"] == "GetWeather"
    assert tool_call["args"] == '{"location": "San Francisco, CA"}'
    assert tool_call["id"] == "call_123"
    assert tool_call["index"] == 0


def test_convert_responses_chunk_to_generation_chunk_function_call_output():
    """Test conversion of response.output_item.done chunks with function_call_output type."""
    chunk = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call_output",
            "output": "The weather is sunny and 75°F"
        }
    }
    
    result = _convert_responses_chunk_to_generation_chunk(chunk)
    
    assert result is not None
    assert result.message.content == "The weather is sunny and 75°F"
    assert isinstance(result.message, AIMessageChunk)


def test_convert_responses_chunk_to_generation_chunk_unknown_type():
    """Test that unknown chunk types return None."""
    chunk = {
        "type": "unknown.type",
        "data": "some data"
    }
    
    result = _convert_responses_chunk_to_generation_chunk(chunk)
    assert result is None


def test_convert_responses_to_chat_result_message():
    """Test conversion of ResponsesAgent response with message output."""
    response = {
        "id": "response_id",
        "model": "responses-agent-model",
        "output": [
            {
                "type": "message",
                "output_text": {
                    "text": "Hello, how can I help you?"
                },
                "annotations": {
                    "finish_reason": "stop"
                }
            }
        ]
    }
    
    result = _convert_responses_to_chat_result(response)
    
    assert len(result.generations) == 1
    generation = result.generations[0]
    assert generation.message.content == "Hello, how can I help you?"
    assert generation.message.additional_kwargs["annotations"]["finish_reason"] == "stop"
    assert isinstance(generation.message, AIMessage)
    
    # Check llm_output
    assert result.llm_output["id"] == "response_id"
    assert result.llm_output["model"] == "responses-agent-model"
    assert "output" not in result.llm_output
    assert "choices" not in result.llm_output


def test_convert_responses_to_chat_result_function_call():
    """Test conversion of ResponsesAgent response with function_call output."""
    response = {
        "id": "response_id",
        "model": "responses-agent-model",
        "output": [
            {
                "type": "function_call",
                "function_call": {
                    "id": "call_123",
                    "name": "GetWeather",
                    "arguments": {"location": "San Francisco, CA"}
                }
            }
        ]
    }
    
    result = _convert_responses_to_chat_result(response)
    
    assert len(result.generations) == 1
    generation = result.generations[0]
    assert generation.message.content == ""
    assert len(generation.message.tool_calls) == 1
    
    tool_call = generation.message.tool_calls[0]
    assert tool_call["id"] == "call_123"
    assert tool_call["name"] == "GetWeather"
    assert tool_call["args"] == {"location": "San Francisco, CA"}
    assert isinstance(generation.message, AIMessage)


def test_convert_responses_to_chat_result_empty_output():
    """Test conversion with empty output creates default generation."""
    response = {
        "id": "response_id",
        "model": "responses-agent-model",
        "output": []
    }
    
    result = _convert_responses_to_chat_result(response)
    
    assert len(result.generations) == 1
    generation = result.generations[0]
    assert generation.message.content == ""
    assert isinstance(generation.message, AIMessage)


def test_convert_responses_to_chat_result_single_output():
    """Test conversion with single output (not in list)."""
    response = {
        "id": "response_id",
        "model": "responses-agent-model",
        "output": {
            "type": "message",
            "output_text": {
                "text": "Single output message"
            }
        }
    }
    
    result = _convert_responses_to_chat_result(response)
    
    assert len(result.generations) == 1
    generation = result.generations[0]
    assert generation.message.content == "Single output message"


def test_chat_databricks_use_responses_api_flag():
    """Test that use_responses_api flag is properly set."""
    llm = ChatDatabricks(
        model="responses-agent-model",
        use_responses_api=True
    )
    
    assert llm.use_responses_api is True
    
    # Test default value
    llm_default = ChatDatabricks(model="standard-model")
    assert llm_default.use_responses_api is False


@pytest.fixture
def responses_llm() -> ChatDatabricks:
    """Fixture for ChatDatabricks with responses API enabled."""
    return ChatDatabricks(
        model="responses-agent-model",
        use_responses_api=True,
        target_uri="databricks"
    )


def test_chat_model_generate_with_responses_api(responses_llm: ChatDatabricks):
    """Test _generate method with responses API enabled."""
    # Mock the client to return responses API format
    with mock.patch.object(responses_llm.client, 'predict', return_value=_MOCK_RESPONSES_RESPONSE):
        result = responses_llm._generate([HumanMessage(content="Test message")])
        
        assert len(result.generations) == 1
        generation = result.generations[0]
        assert generation.message.content == "To calculate the result: 36939 * 8922.4 = 329,511,111.6"
        assert isinstance(generation.message, AIMessage)


def test_chat_model_stream_with_responses_api(responses_llm: ChatDatabricks):
    """Test _stream method with responses API enabled."""
    # Mock the client to return responses API format
    with mock.patch.object(responses_llm.client, 'predict_stream', return_value=_MOCK_RESPONSES_STREAM):
        chunks = list(responses_llm._stream([HumanMessage(content="Test message")]))
        
        # Should have 4 chunks: 3 text deltas + 1 final message
        assert len(chunks) == 4
        
        # Check text delta chunks
        assert chunks[0].message.content == "To calculate"
        assert chunks[1].message.content == " the result:"
        assert chunks[2].message.content == " 36939 * 8922.4 = 329,511,111.6"
        
        # Check final message chunk
        assert chunks[3].message.content == "To calculate the result: 36939 * 8922.4 = 329,511,111.6"
        assert chunks[3].generation_info["finish_reason"] == "stop"


def test_chat_model_stream_with_responses_api_function_calls(responses_llm: ChatDatabricks):
    """Test _stream method with responses API and function calls."""
    with mock.patch.object(responses_llm.client, 'predict_stream', return_value=_MOCK_RESPONSES_FUNCTION_STREAM):
        chunks = list(responses_llm._stream([HumanMessage(content="Get weather for SF")]))
        
        # Should have 2 chunks: 1 text delta + 1 function call
        assert len(chunks) == 2
        
        # Check text chunk
        assert chunks[0].message.content == "I'll help you get the weather."
        
        # Check function call chunk
        assert chunks[1].message.content == ""
        assert len(chunks[1].message.tool_call_chunks) == 1
        tool_call = chunks[1].message.tool_call_chunks[0]
        assert tool_call["name"] == "GetWeather"
        assert tool_call["id"] == "call_123"


def test_chat_model_generate_with_responses_api_function_calls(responses_llm: ChatDatabricks):
    """Test _generate method with responses API and function calls."""
    with mock.patch.object(responses_llm.client, 'predict', return_value=_MOCK_RESPONSES_FUNCTION_RESPONSE):
        result = responses_llm._generate([HumanMessage(content="Get weather for SF")])
        
        assert len(result.generations) == 1
        generation = result.generations[0]
        assert generation.message.content == ""
        assert len(generation.message.tool_calls) == 1
        
        tool_call = generation.message.tool_calls[0]
        assert tool_call["name"] == "GetWeather"
        assert tool_call["id"] == "call_123"
        assert tool_call["args"] == {"location": "San Francisco, CA"}


def test_chat_model_backward_compatibility():
    """Test that standard API still works when use_responses_api=False."""
    llm = ChatDatabricks(
        model="standard-model",
        use_responses_api=False,
        target_uri="databricks"
    )
    
    # Mock standard API response
    with mock.patch.object(llm.client, 'predict', return_value=_MOCK_CHAT_RESPONSE):
        result = llm._generate([HumanMessage(content="Test message")])
        
        # Should use standard conversion
        assert len(result.generations) == 1
        generation = result.generations[0]
        assert "To calculate the result of 36939 multiplied by 8922.4" in generation.message.content


def test_chat_model_stream_backward_compatibility():
    """Test that standard streaming API still works when use_responses_api=False."""
    llm = ChatDatabricks(
        model="standard-model", 
        use_responses_api=False,
        target_uri="databricks"
    )
    
    # Mock standard API streaming response
    with mock.patch.object(llm.client, 'predict_stream', return_value=_MOCK_STREAM_RESPONSE):
        chunks = list(llm._stream([HumanMessage(content="Test message")]))
        
        # Should use standard conversion and have same number of chunks as mock
        assert len(chunks) == len(_MOCK_STREAM_RESPONSE)
        assert chunks[0].message.content == "36939"
        assert chunks[1].message.content == "x"
