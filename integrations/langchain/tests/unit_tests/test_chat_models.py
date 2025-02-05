"""Test chat model integration."""

import json

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


def test_convert_message_to_dict_function() -> None:
    with pytest.raises(ValueError, match="Function messages are not supported"):
        _convert_message_to_dict(FunctionMessage(content="", name="name"))
