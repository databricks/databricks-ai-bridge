"""Test chat model integration."""

import json
from types import SimpleNamespace
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
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    Response,
    ResponseErrorEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
)
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
    # workspace_client is excluded from serialization
    assert "workspace_client" not in d


def test_workspace_client_parameter() -> None:
    """Test the workspace_client parameter works correctly."""
    from unittest.mock import Mock, patch

    mock_workspace_client = Mock()
    mock_openai_client = Mock()

    with patch(
        "databricks_langchain.chat_models.get_openai_client", return_value=mock_openai_client
    ) as mock_get_client:
        llm = ChatDatabricks(model="test-model", workspace_client=mock_workspace_client)

    assert llm.client == mock_openai_client
    mock_get_client.assert_called_once_with(workspace_client=mock_workspace_client)


def test_workspace_client_and_target_uri_conflict() -> None:
    """Test that specifying both workspace_client and target_uri raises ValueError."""
    from unittest.mock import Mock

    mock_workspace_client = Mock()
    with pytest.raises(ValueError, match="Cannot specify both 'workspace_client' and 'target_uri'"):
        ChatDatabricks(
            model="test-model", workspace_client=mock_workspace_client, target_uri="databricks"
        )


def test_timeout_and_max_retries_parameters() -> None:
    """Test that timeout and max_retries parameters are properly passed to the OpenAI client."""
    from unittest.mock import Mock, patch

    mock_openai_client = Mock()
    mock_openai_client.timeout = None
    mock_openai_client.max_retries = None

    with patch(
        "databricks_langchain.chat_models.get_openai_client", return_value=mock_openai_client
    ) as mock_get_client:
        # Test with timeout and max_retries
        llm = ChatDatabricks(model="test-model", timeout=60.0, max_retries=5)

    # Verify get_openai_client was called with the correct parameters
    mock_get_client.assert_called_once_with(workspace_client=None, timeout=60.0, max_retries=5)

    # Test that client is set
    assert llm.client == mock_openai_client
    assert llm.timeout == 60.0
    assert llm.max_retries == 5


def test_timeout_and_max_retries_with_workspace_client() -> None:
    """Test timeout and max_retries parameters work with workspace_client."""
    from unittest.mock import Mock, patch

    mock_workspace_client = Mock()
    mock_openai_client = Mock()
    mock_openai_client.timeout = None
    mock_openai_client.max_retries = None

    with patch(
        "databricks_langchain.chat_models.get_openai_client", return_value=mock_openai_client
    ) as mock_get_client:
        llm = ChatDatabricks(
            model="test-model", workspace_client=mock_workspace_client, timeout=30.0, max_retries=2
        )

    # Verify get_openai_client was called with all parameters
    mock_get_client.assert_called_once_with(
        workspace_client=mock_workspace_client, timeout=30.0, max_retries=2
    )

    assert llm.client == mock_openai_client
    assert llm.timeout == 30.0
    assert llm.max_retries == 2


def test_default_workspace_client() -> None:
    """Test that default WorkspaceClient is created when none provided."""
    from unittest.mock import Mock, patch

    mock_workspace_client = Mock()
    mock_openai_client = Mock()
    mock_workspace_client.serving_endpoints.get_open_ai_client.return_value = mock_openai_client

    # Patch both WorkspaceClient and get_openai_client
    with patch("databricks.sdk.WorkspaceClient", return_value=mock_workspace_client):
        with patch(
            "databricks_langchain.chat_models.get_openai_client", return_value=mock_openai_client
        ) as mock_get_client:
            llm = ChatDatabricks(model="test-model")

    assert llm.client == mock_openai_client
    mock_get_client.assert_called_once_with(workspace_client=None)


def test_target_uri_deprecation_warning() -> None:
    """Test that using target_uri shows deprecation warning."""
    from unittest.mock import Mock, patch

    mock_workspace_client = Mock()
    mock_openai_client = Mock()
    mock_workspace_client.serving_endpoints.get_open_ai_client.return_value = mock_openai_client

    with patch("databricks.sdk.WorkspaceClient", return_value=mock_workspace_client):
        with pytest.warns(DeprecationWarning, match="The 'target_uri' parameter is deprecated"):
            ChatDatabricks(model="test-model", target_uri="databricks")


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


def test_chat_model_stream_usage_chunk_emission():
    """Test that stream_usage=True emits a final usage-only chunk with empty content by default."""
    from unittest.mock import Mock, patch

    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5

    mock_chunks = [
        Mock(
            choices=[
                Mock(
                    delta=Mock(
                        role="assistant",
                        content="Hello",
                        model_dump=Mock(return_value={"role": "assistant", "content": "Hello"}),
                    ),
                    finish_reason="stop",
                )
            ],
            usage=mock_usage,
        ),
    ]

    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(mock_chunks)

        llm = ChatDatabricks(model="test-model")
        messages = [HumanMessage(content="Hello")]

        # Test with stream_usage=True
        chunks = list(llm.stream(messages))

        # Find the usage chunk (empty content with usage_metadata)
        usage_chunks = [
            chunk for chunk in chunks if chunk.content == "" and chunk.usage_metadata is not None
        ]
        assert len(usage_chunks) == 1

        # Verify usage chunk structure
        usage_chunk = usage_chunks[0]
        assert isinstance(usage_chunk, AIMessageChunk)
        assert usage_chunk.content == ""
        assert usage_chunk.usage_metadata["input_tokens"] == 10
        assert usage_chunk.usage_metadata["output_tokens"] == 5
        assert usage_chunk.usage_metadata["total_tokens"] == 15


def test_chat_model_stream_no_duplicate_usage_chunks():
    """Test that usage_chunk_emitted flag prevents duplicate usage chunks."""
    from unittest.mock import Mock, patch

    mock_usage = Mock()
    mock_usage.prompt_tokens = 20
    mock_usage.completion_tokens = 8

    # Multiple chunks with usage data to test the duplicate prevention logic
    mock_chunks = [
        Mock(
            choices=[
                Mock(
                    delta=Mock(
                        role="assistant",
                        content="Hello",
                        model_dump=Mock(return_value={"role": "assistant", "content": "Hello"}),
                    ),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
            usage=mock_usage,
        ),
        Mock(
            choices=[
                Mock(
                    delta=Mock(
                        role="assistant",
                        content=" world",
                        model_dump=Mock(return_value={"role": "assistant", "content": " world"}),
                    ),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=mock_usage,
        ),
    ]

    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(mock_chunks)

        llm = ChatDatabricks(model="test-model")
        messages = [HumanMessage(content="Hello")]

        chunks = list(llm.stream(messages, stream_usage=True))

        # Should emit exactly ONE usage chunk despite multiple chunks having usage data
        usage_chunks = [
            chunk for chunk in chunks if chunk.content == "" and chunk.usage_metadata is not None
        ]
        assert len(usage_chunks) == 1, f"Expected exactly 1 usage chunk, got {len(usage_chunks)}"


def test_chat_model_stream_usage_only_final_chunk():
    """Test that a final chunk with only usage data (no choices) correctly emits usage metadata."""
    from unittest.mock import Mock, patch

    mock_usage = Mock()
    mock_usage.prompt_tokens = 15
    mock_usage.completion_tokens = 10

    # Simulate GPT-5 streaming behavior: content chunks followed by usage-only chunk
    mock_chunks = [
        Mock(
            choices=[
                Mock(
                    delta=Mock(
                        role="assistant",
                        content="Hello",
                        model_dump=Mock(return_value={"role": "assistant", "content": "Hello"}),
                    ),
                    finish_reason=None,
                    logprobs=None,
                )
            ],
            usage=None,
        ),
        Mock(
            choices=[
                Mock(
                    delta=Mock(
                        role="assistant",
                        content=" world",
                        model_dump=Mock(return_value={"role": "assistant", "content": " world"}),
                    ),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=None,
        ),
        # Final chunk with ONLY usage data, no choices/delta
        Mock(
            choices=[],
            usage=mock_usage,
        ),
    ]

    # Verify mock structure matches GPT-5 behavior
    # Final chunk has empty choices list and usage data (no delta)
    assert len(mock_chunks[2].choices) == 0
    assert mock_chunks[2].usage is not None

    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(mock_chunks)

        llm = ChatDatabricks(model="test-model")
        messages = [HumanMessage(content="Hello")]

        chunks = list(llm.stream(messages, stream_usage=True))

        # Should get content chunks plus one usage chunk
        content_chunks = [chunk for chunk in chunks if chunk.content != ""]
        assert len(content_chunks) == 2
        assert content_chunks[0].content == "Hello"
        assert content_chunks[1].content == " world"

        # Should emit exactly ONE usage chunk
        usage_chunks = [
            chunk for chunk in chunks if chunk.content == "" and chunk.usage_metadata is not None
        ]
        assert len(usage_chunks) == 1, f"Expected exactly 1 usage chunk, got {len(usage_chunks)}"

        # Verify usage chunk has correct metadata
        usage_chunk = usage_chunks[0]
        assert isinstance(usage_chunk, AIMessageChunk)
        assert usage_chunk.content == ""
        assert usage_chunk.usage_metadata["input_tokens"] == 15
        assert usage_chunk.usage_metadata["output_tokens"] == 10
        assert usage_chunk.usage_metadata["total_tokens"] == 25


def test_chat_model_stream_usage_only_chunk_missing_tokens():
    """Test that a usage-only chunk with missing token data doesn't emit usage metadata."""
    from unittest.mock import Mock, patch

    mock_usage = Mock()
    mock_usage.prompt_tokens = None  # Missing prompt_tokens
    mock_usage.completion_tokens = 10

    mock_chunks = [
        Mock(
            choices=[
                Mock(
                    delta=Mock(
                        role="assistant",
                        content="Hello",
                        model_dump=Mock(return_value={"role": "assistant", "content": "Hello"}),
                    ),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=None,
        ),
        # Final chunk with usage data but missing prompt_tokens
        Mock(
            choices=[],
            usage=mock_usage,
        ),
    ]

    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(mock_chunks)

        llm = ChatDatabricks(model="test-model")
        messages = [HumanMessage(content="Hello")]

        chunks = list(llm.stream(messages, stream_usage=True))

        # Should get content chunks but NO usage chunk (due to missing tokens)
        content_chunks = [chunk for chunk in chunks if chunk.content != ""]
        assert len(content_chunks) == 1

        # Should NOT emit a usage chunk when tokens are missing
        usage_chunks = [
            chunk for chunk in chunks if chunk.content == "" and chunk.usage_metadata is not None
        ]
        assert len(usage_chunks) == 0, (
            f"Expected 0 usage chunks when tokens are missing, got {len(usage_chunks)}"
        )


def test_chat_model_stream_usage_only_chunk_stream_usage_false():
    """Test that a usage-only chunk is ignored when stream_usage=False."""
    from unittest.mock import Mock, patch

    mock_usage = Mock()
    mock_usage.prompt_tokens = 15
    mock_usage.completion_tokens = 10

    mock_chunks = [
        Mock(
            choices=[
                Mock(
                    delta=Mock(
                        role="assistant",
                        content="Hello",
                        model_dump=Mock(return_value={"role": "assistant", "content": "Hello"}),
                    ),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=None,
        ),
        # Final chunk with usage data
        Mock(
            choices=[],
            usage=mock_usage,
        ),
    ]

    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter(mock_chunks)

        llm = ChatDatabricks(model="test-model")
        messages = [HumanMessage(content="Hello")]

        chunks = list(llm.stream(messages, stream_usage=False))

        # Should get content chunks only
        content_chunks = [chunk for chunk in chunks if chunk.content != ""]
        assert len(content_chunks) == 1

        # Should NOT emit a usage chunk when stream_usage=False
        usage_chunks = [
            chunk for chunk in chunks if chunk.content == "" and chunk.usage_metadata is not None
        ]
        assert len(usage_chunks) == 0, (
            f"Expected 0 usage chunks when stream_usage=False, got {len(usage_chunks)}"
        )


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
    # Create OpenAI objects from mock data
    expected_content = _MOCK_CHAT_RESPONSE["choices"][0]["message"]["content"]
    message = ChatCompletionMessage(role="assistant", content=expected_content, tool_calls=None)
    choice = Choice(index=0, message=message, finish_reason="stop", logprobs=None)
    usage = CompletionUsage(**_MOCK_CHAT_RESPONSE["usage"])
    response = ChatCompletion(
        id=_MOCK_CHAT_RESPONSE["id"],
        choices=[choice],
        created=_MOCK_CHAT_RESPONSE["created"],
        model=_MOCK_CHAT_RESPONSE["model"],
        object="chat.completion",
        usage=usage,
    )

    result = llm._convert_response_to_chat_result(response)

    expected = ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content=expected_content),
                generation_info={"finish_reason": "stop"},
            ),
        ],
        llm_output={
            "usage": _MOCK_CHAT_RESPONSE["usage"],
            "prompt_tokens": _MOCK_CHAT_RESPONSE["usage"]["prompt_tokens"],
            "completion_tokens": _MOCK_CHAT_RESPONSE["usage"]["completion_tokens"],
            "total_tokens": _MOCK_CHAT_RESPONSE["usage"]["total_tokens"],
            "model": _MOCK_CHAT_RESPONSE["model"],
            "model_name": _MOCK_CHAT_RESPONSE["model"],
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
    chunk = ResponseTextDeltaEvent.model_construct(
        type="response.output_text.delta", item_id="item_123", delta="Hello"
    )

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)

    assert isinstance(result, AIMessageChunk)
    assert result.content == [{"type": "text", "text": "Hello"}]
    assert result.id == "item_123"


def test_convert_responses_api_chunk_to_lc_chunk_function_call():
    """Test _convert_responses_api_chunk_to_lc_chunk with function call."""
    chunk = ResponseOutputItemDoneEvent.model_construct(
        type="response.output_item.done",
        item=ResponseFunctionToolCall.model_construct(
            type="function_call",
            id="item_123",
            call_id="call_456",
            name="get_weather",
            arguments='{"location": "SF"}',
        ),
    )

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)
    expected = AIMessageChunk(
        id="call_456",
        content=[],
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"location": "SF"},
                "id": "call_456",
                "type": "tool_call",
            }
        ],
        tool_call_chunks=[
            ToolCallChunk(name="get_weather", args='{"location": "SF"}', id="call_456")
        ],
    )
    assert result == expected


def test_convert_responses_api_chunk_to_lc_chunk_function_call_output():
    chunk = ResponseOutputItemDoneEvent.model_construct(
        type="response.output_item.done",
        item=ResponseFunctionToolCallOutputItem.model_construct(
            type="function_call_output", call_id="call_456", output="Sunny, 72°F"
        ),
    )

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)

    assert isinstance(result, ToolMessageChunk)
    assert result.content == "Sunny, 72°F"
    assert result.tool_call_id == "call_456"


def test_convert_responses_api_chunk_to_lc_chunk_message():
    """Test _convert_responses_api_chunk_to_lc_chunk with message."""
    chunk = ResponseOutputItemDoneEvent.model_construct(
        type="response.output_item.done",
        item=ResponseOutputMessage.model_construct(
            type="message",
            id="msg_123",
            content=[
                ResponseOutputText.model_construct(
                    type="output_text", text="Hello!", annotations=[{"key": "value"}]
                ),
                ResponseOutputRefusal.model_construct(
                    type="refusal", refusal="I cannot help with that."
                ),
            ],
        ),
    )

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)

    assert isinstance(result, AIMessageChunk)
    assert result.id == "msg_123"
    assert len(result.content) == 2

    # Check text content with annotations
    text_content = result.content[0]
    assert text_content["type"] == "text"
    assert text_content["text"] == "Hello!"
    assert len(text_content["annotations"]) == 1
    # Check that annotation was converted to dict and contains our key
    annotation = text_content["annotations"][0]
    assert isinstance(annotation, dict)
    assert annotation["key"] == "value"

    # Check refusal content
    refusal_content = result.content[1]
    assert refusal_content["type"] == "refusal"
    assert refusal_content["refusal"] == "I cannot help with that."


def test_convert_responses_api_chunk_to_lc_chunk_skip_duplicate():
    """Test _convert_responses_api_chunk_to_lc_chunk skips duplicate text."""
    previous_chunk = ResponseTextDeltaEvent.model_construct(
        type="response.output_text.delta", item_id="item_123", delta="Hello"
    )

    chunk = ResponseOutputItemDoneEvent.model_construct(
        type="response.output_item.done",
        item=ResponseOutputMessage.model_construct(
            type="message",
            id="item_123",
            content=[ResponseOutputText.model_construct(type="output_text", text="Hello")],
        ),
    )

    result = _convert_responses_api_chunk_to_lc_chunk(chunk, previous_chunk)
    assert result is None


def test_convert_responses_api_chunk_to_lc_chunk_skip_duplicate_with_annotations():
    """Test _convert_responses_api_chunk_to_lc_chunk skips duplicate text."""
    previous_chunk = ResponseTextDeltaEvent.model_construct(
        type="response.output_text.delta", item_id="item_123", delta="Hello"
    )

    chunk = ResponseOutputItemDoneEvent.model_construct(
        type="response.output_item.done",
        item=ResponseOutputMessage.model_construct(
            type="message",
            id="item_123",
            content=[
                ResponseOutputText.model_construct(
                    type="output_text",
                    text="Hello",
                    annotations=[{"type": "url_citation", "title": "title", "url": "google.com"}],
                )
            ],
        ),
    )

    result = _convert_responses_api_chunk_to_lc_chunk(chunk, previous_chunk)

    assert isinstance(result, AIMessageChunk)
    assert result.id == "item_123"
    assert len(result.content) == 1

    # Check that annotations were included and converted to dict
    annotation_content = result.content[0]
    assert "annotations" in annotation_content
    assert len(annotation_content["annotations"]) == 1

    annotation = annotation_content["annotations"][0]
    assert isinstance(annotation, dict)
    assert annotation["type"] == "url_citation"
    assert annotation["title"] == "title"
    assert annotation["url"] == "google.com"


def test_convert_responses_api_chunk_to_lc_chunk_error():
    """Test _convert_responses_api_chunk_to_lc_chunk with error."""
    chunk = ResponseErrorEvent.model_construct(type="error", error="Something went wrong")

    with pytest.raises(ValueError, match="Something went wrong"):
        _convert_responses_api_chunk_to_lc_chunk(chunk)


def test_convert_responses_api_chunk_to_lc_chunk_unknown_type():
    """Test _convert_responses_api_chunk_to_lc_chunk with unknown type."""

    # Create a simple object for unknown type
    class UnknownEvent:
        def __init__(self):
            self.type = "unknown_type"
            self.data = "some data"

    chunk = UnknownEvent()

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)
    assert result is None


def test_convert_responses_api_chunk_to_lc_chunk_special_items():
    """Test _convert_responses_api_chunk_to_lc_chunk with special item types."""
    chunk = ResponseOutputItemDoneEvent.model_construct(
        type="response.output_item.done",
        item=ResponseReasoningItem.model_construct(
            type="reasoning",
            summary=[{"type": "summary_text", "text": "Let me think about this..."}],
        ),
    )

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)

    assert isinstance(result, AIMessageChunk)
    assert len(result.content) == 1

    reasoning_content = result.content[0]
    assert isinstance(reasoning_content, dict)
    assert reasoning_content["type"] == "reasoning"
    assert "summary" in reasoning_content
    assert len(reasoning_content["summary"]) == 1
    assert reasoning_content["summary"][0]["type"] == "summary_text"
    assert reasoning_content["summary"][0]["text"] == "Let me think about this..."


### Test ChatDatabricks response conversion methods ###


def test_convert_responses_api_response_to_chat_result():
    """Test _convert_responses_api_response_to_chat_result method."""
    llm = ChatDatabricks(model="test-model", use_responses_api=True)

    response = Response.model_construct(
        id="response_123",
        output=[
            ResponseOutputMessage.model_construct(
                type="message",
                content=[
                    ResponseOutputText.model_construct(
                        type="output_text",
                        text="Hello!",
                        id="text_123",
                        annotations=[{"key": "value"}],
                    )
                ],
            ),
            ResponseFunctionToolCall.model_construct(
                type="function_call",
                name="get_weather",
                arguments='{"location": "SF"}',
                call_id="call_123",
            ),
        ],
    )

    result = llm._convert_responses_api_response_to_chat_result(response)

    assert isinstance(result, ChatResult)
    assert len(result.generations) == 1

    generation = result.generations[0]
    assert isinstance(generation, ChatGeneration)

    message = generation.message
    assert isinstance(message, AIMessage)
    assert message.id == "response_123"
    assert len(message.content) == 2

    # Check text content
    text_content = message.content[0]
    assert text_content["type"] == "text"
    assert text_content["text"] == "Hello!"
    assert text_content["id"] == "text_123"
    assert len(text_content["annotations"]) == 1
    assert isinstance(text_content["annotations"][0], dict)
    assert text_content["annotations"][0]["key"] == "value"

    # Check function call content
    func_content = message.content[1]
    assert func_content["type"] == "function_call"
    assert func_content["name"] == "get_weather"
    assert func_content["arguments"] == '{"location": "SF"}'
    assert func_content["call_id"] == "call_123"

    # Check tool calls
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert tool_call["args"] == {"location": "SF"}
    assert tool_call["id"] == "call_123"
    assert tool_call["type"] == "tool_call"


def test_convert_responses_api_response_to_chat_result_with_error():
    """Test _convert_responses_api_response_to_chat_result with error."""
    llm = ChatDatabricks(model="test-model", use_responses_api=True)

    response = Response.model_construct(error="Something went wrong")

    with pytest.raises(ValueError, match="Something went wrong"):
        llm._convert_responses_api_response_to_chat_result(response)


def test_convert_chatagent_response_to_chat_result():
    """Test _convert_chatagent_response_to_chat_result method."""
    llm = ChatDatabricks(model="test-model")
    response = SimpleNamespace(messages=[{"role": "assistant", "content": "Hello from ChatAgent!"}])
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


def test_chat_databricks_responses_api_stream():
    """Test ChatDatabricks streaming with responses API using mocked client."""
    from unittest.mock import Mock, patch

    # Create mock streaming response chunks
    mock_chunks = [
        ResponseTextDeltaEvent.model_construct(
            type="response.output_text.delta", item_id="item_123", delta="Hello"
        ),
        ResponseTextDeltaEvent.model_construct(
            type="response.output_text.delta", item_id="item_123", delta=" world"
        ),
        ResponseOutputItemDoneEvent.model_construct(
            type="response.output_item.done",
            item=ResponseOutputMessage.model_construct(
                type="message",
                id="item_123",
                content=[
                    ResponseOutputText.model_construct(
                        type="output_text", text="Hello world", id="text_123"
                    )
                ],
            ),
        ),
    ]

    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock the responses.create method to return our chunks
        mock_client.responses.create.return_value = iter(mock_chunks)

        llm = ChatDatabricks(model="test-model", use_responses_api=True)

        messages = [HumanMessage(content="Hello")]
        chunks = list(llm.stream(messages))

        # Should get chunks for the text deltas but skip the duplicate done event
        assert len(chunks) == 2  # Two text delta chunks

        # Check first chunk
        assert isinstance(chunks[0], AIMessageChunk)
        assert chunks[0].content == [{"type": "text", "text": "Hello"}]
        assert chunks[0].id == "item_123"

        # Check second chunk
        assert isinstance(chunks[1], AIMessageChunk)
        assert chunks[1].content == [{"type": "text", "text": " world"}]
        assert chunks[1].id == "item_123"


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
    """Test ChatDatabricks initialization sets OpenAI client."""
    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        llm = ChatDatabricks(model="test-model")

        mock_get_client.assert_called_once_with(workspace_client=None)
        assert llm.client == mock_client


### Test ChatDatabricks _prepare_inputs method ###


def test_prepare_inputs_basic():
    llm = ChatDatabricks(model="test-model", temperature=0.7, max_tokens=100, stop=["stop"], n=2)

    messages = [HumanMessage(content="Hello")]
    result = llm._prepare_inputs(messages)

    expected = {
        "model": "test-model",
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 100,
        "stop": ["stop"],
        "n": 2,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    assert result == expected


def test_prepare_inputs_with_responses_api():
    llm = ChatDatabricks(model="test-model", use_responses_api=True, temperature=0.5)

    messages = [HumanMessage(content="Hello")]
    result = llm._prepare_inputs(messages)
    expected = {
        "model": "test-model",
        "stream": False,
        "temperature": 0.5,
        "input": [{"role": "user", "content": "Hello"}],
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


def test_convert_dict_to_message_with_non_string_content():
    """Test _convert_dict_to_message handles non-string content by JSON encoding it."""
    # Test with list of dict content (matching gpt oss)
    message_dict = {
        "role": "assistant",
        "content": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "asdf"}]},
            {"type": "text", "text": "asdf"},
        ],
    }
    result = _convert_dict_to_message(message_dict)
    expected = AIMessage(
        content='[{"type": "reasoning", "summary": [{"type": "summary_text", "text": "asdf"}]}, {"type": "text", "text": "asdf"}]'
    )
    assert result == expected


### Test custom_inputs and custom_outputs functionality ###


def test_prepare_inputs_with_custom_inputs():
    llm = ChatDatabricks(model="test-model")

    messages = [HumanMessage(content="Hello")]
    custom_inputs = {"user_id": "123", "session_id": "abc"}
    result = llm._prepare_inputs(messages, custom_inputs=custom_inputs)

    assert "extra_body" in result
    assert result["extra_body"]["custom_inputs"] == custom_inputs

    result = llm._prepare_inputs(messages, custom_inputs=None)
    # When custom_inputs is None, extra_body should not be added
    assert "extra_body" not in result


def test_generate_with_custom_inputs():
    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": "Hello!",
        }
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None
        mock_response.model = "test-model"
        mock_client.chat.completions.create.return_value = mock_response

        llm = ChatDatabricks(model="test-model")
        messages = [HumanMessage(content="Hello")]
        custom_inputs = {"user_id": "123"}

        # Call _generate with custom_inputs
        result = llm._generate(messages, custom_inputs=custom_inputs)

        # Verify client was called with prepared inputs that include custom_inputs
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "extra_body" in call_args
        assert call_args["extra_body"]["custom_inputs"] == custom_inputs


def test_stream_with_custom_inputs():
    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock streaming response
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta.model_dump.return_value = {
            "role": "assistant",
            "content": "Hello",
        }
        mock_chunk.choices[0].finish_reason = "stop"
        mock_chunk.usage = None
        mock_client.chat.completions.create.return_value = iter([mock_chunk])

        llm = ChatDatabricks(model="test-model")
        messages = [HumanMessage(content="Hello")]
        custom_inputs = {"session_id": "abc"}

        # Call stream with custom_inputs
        list(llm.stream(messages, custom_inputs=custom_inputs))

        # Verify client was called with prepared inputs that include custom_inputs
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "extra_body" in call_args
        assert call_args["extra_body"]["custom_inputs"] == custom_inputs


def test_convert_dict_to_message_with_custom_outputs():
    message_dict = {
        "role": "assistant",
        "content": "Hello!",
        "custom_outputs": {"confidence": 0.95, "reasoning": "high confidence"},
    }
    result = _convert_dict_to_message(message_dict)

    assert isinstance(result, AIMessage)
    assert result.content == "Hello!"
    assert hasattr(result, "custom_outputs")
    assert result.custom_outputs == {"confidence": 0.95, "reasoning": "high confidence"}


def test_convert_dict_to_message_without_custom_outputs():
    message_dict = {"role": "assistant", "content": "Hello!"}
    result = _convert_dict_to_message(message_dict)

    assert isinstance(result, AIMessage)
    assert result.content == "Hello!"
    # Should not have custom_outputs attribute when not provided
    assert not hasattr(result, "custom_outputs")


def test_convert_dict_to_message_chunk_with_custom_outputs():
    chunk_dict = {
        "role": "assistant",
        "content": "Hello",
        "custom_outputs": {"stream_id": "xyz123"},
    }
    result = _convert_dict_to_message_chunk(chunk_dict, "assistant")

    assert isinstance(result, AIMessageChunk)
    assert result.content == "Hello"
    assert hasattr(result, "custom_outputs")
    assert result.custom_outputs == {"stream_id": "xyz123"}


def test_convert_dict_to_message_chunk_without_custom_outputs():
    chunk_dict = {"role": "assistant", "content": "Hello"}
    result = _convert_dict_to_message_chunk(chunk_dict, "assistant")

    assert isinstance(result, AIMessageChunk)
    assert result.content == "Hello"
    # Should not have custom_outputs attribute when not provided
    assert not hasattr(result, "custom_outputs")


def test_convert_responses_api_response_with_custom_outputs():
    llm = ChatDatabricks(model="test-model", use_responses_api=True)

    # Create a simple object with custom_outputs since Response.model_construct doesn't support it
    class MockResponse:
        def __init__(self):
            self.id = "response_123"
            self.custom_outputs = {"model_version": "v2.1", "processing_time": 150}
            self.error = None
            self.output = [
                ResponseOutputMessage.model_construct(
                    type="message",
                    content=[
                        ResponseOutputText.model_construct(
                            type="output_text", text="Hello!", id="text_123"
                        )
                    ],
                )
            ]

    response = MockResponse()
    result = llm._convert_responses_api_response_to_chat_result(response)

    assert isinstance(result, ChatResult)
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    assert hasattr(message, "custom_outputs")
    assert message.custom_outputs == {"model_version": "v2.1", "processing_time": 150}


def test_convert_chatagent_response_with_custom_outputs():
    llm = ChatDatabricks(model="test-model")

    response = SimpleNamespace(
        messages=[{"role": "assistant", "content": "Hello from ChatAgent!"}],
        custom_outputs={"agent_version": "1.0", "tokens_used": 25},
    )

    result = llm._convert_chatagent_response_to_chat_result(response)

    assert isinstance(result, ChatResult)
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    assert hasattr(message, "custom_outputs")
    assert message.custom_outputs == {"agent_version": "1.0", "tokens_used": 25}


def test_convert_responses_api_chunk_with_custom_outputs():
    chunk = ResponseTextDeltaEvent.model_construct(
        type="response.output_text.delta",
        item_id="item_123",
        delta="Hello",
        custom_outputs={"chunk_index": 0},
    )

    result = _convert_responses_api_chunk_to_lc_chunk(chunk)

    assert isinstance(result, AIMessageChunk)
    assert result.content == [{"type": "text", "text": "Hello"}]
    assert hasattr(result, "custom_outputs")
    assert result.custom_outputs == {"chunk_index": 0}


def test_invoke_with_custom_inputs_integration():
    with patch("databricks_langchain.chat_models.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock successful response with custom_outputs
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": "Response with custom outputs",
            "custom_outputs": {"confidence": 0.99},
        }
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None
        mock_response.model = "test-model"
        mock_client.chat.completions.create.return_value = mock_response

        llm = ChatDatabricks(model="test-model")
        messages = [HumanMessage(content="Test message")]
        custom_inputs = {"user_id": "test_user", "context": "unit_test"}

        # Test public invoke method with custom_inputs
        result = llm.invoke(messages, custom_inputs=custom_inputs)

        # Verify the result
        assert isinstance(result, AIMessage)
        assert result.content == "Response with custom outputs"
        assert hasattr(result, "custom_outputs")
        assert result.custom_outputs == {"confidence": 0.99}

        # Verify the API call included custom_inputs
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "extra_body" in call_args
        assert call_args["extra_body"]["custom_inputs"] == custom_inputs
