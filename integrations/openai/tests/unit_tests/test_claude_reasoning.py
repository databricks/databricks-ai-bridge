"""Claude 5.x reasoning content through the AI Gateway's chat completions API."""

import copy
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.items import ReasoningItem
from databricks.sdk import WorkspaceClient
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import ResponseTextDeltaEvent

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI
from databricks_openai.agents import DatabricksChatCompletionsModel

_MODEL = "system.ai.claude-opus-5-5"
_SIGNATURE = "CAQSqwUKDAgBEAEYAiIEdGVzdA=="
_REASONING = "The user greets me, so I greet them back."
_ANSWER = "Hello! How can I help?"


def _reasoning(text: str, signature: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": text, "signature": signature}],
        }
    ]


# Recorded shapes from `system.ai.claude-opus-5-5` through the AI Gateway.
_MESSAGE: dict[str, Any] = {
    "id": "chatcmpl_claude5",
    "object": "chat.completion",
    "created": 1761300000,
    "model": "claude-opus-5-5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [*_reasoning(_REASONING, _SIGNATURE), {"type": "text", "text": _ANSWER}],
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42},
}

_ANSWER_DELTAS: list[dict[str, Any]] = [
    {"role": "assistant", "content": ""},
    {"content": _reasoning("", "")},
    {"content": _reasoning("The user greets me, ", "")},
    {"content": _reasoning("so I greet them back.", "")},
    {"content": _reasoning("", _SIGNATURE)},
    {"content": "Hello! "},
    {"content": "How can I help?"},
]

_TOOL_CALL_DELTAS: list[dict[str, Any]] = [
    {"role": "assistant", "content": ""},
    {"content": _reasoning("I should look up the weather.", "")},
    {"content": _reasoning("", "sig-tool")},
    {
        "tool_calls": [
            {
                "index": 0,
                "id": "toolu_01",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ]
    },
]


def _chunks(deltas: list[dict[str, Any]], finish_reason: str = "stop") -> list[ChatCompletionChunk]:
    return [
        ChatCompletionChunk.construct(
            id="chatcmpl_claude5",
            object="chat.completion.chunk",
            created=1761300000,
            model="claude-opus-5-5",
            choices=[
                {
                    "index": 0,
                    "delta": copy.deepcopy(delta),
                    "finish_reason": finish_reason if index == len(deltas) - 1 else None,
                }
            ],
        )
        for index, delta in enumerate(deltas)
    ]


async def _astream(chunks: list[ChatCompletionChunk]):
    for chunk in chunks:
        yield chunk


def _workspace_client() -> MagicMock:
    client = MagicMock(spec=WorkspaceClient)
    client.config.host = "https://test.databricks.com"
    client.config.authenticate.return_value = {"Authorization": "Bearer token"}
    return client


@pytest.fixture
def async_client() -> AsyncDatabricksOpenAI:
    return AsyncDatabricksOpenAI(workspace_client=_workspace_client(), use_ai_gateway=True)


@pytest.mark.asyncio
async def test_non_stream_reply_is_text_with_thinking_blocks(async_client):
    response = ChatCompletion.construct(**copy.deepcopy(_MESSAGE))
    with patch.object(AsyncCompletions, "create", new_callable=AsyncMock, return_value=response):
        result = await async_client.chat.completions.create(
            model=_MODEL, messages=[{"role": "user", "content": "hi"}]
        )

    message = result.choices[0].message
    assert message.content == _ANSWER
    assert cast(Any, message).thinking_blocks == [
        {"type": "thinking", "thinking": _REASONING, "signature": _SIGNATURE}
    ]


@pytest.mark.asyncio
async def test_stream_deltas_carry_string_content_and_thinking_blocks(async_client):
    with patch.object(
        AsyncCompletions,
        "create",
        new_callable=AsyncMock,
        return_value=_astream(_chunks(_ANSWER_DELTAS)),
    ):
        stream = await async_client.chat.completions.create(
            model=_MODEL, messages=[{"role": "user", "content": "hi"}], stream=True
        )
        deltas = [chunk.choices[0].delta async for chunk in stream]

    assert all(delta.content is None or isinstance(delta.content, str) for delta in deltas)
    assert "".join(delta.content or "" for delta in deltas) == _ANSWER
    signatures = [
        block.get("signature")
        for delta in deltas
        for block in getattr(delta, "thinking_blocks", None) or []
    ]
    assert _SIGNATURE in signatures


def test_sync_client_splits_reasoning():
    client = DatabricksOpenAI(workspace_client=_workspace_client(), use_ai_gateway=True)
    response = ChatCompletion.construct(**copy.deepcopy(_MESSAGE))
    with patch.object(Completions, "create", return_value=response):
        result = client.chat.completions.create(
            model=_MODEL, messages=[{"role": "user", "content": "hi"}]
        )
    assert result.choices[0].message.content == _ANSWER


@pytest.mark.asyncio
async def test_assistant_thinking_blocks_are_sent_as_gateway_reasoning(async_client):
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": [{"type": "thinking", "thinking": "look", "signature": "sig-1"}],
            "tool_calls": [{"id": "toolu_01"}],
        },
        {"role": "tool", "content": "sunny", "tool_call_id": "toolu_01"},
        {
            "role": "assistant",
            "content": "It is sunny.",
            "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "sig-2"}],
        },
    ]
    with patch.object(AsyncCompletions, "create", new_callable=AsyncMock) as create:
        await async_client.chat.completions.create(model=_MODEL, messages=cast(Any, messages))

    sent = create.call_args.kwargs["messages"]
    assert sent[1]["content"] == _reasoning("look", "sig-1")
    assert "thinking_blocks" not in sent[1]
    assert sent[3]["content"] == [
        *_reasoning("", "sig-2"),
        {"type": "text", "text": "It is sunny."},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["system.ai.claude-sonnet-4-5", "databricks-gpt-oss-120b"])
async def test_other_models_are_unchanged(async_client, model):
    response = ChatCompletion.construct(**copy.deepcopy(_MESSAGE))
    if "claude" in model:
        response.choices[0].message.content = _ANSWER
    with patch.object(AsyncCompletions, "create", new_callable=AsyncMock, return_value=response):
        result = await async_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}]
        )
    assert result is response
    expected = _ANSWER if "claude" in model else _MESSAGE["choices"][0]["message"]["content"]
    assert result.choices[0].message.content == expected
    assert getattr(result.choices[0].message, "thinking_blocks", None) is None


async def _drain(result: Any) -> None:
    async for _ in result.stream_events():
        pass


def _user(text: str) -> Any:
    return {"role": "user", "content": text}


@function_tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"Sunny in {city}."


@pytest.mark.asyncio
async def test_agent_streams_reply_and_replays_signature_after_tool_call(async_client):
    set_tracing_disabled(True)
    agent = Agent(
        name="Agent",
        model=DatabricksChatCompletionsModel(model=_MODEL, openai_client=async_client),
        tools=[get_weather],
    )
    responses = [
        _astream(_chunks(_TOOL_CALL_DELTAS, finish_reason="tool_calls")),
        _astream(_chunks(_ANSWER_DELTAS)),
        _astream(_chunks(_ANSWER_DELTAS)),
    ]
    with patch.object(
        AsyncCompletions, "create", new_callable=AsyncMock, side_effect=responses
    ) as create:
        result = Runner.run_streamed(agent, "Weather in Paris?")
        text_deltas = [
            event.data.delta
            async for event in result.stream_events()
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent)
        ]
        assert result.final_output == _ANSWER
        assert "".join(text_deltas) == _ANSWER
        reasoning = [item for item in result.new_items if isinstance(item, ReasoningItem)]
        assert len(reasoning) == 2

        tool_turn = create.call_args_list[1].kwargs["messages"]
        assistant = next(m for m in tool_turn if m["role"] == "assistant")
        assert assistant["tool_calls"][0]["id"] == "toolu_01"
        assert assistant["content"][0]["type"] == "reasoning"
        assert assistant["content"][0]["summary"][0]["signature"] == "sig-tool"

        # The next turn replays both signed reasoning blocks from history.
        await _drain(Runner.run_streamed(agent, [*result.to_input_list(), _user("Thanks")]))
        next_turn = create.call_args_list[2].kwargs["messages"]
        signatures = [
            part["summary"][0]["signature"]
            for message in next_turn
            if message["role"] == "assistant" and isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "reasoning"
        ]
        assert signatures == ["sig-tool", _SIGNATURE]


@pytest.mark.asyncio
async def test_default_model_does_not_replay_thinking_blocks(async_client):
    """Without DatabricksChatCompletionsModel, the SDK's own behavior is unchanged."""
    from agents import OpenAIChatCompletionsModel

    set_tracing_disabled(True)
    agent = Agent(
        name="Agent", model=OpenAIChatCompletionsModel(model=_MODEL, openai_client=async_client)
    )
    responses = [_astream(_chunks(_ANSWER_DELTAS)), _astream(_chunks(_ANSWER_DELTAS))]
    with patch.object(
        AsyncCompletions, "create", new_callable=AsyncMock, side_effect=responses
    ) as create:
        first = Runner.run_streamed(agent, "hi")
        await _drain(first)
        await _drain(Runner.run_streamed(agent, [*first.to_input_list(), _user("again")]))

    replayed = create.call_args_list[1].kwargs["messages"]
    assistant = next(m for m in replayed if m["role"] == "assistant")
    assert assistant["content"] == _ANSWER
