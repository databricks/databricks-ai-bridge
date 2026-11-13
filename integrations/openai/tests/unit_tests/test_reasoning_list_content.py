"""Reasoning list content from non-Claude models (Qwen, Gemini, gpt-oss) through the AI Gateway."""

import copy
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RawResponsesStreamEvent,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from agents.items import ReasoningItem
from databricks.sdk import WorkspaceClient
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import ResponseTextDeltaEvent

from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import DatabricksChatCompletionsModel

_QWEN = "system.ai.qwen35-122b-a10b"
_GEMINI = "system.ai.gemini-3-pro-image"
_REASONING = "The user says hi."
_ANSWER = "Hello! How can I help?"


def _unsigned(text: str) -> list[dict[str, Any]]:
    """Qwen / gpt-oss reasoning: summary text without a signature."""
    return [{"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}]


def _message(content: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl_qwen",
        "object": "chat.completion",
        "created": 1761400000,
        "model": "qwen35-122b-a10b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


# The shape `system.ai.qwen35-122b-a10b` streams on the OpenAI template (CAJ trial
# 20261025-165909-4af4): reasoning list deltas, then the answer as strings.
_QWEN_DELTAS: list[dict[str, Any]] = [
    {"role": "assistant", "content": ""},
    {"content": _unsigned("The")},
    {"content": _unsigned(" user says hi.")},
    {"content": "Hello! "},
    {"content": "How can I help?"},
]
# Answer text can also arrive as list parts, e.g. Gemini's text parts with a thought signature.
_GEMINI_DELTAS: list[dict[str, Any]] = [
    {"role": "assistant", "content": ""},
    {"content": _unsigned(_REASONING)},
    {"content": [{"type": "text", "text": "Hello! ", "thoughtSignature": "Z2VtaW5p"}]},
    {"content": [{"type": "text", "text": "How can I help?"}]},
]
_TOOL_CALL_DELTAS: list[dict[str, Any]] = [
    {"role": "assistant", "content": ""},
    {"content": _unsigned("I should look up the weather.")},
    {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_01",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ]
    },
]


def _chunks(deltas: list[dict[str, Any]], finish_reason: str = "stop") -> list[ChatCompletionChunk]:
    return [
        ChatCompletionChunk.construct(
            id="chatcmpl_qwen",
            object="chat.completion.chunk",
            created=1761400000,
            model="qwen35-122b-a10b",
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


@pytest.fixture
def async_client() -> AsyncDatabricksOpenAI:
    workspace_client = MagicMock(spec=WorkspaceClient)
    workspace_client.config.host = "https://test.databricks.com"
    workspace_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
    return AsyncDatabricksOpenAI(workspace_client=workspace_client, use_ai_gateway=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [_QWEN, _GEMINI, "databricks-gpt-oss-120b"])
async def test_non_stream_reply_is_text_with_unsigned_thinking_blocks(async_client, model):
    content = [*_unsigned(_REASONING), {"type": "text", "text": _ANSWER}]
    response = ChatCompletion.construct(**_message(content))
    with patch.object(AsyncCompletions, "create", new_callable=AsyncMock, return_value=response):
        result = await async_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}]
        )

    message = result.choices[0].message
    assert message.content == _ANSWER
    assert cast(Any, message).thinking_blocks == [{"type": "thinking", "thinking": _REASONING}]


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "deltas"), [(_QWEN, _QWEN_DELTAS), (_GEMINI, _GEMINI_DELTAS)])
async def test_stream_deltas_carry_string_content(async_client, model, deltas):
    with patch.object(
        AsyncCompletions, "create", new_callable=AsyncMock, return_value=_astream(_chunks(deltas))
    ):
        stream = await async_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}], stream=True
        )
        deltas = [chunk.choices[0].delta async for chunk in stream]

    assert all(delta.content is None or isinstance(delta.content, str) for delta in deltas)
    assert "".join(delta.content or "" for delta in deltas) == _ANSWER
    thinking = "".join(
        block["thinking"]
        for delta in deltas
        for block in getattr(delta, "thinking_blocks", None) or []
    )
    assert thinking == _REASONING


@pytest.mark.asyncio
async def test_other_list_content_is_unchanged(async_client):
    image = [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]
    response = ChatCompletion.construct(**_message(copy.deepcopy(image)))
    with patch.object(AsyncCompletions, "create", new_callable=AsyncMock, return_value=response):
        result = await async_client.chat.completions.create(
            model=_GEMINI, messages=[{"role": "user", "content": "draw"}]
        )
    assert result.choices[0].message.content == image


@pytest.mark.asyncio
async def test_only_signed_thinking_blocks_are_sent_back(async_client):
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": _ANSWER,
            "thinking_blocks": [{"type": "thinking", "thinking": _REASONING}],
        },
        {
            "role": "assistant",
            "content": "Signed.",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "unsigned"},
                {"type": "thinking", "thinking": "signed", "signature": "sig-1"},
            ],
        },
    ]
    with patch.object(AsyncCompletions, "create", new_callable=AsyncMock) as create:
        await async_client.chat.completions.create(model=_QWEN, messages=cast(Any, messages))

    sent = create.call_args.kwargs["messages"]
    assert sent[1] == {"role": "assistant", "content": _ANSWER}
    assert sent[2]["content"] == [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "signed", "signature": "sig-1"}],
        },
        {"type": "text", "text": "Signed."},
    ]


@function_tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"Sunny in {city}."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_class", [DatabricksChatCompletionsModel, OpenAIChatCompletionsModel]
)
async def test_agent_streams_qwen_reply_without_sending_reasoning_back(async_client, model_class):
    set_tracing_disabled(True)
    agent = Agent(
        name="Agent",
        model=model_class(model=_QWEN, openai_client=async_client),
        tools=[get_weather],
    )
    responses = [
        _astream(_chunks(_TOOL_CALL_DELTAS, finish_reason="tool_calls")),
        _astream(_chunks(_QWEN_DELTAS)),
    ]
    with patch.object(
        AsyncCompletions, "create", new_callable=AsyncMock, side_effect=responses
    ) as create:
        result = Runner.run_streamed(agent, "Weather in Paris?")
        text_deltas = [
            event.data.delta
            async for event in result.stream_events()
            if isinstance(event, RawResponsesStreamEvent)
            and isinstance(event.data, ResponseTextDeltaEvent)
        ]

    assert result.final_output == _ANSWER
    assert "".join(text_deltas) == _ANSWER
    assert [item for item in result.new_items if isinstance(item, ReasoningItem)]
    tool_turn = create.call_args_list[1].kwargs["messages"]
    assistant = next(m for m in tool_turn if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "call_01"
    assert not isinstance(assistant.get("content"), list)
    assert "thinking_blocks" not in assistant
