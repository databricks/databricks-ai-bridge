"""ChatDatabricks on Claude 5.x AI Gateway responses, whose content mixes reasoning and text parts."""

import copy
import json
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from databricks_openai import DatabricksOpenAI
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from databricks_langchain import ChatDatabricks

_SIGNATURE = "CAQSqwUKDAgBEAEYAiIEdGVzdA=="
_REASONING = "The user greets me, so I greet them back."
_ANSWER = "Hello! How can I help?"

# Recorded shapes from `system.ai.claude-opus-5-5` through the AI Gateway's chat completions API.
_CLAUDE5_MESSAGE: dict[str, Any] = {
    "id": "chatcmpl_claude5",
    "object": "chat.completion",
    "created": 1761300000,
    "model": "claude-opus-5-5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": _REASONING, "signature": _SIGNATURE}
                        ],
                    },
                    {"type": "text", "text": _ANSWER},
                ],
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42},
}


def _reasoning_delta(text: str, signature: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": text, "signature": signature}],
        }
    ]


_CLAUDE5_STREAM_DELTAS: list[Any] = [
    "",
    _reasoning_delta("", ""),
    _reasoning_delta("The user greets me, ", ""),
    _reasoning_delta("so I greet them back.", ""),
    _reasoning_delta("", _SIGNATURE),
    "Hello! ",
    "How can I help?",
]

_CLAUDE5_TOOL_CALL_MESSAGE: dict[str, Any] = {
    **_CLAUDE5_MESSAGE,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": _reasoning_delta("I should look up the weather.", _SIGNATURE),
                "tool_calls": [
                    {
                        "id": "toolu_01",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}


def _stream_chunks(deltas: list[Any]) -> list[ChatCompletionChunk]:
    chunks = []
    for index, content in enumerate(deltas):
        delta: dict[str, Any] = {"content": content}
        if index == 0:
            delta["role"] = "assistant"
        last = index == len(deltas) - 1
        fields: dict[str, Any] = {
            "id": "chatcmpl_claude5",
            "object": "chat.completion.chunk",
            "created": 1761300000,
            "model": "claude-opus-5-5",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "stop" if last else None,
                }
            ],
        }
        chunks.append(ChatCompletionChunk.construct(**fields))
    return chunks


def _llm(model: str, client: Mock) -> ChatDatabricks:
    with patch("databricks_langchain.chat_models.get_openai_client", return_value=client):
        llm = ChatDatabricks(model=model)
        _ = llm.client
    return llm


def _completion_client(*responses: dict[str, Any]) -> Mock:
    client = Mock()
    client.chat.completions.create.side_effect = [
        ChatCompletion.construct(**copy.deepcopy(response)) for response in responses
    ]
    return client


def _stream_client(deltas: list[Any]) -> Mock:
    client = Mock()
    client.chat.completions.create.side_effect = lambda **_: iter(_stream_chunks(deltas))
    return client


def _blocks(message: AIMessage) -> list[dict[str, Any]]:
    assert isinstance(message.content, list)
    return [block for block in message.content if isinstance(block, dict)]


def test_claude5_invoke_returns_answer_text_and_one_reasoning_block() -> None:
    llm = _llm("system.ai.claude-opus-5-5", _completion_client(_CLAUDE5_MESSAGE))

    message = llm.invoke("hi")

    assert message.text == _ANSWER
    reasoning, text = _blocks(message)
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"][0]["text"] == _REASONING
    assert reasoning["summary"][0]["signature"] == _SIGNATURE
    assert text == {"type": "text", "text": _ANSWER}


def test_claude5_stream_merges_into_one_reasoning_block_and_one_text_block() -> None:
    llm = _llm("system.ai.claude-opus-5-5", _stream_client(_CLAUDE5_STREAM_DELTAS))

    chunks = list(llm.stream("hi"))

    assert "".join(chunk.text for chunk in chunks) == _ANSWER
    assert not any("signature" in chunk.text for chunk in chunks)
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    assert isinstance(merged, AIMessageChunk)
    assert merged.text == _ANSWER
    reasoning, text = _blocks(merged)
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"][0]["text"] == _REASONING
    assert reasoning["summary"][0]["signature"] == _SIGNATURE
    assert reasoning["summary"][0]["type"] == "summary_text"
    assert text["type"] == "text"
    assert text["text"] == _ANSWER


@pytest.mark.asyncio
async def test_claude5_astream_merges_into_one_reasoning_block_and_one_text_block() -> None:
    async def _astream(**_: Any):
        for chunk in _stream_chunks(_CLAUDE5_STREAM_DELTAS):
            yield chunk

    client = Mock()
    client.chat.completions.create = AsyncMock(side_effect=lambda **_: _astream())
    with patch("databricks_langchain.chat_models.get_async_openai_client", return_value=client):
        llm = ChatDatabricks(model="system.ai.claude-opus-5-5")
        _ = llm.async_client
        chunks = [chunk async for chunk in llm.astream("hi")]

    assert "".join(chunk.text for chunk in chunks) == _ANSWER
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    reasoning, text = _blocks(merged)
    assert reasoning["summary"][0]["signature"] == _SIGNATURE
    assert text["text"] == _ANSWER


def test_claude5_stream_starts_a_new_reasoning_block_after_each_signature() -> None:
    deltas = [
        _reasoning_delta("first", ""),
        _reasoning_delta("", "sig-1"),
        _reasoning_delta("second", ""),
        _reasoning_delta("", "sig-2"),
        "answer",
    ]
    llm = _llm("system.ai.claude-sonnet-5-5", _stream_client(deltas))

    chunks = list(llm.stream("hi"))
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk

    first, second, text = _blocks(merged)
    assert first["summary"][0]["text"] == "first"
    assert first["summary"][0]["signature"] == "sig-1"
    assert second["summary"][0]["text"] == "second"
    assert second["summary"][0]["signature"] == "sig-2"
    assert text["text"] == "answer"


def test_claude5_reasoning_signature_round_trips_on_tool_call_turn() -> None:
    client = _completion_client(_CLAUDE5_TOOL_CALL_MESSAGE, _CLAUDE5_MESSAGE)
    llm = _llm("system.ai.claude-opus-5-5", client)

    tool_turn = llm.invoke("weather in Paris?")
    assert tool_turn.tool_calls[0]["name"] == "get_weather"
    llm.invoke(
        [
            HumanMessage("weather in Paris?"),
            tool_turn,
            ToolMessage("sunny", tool_call_id="toolu_01"),
        ]
    )

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assistant = sent[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == _reasoning_delta("I should look up the weather.", _SIGNATURE)
    assert assistant["tool_calls"][0]["id"] == "toolu_01"


def test_claude5_streamed_turn_replays_gateway_blocks_without_merge_indexes() -> None:
    llm = _llm("system.ai.claude-opus-5-5", _stream_client(_CLAUDE5_STREAM_DELTAS))
    chunks = list(llm.stream("hi"))
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk

    data = llm._prepare_inputs([HumanMessage("hi"), merged, HumanMessage("and again?")])

    assert data["messages"][1]["content"] == [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": _REASONING, "signature": _SIGNATURE}],
        },
        {"type": "text", "text": _ANSWER},
    ]


class _Gateway:
    """Serves recorded AI Gateway responses to a real ``DatabricksOpenAI`` client and keeps requests."""

    def __init__(self, *responses: dict[str, Any] | list[Any]):
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        response = self._responses.pop(0)
        if isinstance(response, dict):
            return httpx.Response(200, json=response)
        events = "".join(
            f"data: {json.dumps(chunk.model_dump(exclude_unset=True))}\n\n"
            for chunk in _stream_chunks(response)
        )
        return httpx.Response(
            200,
            content=events + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )


def _gateway_llm(gateway: _Gateway) -> ChatDatabricks:
    workspace_client = Mock()
    workspace_client.config.host = "https://example.cloud.databricks.com"
    workspace_client.config.authenticate.return_value = {"Authorization": "Bearer t"}
    with patch(
        "databricks_openai.utils.clients._get_authorized_http_client",
        return_value=httpx.Client(transport=httpx.MockTransport(gateway)),
    ):
        client = DatabricksOpenAI(workspace_client=workspace_client, use_ai_gateway=True)
    return _llm("system.ai.claude-opus-5-5", cast(Mock, client))


def test_claude5_invoke_through_databricks_openai_keeps_signed_reasoning() -> None:
    gateway = _Gateway(_CLAUDE5_TOOL_CALL_MESSAGE, _CLAUDE5_MESSAGE)
    llm = _gateway_llm(gateway)

    tool_turn = llm.invoke("weather in Paris?")
    reasoning = _blocks(tool_turn)[0]
    assert reasoning["summary"][0]["signature"] == _SIGNATURE
    answer = llm.invoke(
        [
            HumanMessage("weather in Paris?"),
            tool_turn,
            ToolMessage("sunny", tool_call_id="toolu_01"),
        ]
    )

    assert answer.text == _ANSWER
    assert _blocks(answer)[0]["summary"][0]["signature"] == _SIGNATURE
    assert gateway.requests[1]["messages"][1]["content"] == _reasoning_delta(
        "I should look up the weather.", _SIGNATURE
    )


def test_claude5_stream_through_databricks_openai_keeps_signed_reasoning() -> None:
    gateway = _Gateway(_CLAUDE5_STREAM_DELTAS, _CLAUDE5_MESSAGE)
    llm = _gateway_llm(gateway)

    chunks = list(llm.stream("hi"))
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    assert merged.text == _ANSWER
    reasoning, text = _blocks(cast(AIMessage, merged))
    assert reasoning["summary"][0]["text"] == _REASONING
    assert reasoning["summary"][0]["signature"] == _SIGNATURE
    assert text["text"] == _ANSWER

    llm.invoke([HumanMessage("hi"), merged, HumanMessage("and again?")])
    assert gateway.requests[1]["messages"][1]["content"] == [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": _REASONING, "signature": _SIGNATURE}],
        },
        {"type": "text", "text": _ANSWER},
    ]


@pytest.mark.parametrize("model", ["system.ai.claude-sonnet-4-5", "databricks-gpt-5-2"])
def test_string_content_models_are_unchanged(model: str) -> None:
    string_message = copy.deepcopy(_CLAUDE5_MESSAGE)
    string_message["choices"][0]["message"]["content"] = _ANSWER
    llm = _llm(model, _completion_client(string_message))
    assert llm.invoke("hi").content == _ANSWER

    llm = _llm(model, _stream_client(["", "Hello! ", "How can I help?"]))
    chunks = list(llm.stream("hi"))
    assert all(isinstance(chunk.content, str) for chunk in chunks)
    assert "".join(cast(str, chunk.content) for chunk in chunks) == _ANSWER

    history: list[BaseMessage] = [HumanMessage("hi"), AIMessage(_ANSWER), HumanMessage("again")]
    assert llm._prepare_inputs(history)["messages"][1] == {
        "role": "assistant",
        "content": _ANSWER,
    }


def test_non_claude_list_content_is_still_json_encoded() -> None:
    llm = _llm("databricks-gpt-oss-120b", _completion_client(_CLAUDE5_MESSAGE))
    content = llm.invoke("hi").content
    assert isinstance(content, str)
    assert content.startswith('[{"type": "reasoning"')
