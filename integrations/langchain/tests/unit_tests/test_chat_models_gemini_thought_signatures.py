"""Gemini thought signatures round-trip through AI Gateway's `thoughtSignature` tool-call field."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from databricks_langchain import ChatDatabricks
from databricks_langchain.chat_models import _convert_message_to_dict

GEMINI = "system.ai.gemini-3-5-flash"


def _raw_call(sig: str | None, tz: str) -> dict[str, Any]:
    # Recorded gateway shape: the id is the function name and the signature is a sibling of
    # "function" (trial 20261025-165909-4af4, /ai-gateway/mlflow/v1/chat/completions).
    call: dict[str, Any] = {
        "id": "get_current_time",
        "type": "function",
        "function": {"name": "get_current_time", "arguments": json.dumps({"tz": tz})},
    }
    if sig:
        call["thoughtSignature"] = sig
    return call


def _completion(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        "id": None,
        "object": "chat.completion",
        "created": 0,
        "model": GEMINI,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


TOOL_TURN = _completion(
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            _raw_call("SIG_TOKYO", "Asia/Tokyo"),
            _raw_call("SIG_PARIS", "Europe/Paris"),
        ],
    },
    "tool_calls",
)
ANSWER_TURN = _completion(
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "03:12 in Tokyo.", "thoughtSignature": "SIG_TEXT"}],
    },
    "stop",
)


def _sse(*deltas: dict[str, Any]) -> str:
    chunks = [
        {
            "id": "chunk",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": GEMINI,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


STREAMED_TOOL_TURN = _sse(
    {"role": "assistant", "tool_calls": [_raw_call("SIG_", "Asia/Tokyo") | {"index": 0}]},
    {"tool_calls": [{"index": 0, "thoughtSignature": "TOKYO"}]},
    {"tool_calls": [_raw_call("SIG_PARIS", "Europe/Paris") | {"index": 1}]},
)


class _Gateway:
    """Replays recorded gateway replies and records each request body."""

    def __init__(self, *replies: dict[str, Any] | str):
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            return httpx.Response(
                200, content=reply.encode(), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=reply)


@tool
def get_current_time(tz: str) -> str:
    """Current time in a timezone."""
    return f"03:12 in {tz}"


def _llm(gateway: _Gateway) -> ChatDatabricks:
    workspace_client = MagicMock(spec=WorkspaceClient)
    workspace_client.config.host = "https://test.databricks.com"
    workspace_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
    llm = ChatDatabricks(model=GEMINI, workspace_client=workspace_client, use_ai_gateway=True)
    with patch(
        "databricks_openai.utils.clients._get_authorized_http_client",
        return_value=httpx.Client(transport=httpx.MockTransport(gateway)),
    ):
        llm.client  # noqa: B018 - build the cached client with the mock transport
    return llm


def _tool_turn(ai: AIMessage) -> list:
    return [
        HumanMessage("Time in Tokyo and Paris?"),
        ai,
        *[ToolMessage("03:12", tool_call_id=call["id"]) for call in ai.tool_calls],
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_next_request_carries_signatures_and_gateway_ids(stream):
    gateway = _Gateway(STREAMED_TOOL_TURN if stream else TOOL_TURN, ANSWER_TURN)
    llm = _llm(gateway).bind_tools([get_current_time])
    question = [HumanMessage("Time in Tokyo and Paris?")]
    if stream:
        ai = None
        for chunk in llm.stream(question):
            ai = chunk if ai is None else ai + chunk
    else:
        ai = llm.invoke(question)
    assert isinstance(ai, AIMessage)
    assert len(ai.tool_calls) == 2

    answer = llm.invoke(_tool_turn(ai))

    assert answer.content == "03:12 in Tokyo."
    messages = gateway.requests[1]["messages"]
    assert [c["thoughtSignature"] for c in messages[1]["tool_calls"]] == ["SIG_TOKYO", "SIG_PARIS"]
    assert {c["id"] for c in messages[1]["tool_calls"]} == {"get_current_time"}
    assert [m["tool_call_id"] for m in messages[2:]] == ["get_current_time"] * 2


def test_repeated_ids_pair_with_raw_tool_calls_in_order():
    # As produced by databricks-openai releases that don't make Gemini's ids unique.
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_current_time", "args": {"tz": "Asia/Tokyo"}, "id": "get_current_time"},
            {"name": "get_current_time", "args": {"tz": "Europe/Paris"}, "id": "get_current_time"},
        ],
        additional_kwargs={
            "tool_calls": [_raw_call("SIG_TOKYO", "Asia/Tokyo"), _raw_call(None, "Europe/Paris")]
        },
    )
    tool_calls = _convert_message_to_dict(message)["tool_calls"]
    assert tool_calls[0]["thoughtSignature"] == "SIG_TOKYO"
    assert "thoughtSignature" not in tool_calls[1]


def test_edited_tool_call_keeps_its_signature():
    # HumanInTheLoopMiddleware's "edit" decision rewrites args and keeps additional_kwargs.
    message = AIMessage(
        content="",
        tool_calls=[{"name": "get_current_time", "args": {"tz": "UTC"}, "id": "get_current_time"}],
        additional_kwargs={"tool_calls": [_raw_call("SIG_TOKYO", "Asia/Tokyo")]},
    )
    (tool_call,) = _convert_message_to_dict(message)["tool_calls"]
    assert tool_call["thoughtSignature"] == "SIG_TOKYO"
    assert json.loads(tool_call["function"]["arguments"]) == {"tz": "UTC"}


def test_tool_calls_without_signature_are_unchanged():
    message = AIMessage(
        content="",
        tool_calls=[{"name": "get_current_time", "args": {"tz": "UTC"}, "id": "call_1"}],
        additional_kwargs={"tool_calls": [_raw_call(None, "UTC") | {"id": "call_1"}]},
    )
    assert _convert_message_to_dict(message)["tool_calls"] == [
        {
            "type": "function",
            "id": "call_1",
            "function": {"name": "get_current_time", "arguments": '{"tz": "UTC"}'},
        }
    ]
