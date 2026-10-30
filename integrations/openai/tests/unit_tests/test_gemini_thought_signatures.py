"""Gemini thought signatures round-trip through AI Gateway's `thoughtSignature` tool-call field."""

import copy
import json
from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from databricks.sdk import WorkspaceClient

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI
from databricks_openai.utils.clients import _restore_gateway_messages

GEMINI = "system.ai.gemini-3-5-flash"


def _tool_call(sig: str | None, tz: str) -> dict[str, Any]:
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
            _tool_call("SIG_TOKYO", "Asia/Tokyo"),
            _tool_call("SIG_PARIS", "Europe/Paris"),
        ],
    },
    "tool_calls",
)
# The answer after a tool result comes back as text parts that carry their own signature.
ANSWER_TURN = _completion(
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "03:12 in Tokyo.", "thoughtSignature": "SIG_TEXT"}],
    },
    "stop",
)


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    body = {
        "id": "chunk",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": GEMINI,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body)}\n\n"


def _streamed_tool_turn(split_signature: bool) -> str:
    first = _tool_call("SIG_TOKYO", "Asia/Tokyo") | {"index": 0}
    second = _tool_call("SIG_PARIS", "Europe/Paris") | {"index": 1}
    if split_signature:
        first["thoughtSignature"] = "SIG_"
        chunks = [
            _chunk({"role": "assistant", "tool_calls": [first]}),
            _chunk({"tool_calls": [{"index": 0, "thoughtSignature": "TOKYO"}]}),
        ]
    else:
        chunks = [_chunk({"role": "assistant", "tool_calls": [first]})]
    chunks += [_chunk({"tool_calls": [second]}), _chunk({}, "tool_calls"), "data: [DONE]\n\n"]
    return "".join(chunks)


def _streamed_answer() -> str:
    parts = [{"type": "text", "text": "03:12 in Tokyo.", "thoughtSignature": "SIG_TEXT"}]
    return _chunk({"role": "assistant", "content": parts}) + _chunk({}, "stop") + "data: [DONE]\n\n"


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

    def assistant_tool_calls(self, request: int) -> list[dict[str, Any]]:
        """The first assistant message's tool calls, checking ids are the gateway's own."""
        messages = self.requests[request]["messages"]
        tool_calls = next(m["tool_calls"] for m in messages if m.get("tool_calls"))
        assert {c["id"] for c in tool_calls} == {"get_current_time"}
        assert {m["tool_call_id"] for m in messages if m["role"] == "tool"} == {"get_current_time"}
        return tool_calls


def _workspace_client() -> MagicMock:
    client = MagicMock(spec=WorkspaceClient)
    client.config.host = "https://test.databricks.com"
    client.config.authenticate.return_value = {"Authorization": "Bearer token"}
    return client


def _async_client(gateway: _Gateway) -> AsyncDatabricksOpenAI:
    transport = httpx.MockTransport(gateway)
    with patch(
        "databricks_openai.utils.clients._get_authorized_async_http_client",
        return_value=httpx.AsyncClient(transport=transport),
    ):
        return AsyncDatabricksOpenAI(workspace_client=_workspace_client(), use_ai_gateway=True)


@function_tool
def get_current_time(tz: str) -> str:
    """Current time in a timezone."""
    return f"03:12 in {tz}"


def _agent(client: AsyncDatabricksOpenAI, model: str = GEMINI) -> Agent:
    return Agent(
        name="Agent",
        model=OpenAIChatCompletionsModel(model=model, openai_client=client),
        tools=[get_current_time],
    )


@pytest.mark.asyncio
async def test_agents_sdk_sends_signatures_back_for_two_calls_with_one_id():
    gateway = _Gateway(TOOL_TURN, ANSWER_TURN)
    result = await Runner.run(_agent(_async_client(gateway)), "Time in Tokyo and Paris?")

    assert result.final_output == "03:12 in Tokyo."
    tool_calls = gateway.assistant_tool_calls(1)
    assert [c["thoughtSignature"] for c in tool_calls] == ["SIG_TOKYO", "SIG_PARIS"]
    assert all("extra_content" not in c for c in tool_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("split_signature", [False, True])
async def test_agents_sdk_streamed_signatures_are_sent_back(split_signature):
    gateway = _Gateway(_streamed_tool_turn(split_signature), _streamed_answer())
    result = Runner.run_streamed(_agent(_async_client(gateway)), "Time in Tokyo and Paris?")
    async for _ in result.stream_events():
        pass

    assert result.final_output == "03:12 in Tokyo."
    tool_calls = gateway.assistant_tool_calls(1)
    assert [c["thoughtSignature"] for c in tool_calls] == ["SIG_TOKYO", "SIG_PARIS"]


@pytest.mark.asyncio
async def test_same_tool_in_consecutive_responses_of_one_run():
    one_call = _completion(
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("SIG_1", "UTC")]},
        "tool_calls",
    )
    gateway = _Gateway(one_call, copy.deepcopy(one_call), ANSWER_TURN)
    result = await Runner.run(_agent(_async_client(gateway)), "Time, twice?")

    assert result.final_output == "03:12 in Tokyo."
    assert [c["thoughtSignature"] for c in gateway.assistant_tool_calls(2)] == ["SIG_1"]


@pytest.mark.asyncio
async def test_signatures_survive_a_follow_up_turn():
    gateway = _Gateway(TOOL_TURN, ANSWER_TURN, ANSWER_TURN)
    agent = _agent(_async_client(gateway))
    first = await Runner.run(agent, "Time in Tokyo and Paris?")
    follow_up = first.to_input_list() + [{"role": "user", "content": "Thanks!"}]
    await Runner.run(agent, cast(Any, follow_up))

    tool_calls = gateway.assistant_tool_calls(2)
    assert [c["thoughtSignature"] for c in tool_calls] == ["SIG_TOKYO", "SIG_PARIS"]


def test_sync_client_exposes_signatures_and_flattens_signed_text():
    gateway = _Gateway(TOOL_TURN, ANSWER_TURN)
    transport = httpx.MockTransport(gateway)
    with patch(
        "databricks_openai.utils.clients._get_authorized_http_client",
        return_value=httpx.Client(transport=transport),
    ):
        client = DatabricksOpenAI(workspace_client=_workspace_client(), use_ai_gateway=True)

    reply = client.chat.completions.create(model=GEMINI, messages=[])
    calls = reply.choices[0].message.tool_calls or []
    assert len({c.id for c in calls}) == 2
    assert [c.extra_content for c in calls] == [  # ty:ignore[unresolved-attribute]
        {"google": {"thought_signature": "SIG_TOKYO"}},
        {"google": {"thought_signature": "SIG_PARIS"}},
    ]
    answer = client.chat.completions.create(model=GEMINI, messages=[])
    assert answer.choices[0].message.content == "03:12 in Tokyo."


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["databricks-claude-sonnet-4-5", "databricks-gpt-5-2", GEMINI])
async def test_tool_calls_without_signature_are_unchanged(model):
    unsigned = _completion(
        {"role": "assistant", "content": None, "tool_calls": [_tool_call(None, "UTC")]},
        "tool_calls",
    )
    answer = _completion({"role": "assistant", "content": "Done."}, "stop")
    gateway = _Gateway(unsigned, answer)
    await Runner.run(_agent(_async_client(gateway), model), "Time?")

    assert gateway.assistant_tool_calls(1) == [
        {
            "id": "get_current_time",
            "type": "function",
            "function": {"name": "get_current_time", "arguments": '{"tz": "UTC"}'},
        }
    ]


def test_messages_without_google_signatures_are_untouched():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call(None, "UTC")]},
        {"role": "tool", "tool_call_id": "get_current_time", "content": "03:12"},
    ]
    expected = copy.deepcopy(messages)
    _restore_gateway_messages(messages)
    assert messages == expected
