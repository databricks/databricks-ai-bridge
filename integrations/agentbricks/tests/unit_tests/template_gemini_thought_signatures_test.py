"""Stock templates on Gemini: tool-call thought signatures survive tools, turns, and approvals.

Replays recorded AI Gateway replies (trial 20261025-165909-4af4, gemini-3-5-flash): the signature is a
`thoughtSignature` field beside each tool call's `function`, the tool-call id is the function name,
and the answer after a tool result is a list of signed text parts. Without the signature on the next
request the gateway returns 400 "Function call is missing a thought_signature".
"""

import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from template_history_fixtures import SessionStoreTransport

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"
GEMINI = "system.ai.gemini-3-5-flash"


def _call(name: str, args: dict[str, Any], sig: str | None) -> dict[str, Any]:
    call: dict[str, Any] = {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }
    if sig:
        call["thoughtSignature"] = sig
    return call


def _reply(*tool_calls: dict[str, Any], text: str | None = None) -> dict[str, Any]:
    if tool_calls:
        message = {"role": "assistant", "content": None, "tool_calls": list(tool_calls)}
    else:
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": text, "thoughtSignature": "SIG_TEXT"}],
        }
    return {
        "id": None,
        "object": "chat.completion",
        "created": 0,
        "model": GEMINI,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }


# Gemini signs the first call of a parallel set; both carry one here to check pairing by order.
TWO_TIME_CALLS = _reply(
    _call("get_current_time", {}, "SIG_TIME_1"), _call("get_current_time", {}, "SIG_TIME_2")
)
SEND_CALL = _reply(_call("send_message", {"recipient": "bob", "body": "hi"}, "SIG_SEND"))


class Gateway:
    """AI Gateway stand-in: replays replies, records requests, and enforces signatures."""

    def __init__(self):
        self.replies: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.signatures: dict[str, list[str | None]] = {}

    def queue(self, *replies: dict[str, Any]) -> None:
        self.replies.extend(replies)
        for reply in replies:
            for call in reply["choices"][0]["message"].get("tool_calls") or []:
                self.signatures.setdefault(call["function"]["name"], []).append(
                    call.get("thoughtSignature")
                )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        sent: dict[str, list[str | None]] = {}
        for message in body["messages"]:
            for call in message.get("tool_calls") or []:
                assert call["id"] == call["function"]["name"], "gateway ids must round-trip"
                sent.setdefault(call["function"]["name"], []).append(call.get("thoughtSignature"))
        for name, signatures in sent.items():
            if signatures != self.signatures[name][: len(signatures)]:
                return httpx.Response(
                    400,
                    json={"error": {"message": "Function call is missing a thought_signature"}},
                )
        reply = self.replies.pop(0)
        if not body.get("stream"):
            return httpx.Response(200, json=reply)
        return httpx.Response(
            200, content=_as_stream(reply).encode(), headers={"content-type": "text/event-stream"}
        )


def _as_stream(reply: dict[str, Any]) -> str:
    """The same reply as chat-completion chunks: one delta per tool call, then the finish."""
    choice = reply["choices"][0]
    message = choice["message"]
    if message.get("tool_calls"):
        deltas = [
            {"role": "assistant", "tool_calls": [call | {"index": i}]}
            for i, call in enumerate(message["tool_calls"])
        ]
    else:
        deltas = [{"role": "assistant", "content": message["content"]}]
    chunks = [(delta, None) for delta in deltas] + [({}, choice["finish_reason"])]
    return (
        "".join(
            "data: "
            + json.dumps(
                {
                    "id": "chunk",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": GEMINI,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )
            + "\n\n"
            for delta, finish in chunks
        )
        + "data: [DONE]\n\n"
    )


def _requires_gemini_support(*modules: str) -> None:
    """Skip until the installed client libraries carry the Gemini fix (released separately)."""
    openai_clients = importlib.import_module("databricks_openai.utils.clients")
    if not hasattr(openai_clients, "_restore_gateway_messages"):
        pytest.skip("Requires databricks-openai with Gemini thought-signature support")
    if "databricks_langchain" in modules:
        chat_models = importlib.import_module("databricks_langchain.chat_models")
        if not hasattr(chat_models, "_restore_thought_signatures"):
            pytest.skip("Requires databricks-langchain with Gemini thought-signature support")


def _import_template(monkeypatch, name: str):
    monkeypatch.syspath_prepend(str(TEMPLATES / name))
    for module in list(sys.modules):
        if module in {"agent", "runtime"} or module.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, module)
    return importlib.import_module("agent.agent")


def _drop_template_modules() -> None:
    for module in list(sys.modules):
        if module in {"agent", "runtime"} or module.startswith(("agent.", "runtime.")):
            sys.modules.pop(module, None)


def _workspace_client() -> MagicMock:
    from databricks.sdk import WorkspaceClient

    client = MagicMock(spec=WorkspaceClient)
    client.config.host = "https://test.databricks.com"
    client.config.authenticate.return_value = {"Authorization": "Bearer token"}
    return client


def _route_gateway(monkeypatch, gateway: Gateway) -> None:
    transport = httpx.MockTransport(gateway)
    monkeypatch.setattr(
        "databricks_openai.utils.clients._get_authorized_http_client",
        lambda *_args, **_kwargs: httpx.Client(transport=transport),
    )
    monkeypatch.setattr(
        "databricks_openai.utils.clients._get_authorized_async_http_client",
        lambda *_args, **_kwargs: httpx.AsyncClient(transport=transport),
    )


# ---------------------------------------------------------------------------------------------
# OpenAI Agents SDK template
# ---------------------------------------------------------------------------------------------


@pytest.fixture(params=["memory", "managed"])
def openai_template(request, monkeypatch):
    for dependency in ("agents", "databricks_openai"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[openai]")
    _requires_gemini_support("databricks_openai")
    from agents.models import _openai_shared
    from databricks_openai import AsyncDatabricksOpenAI

    from databricks_agentkit.openai import sessions

    agent = _import_template(monkeypatch, "agent-openai")
    gateway = Gateway()
    _route_gateway(monkeypatch, gateway)
    monkeypatch.setattr(
        _openai_shared,
        "_default_openai_client",
        AsyncDatabricksOpenAI(workspace_client=_workspace_client(), use_ai_gateway=True),
    )
    monkeypatch.setattr(_openai_shared, "_use_responses_by_default", False)
    store = "test" if request.param == "managed" else None
    monkeypatch.setattr(
        "databricks_agentkit.runtime.tool_manifest.resolve_session_store", lambda *_args: store
    )
    monkeypatch.setattr(sessions, "_local_sessions", {})
    transport = SessionStoreTransport()
    monkeypatch.setattr(sessions, "SessionStoreClient", lambda *_args: transport)

    async def no_servers(*_args, **_kwargs):
        return []

    monkeypatch.setattr(agent, "mcp_servers", no_servers)
    monkeypatch.setattr(agent, "start_trace", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(agent, "memory_tools", lambda *_args: [])
    monkeypatch.setattr(agent, "genie_tools", lambda **_kwargs: [])
    yield agent, gateway
    for session in sessions._local_sessions.values():
        session.close()
    _drop_template_modules()


async def _openai_turn(agent, session_id: str, prompt: str | None, **kwargs) -> Any:
    agent_input = [{"role": "user", "content": prompt}] if prompt else []
    async with agent.run_agent(
        agent_input, session_id=session_id, model=GEMINI, **kwargs
    ) as result:
        async for _ in result.stream_events():
            pass
    return result


@pytest.mark.asyncio
async def test_openai_template_tool_turn_and_follow_up(openai_template):
    agent, gateway = openai_template
    gateway.queue(TWO_TIME_CALLS, _reply(text="It is 03:12."), _reply(text="You're welcome."))
    session = str(uuid4())

    first = await _openai_turn(agent, session, "What time is it, twice?")
    second = await _openai_turn(agent, session, "Thanks!")

    assert first.final_output == "It is 03:12."
    assert second.final_output == "You're welcome."
    for request in gateway.requests[1:]:
        calls = next(m["tool_calls"] for m in request["messages"] if m.get("tool_calls"))
        assert [c["thoughtSignature"] for c in calls] == ["SIG_TIME_1", "SIG_TIME_2"]


@pytest.mark.asyncio
async def test_openai_template_resumes_approval_from_stored_checkpoint(openai_template):
    agent, gateway = openai_template
    gateway.queue(SEND_CALL, _reply(text="Sent."))
    session, saved = str(uuid4()), {}

    async def save_state(value):
        saved["value"] = json.loads(json.dumps(value))

    async def load_state():
        return saved.get("value")

    paused = await _openai_turn(
        agent, session, "Message bob 'hi'", load_state=load_state, save_state=save_state
    )
    (pending,) = paused.interruptions
    resumed = await _openai_turn(
        agent,
        session,
        None,
        resume={"decisions": [{"type": "approve", "call_id": pending.call_id}]},
        load_state=load_state,
        save_state=save_state,
    )

    assert resumed.final_output == "Sent."
    calls = next(m["tool_calls"] for m in gateway.requests[-1]["messages"] if m.get("tool_calls"))
    assert [c["thoughtSignature"] for c in calls] == ["SIG_SEND"]


# ---------------------------------------------------------------------------------------------
# LangGraph template
# ---------------------------------------------------------------------------------------------


@pytest.fixture(params=["memory", "managed"])
def langgraph_template(request, monkeypatch):
    for dependency in ("databricks_langchain", "langgraph", "langchain", "langchain_mcp_adapters"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[langgraph]")
    _requires_gemini_support("databricks_openai", "databricks_langchain")
    from langgraph.checkpoint.memory import InMemorySaver

    from databricks_agentkit.langgraph import session_store

    agent = _import_template(monkeypatch, "agent-langgraph")
    gateway = Gateway()
    _route_gateway(monkeypatch, gateway)
    transport = SessionStoreTransport()
    saver = (
        InMemorySaver()
        if request.param == "memory"
        else session_store.DatabricksSessionStoreSaver("test", client=transport)
    )

    def checkpointer():
        # A managed store resumes from a fresh saver that reads the stored checkpoint back.
        if request.param == "memory":
            return saver
        return session_store.DatabricksSessionStoreSaver("test", client=transport)

    async def no_tools(*_args, **_kwargs):
        return []

    monkeypatch.setattr(agent, "checkpointer", checkpointer)
    monkeypatch.setattr(agent, "mcp_tools", no_tools)
    monkeypatch.setattr(agent, "memory_tools", lambda *_args: [])
    monkeypatch.setattr(agent, "genie_tools", lambda **_kwargs: [])
    monkeypatch.setattr(agent, "start_trace", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(agent, "_app_workspace_client", _workspace_client)
    yield agent, gateway
    _drop_template_modules()


async def _langgraph_turn(agent, session_id: str, agent_input: Any) -> list[Any]:
    return [
        event
        async for event in agent.run_agent(
            agent_input, session_id=session_id, model=GEMINI, invocation_id=str(uuid4())
        )
    ]


def _last_ai_text(events: list[Any]) -> str:
    updates = [event[1] for event in events if event[0] == "updates"]
    messages = [
        m
        for update in updates
        for node in update.values()
        if node
        for m in node.get("messages", [])
    ]
    return messages[-1].text


@pytest.mark.asyncio
async def test_langgraph_template_tool_turn_and_follow_up(langgraph_template):
    agent, gateway = langgraph_template
    gateway.queue(TWO_TIME_CALLS, _reply(text="It is 03:12."), _reply(text="You're welcome."))
    session = str(uuid4())

    first = await _langgraph_turn(
        agent, session, {"messages": [{"role": "user", "content": "What time is it, twice?"}]}
    )
    second = await _langgraph_turn(
        agent, session, {"messages": [{"role": "user", "content": "Thanks!"}]}
    )

    assert _last_ai_text(first) == "It is 03:12."
    assert _last_ai_text(second) == "You're welcome."
    for request in gateway.requests[1:]:
        calls = next(m["tool_calls"] for m in request["messages"] if m.get("tool_calls"))
        assert [c["thoughtSignature"] for c in calls] == ["SIG_TIME_1", "SIG_TIME_2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision",
    [
        {"type": "approve"},
        {
            "type": "edit",
            "edited_action": {"name": "send_message", "args": {"recipient": "bob", "body": "hey"}},
        },
    ],
)
async def test_langgraph_template_resumes_approval_from_checkpoint(langgraph_template, decision):
    from langgraph.types import Command

    agent, gateway = langgraph_template
    gateway.queue(SEND_CALL, _reply(text="Sent."))
    session = str(uuid4())

    paused = await _langgraph_turn(
        agent, session, {"messages": [{"role": "user", "content": "Message bob 'hi'"}]}
    )
    assert any(event[0] == "updates" and "__interrupt__" in event[1] for event in paused)
    resumed = await _langgraph_turn(agent, session, Command(resume={"decisions": [decision]}))

    assert _last_ai_text(resumed) == "Sent."
    calls = next(m["tool_calls"] for m in gateway.requests[-1]["messages"] if m.get("tool_calls"))
    assert [c["thoughtSignature"] for c in calls] == ["SIG_SEND"]
