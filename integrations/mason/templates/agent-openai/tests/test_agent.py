"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, event serialization) — no
Databricks auth needed, so they run anywhere. The live test builds the full agent and calls the
model; it is skipped unless a workspace profile is configured.
"""

import os

import pytest
from agents import FunctionTool
from agent.agent import _apply_decisions, _normalize_item, _serialize_events, _session_id
from agent.tools import all_tools


def test_tools_autoregister():
    tools = all_tools()
    assert tools, "expected the sample tool to auto-register"
    assert all(isinstance(t, FunctionTool) for t in tools)
    assert {"get_current_time", "send_message"} <= {t.name for t in tools}


def test_gated_tool_needs_approval():
    # The gated demo tool must exist, be listed for approval, and declare needs_approval, or the HITL
    # demo does nothing.
    from agent.agent import REQUIRE_APPROVAL

    assert "send_message" in REQUIRE_APPROVAL
    send = next(t for t in all_tools() if t.name == "send_message")
    assert send.needs_approval is True


class _FakeItem:
    """Stand-in for an Agents SDK run item, matched by _normalize_item's isinstance checks."""


def test_normalize_message_item():
    from agents.items import MessageOutputItem

    item = object.__new__(MessageOutputItem)
    # ItemHelpers.text_message_output reads raw_item.content; give it a text part.
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    item.raw_item = ResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text="hello", annotations=[])],
    )
    assert _normalize_item(item) == {"role": "assistant", "content": "hello"}


class _FakeToolApproval:
    def __init__(self, name, args, call_id):
        self.tool_name, self.arguments, self.call_id = name, args, call_id


class _FakeStreamResult:
    """Minimal RunResultStreaming stand-in: a delta, a message, then a pending interruption."""

    def __init__(self, events, interruptions, state):
        self._events, self.interruptions, self._state = events, interruptions, state

    async def stream_events(self):
        for event in self._events:
            yield event

    def to_state(self):
        return self._state


@pytest.mark.asyncio
async def test_serialize_events_relays_interrupt_as_native_event():
    from agent.agent import _pending_runs

    approval = _FakeToolApproval("send_message", '{"recipient": "x", "body": "y"}', "call-1")
    sentinel_state = object()
    result = _FakeStreamResult([], [approval], sentinel_state)

    events = [e async for e in _serialize_events(result, "sess-1")]

    assert events == [
        {
            "type": "interrupt",
            "id": "call-1",
            "value": {"action_requests": [{"name": "send_message", "args": {"recipient": "x", "body": "y"}}]},
        }
    ]
    # The paused run is stashed in-process, keyed by session id, for a later resume.
    assert _pending_runs.pop("sess-1") is sentinel_state


def test_apply_decisions_approves_pending_run(monkeypatch):
    from agent.agent import _pending_runs

    approved = []

    class _State:
        def get_interruptions(self):
            return ["item-a"]

        def approve(self, item):
            approved.append(item)

        def reject(self, item, rejection_message=None):
            raise AssertionError("should not reject on approve")

    _pending_runs["sess-2"] = _State()
    state = _apply_decisions("sess-2", {"decisions": [{"type": "approve"}]})
    assert approved == ["item-a"]
    assert "sess-2" not in _pending_runs  # popped so it can't be resumed twice


def test_apply_decisions_without_pending_run_raises():
    with pytest.raises(RuntimeError, match="No paused run"):
        _apply_decisions("never-started", {"decisions": [{"type": "approve"}]})


def test_configure_raises_clear_error_without_auth(monkeypatch):
    from agent.agent import configure

    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/nonexistent-databrickscfg")
    with pytest.raises(RuntimeError, match="Databricks auth is not configured"):
        configure()


def test_session_store_defaults_to_in_process(monkeypatch):
    import databricks_mason.openai.sessions as ss

    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    ss._local_sessions.clear()
    # In-process default: same session id returns the same cached SQLiteSession (multi-turn works).
    assert ss.session_store("abc-123") is ss.session_store("abc-123")


def test_session_store_selects_durable_store(monkeypatch):
    import databricks_mason.openai.sessions as ss

    monkeypatch.setenv("AGENT_SESSION_STORE", "my-store")
    monkeypatch.setattr(ss, "SessionStoreClient", lambda *a, **k: _FakeStoreClient())
    store = ss.session_store("abc-123")
    assert isinstance(store, ss.DatabricksSessionStore)


class _FakeStoreClient:
    def set_session_store(self, name):
        return self


def test_session_id_from_request():
    request = {"input": [{"role": "user", "content": "hi"}], "session_id": "abc-123"}
    assert _session_id(request) == "abc-123"


def test_session_id_is_required_from_runtime():
    with pytest.raises(KeyError):
        _session_id({"input": [{"role": "user", "content": "hi"}]})


@pytest.mark.asyncio
async def test_stream_handler_forwards_selected_model(monkeypatch):
    # The demo UI's model picker sends `model` in the request body; it must reach create_agent.
    import agent.agent as agent_module

    captured = {}

    def _fake_create_agent(mcp=None, model=None):
        captured["model"] = model
        return object()

    class _FakeResult:
        interruptions: list = []

        async def stream_events(self):
            return
            yield  # pragma: no cover - marks this an (empty) async generator

    class _FakeRunner:
        @staticmethod
        def run_streamed(agent, run_input, session=None):
            return _FakeResult()

    async def _fake_mcp_servers(servers):
        return []

    monkeypatch.setattr(agent_module, "create_agent", _fake_create_agent)
    monkeypatch.setattr(agent_module, "mcp_servers", _fake_mcp_servers)
    monkeypatch.setattr(agent_module, "session_store", lambda session_id: None)
    monkeypatch.setattr(agent_module, "Runner", _FakeRunner)
    monkeypatch.setattr(agent_module, "tag_session", lambda *a, **k: None)

    events = [
        event
        async for event in agent_module.stream_handler(
            {"session_id": "s1", "input": [], "model": "system.ai.claude-sonnet-4-5"}
        )
    ]
    assert events == []
    assert captured["model"] == "system.ai.claude-sonnet-4-5"


def _has_workspace_auth() -> bool:
    return bool(
        os.getenv("DATABRICKS_CONFIG_PROFILE")
        or (os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))
    )


@pytest.mark.skipif(
    not _has_workspace_auth(),
    reason="no Databricks profile configured; skipping live model call",
)
@pytest.mark.asyncio
async def test_agent_responds_end_to_end():
    from agents import Runner

    from agent.agent import configure, create_agent
    from databricks_mason.openai import session_store

    configure()
    agent = create_agent()
    result = await Runner.run(
        agent,
        [{"role": "user", "content": "Reply with the single word: pong"}],
        session=session_store("test-e2e"),
    )
    assert result.final_output
