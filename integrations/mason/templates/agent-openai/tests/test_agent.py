"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, event serialization) — no
Databricks auth needed, so they run anywhere. The live test builds the full agent and calls the
model; it is skipped unless a workspace profile is configured.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents import FunctionTool
from agent.agent import _apply_decisions, _normalize_item, _serialize_events
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
async def test_agent_events_omits_unavailable_mcp_servers(monkeypatch):
    import agent.agent as agent_module

    def server(name, *, connect_error=None, list_error=None, cleanup_error=None):
        value = MagicMock(name=name)
        value.name = name
        value.tool_filter = lambda *_args: True
        value.cache_tools_list = False
        value.connect = AsyncMock(side_effect=connect_error)
        value.cleanup = AsyncMock(side_effect=cleanup_error)
        value.list_tools = AsyncMock(return_value=[], side_effect=list_error)
        return value

    healthy = server("healthy")
    unavailable = [
        server("connect-failure", connect_error=PermissionError("HTTP error 403")),
        server(
            "list-failure",
            list_error=RuntimeError("tool discovery failed"),
            cleanup_error=RuntimeError("cleanup failed"),
        ),
    ]
    all_servers = [healthy, *unavailable]
    tool_filters = [server.tool_filter for server in all_servers]

    async def mcp_servers(_extra):
        return [healthy, *unavailable]

    create_agent = MagicMock(return_value=object())

    monkeypatch.setattr(agent_module, "mcp_servers", mcp_servers)
    monkeypatch.setattr(agent_module, "build_mcp_servers", lambda: [])
    monkeypatch.setattr(agent_module, "create_agent", create_agent)
    monkeypatch.setattr(agent_module, "tag_session", lambda _session_id: None)
    monkeypatch.setattr(agent_module, "session_store", lambda _session_id, _actor: None)
    monkeypatch.setattr(
        agent_module.Runner,
        "run_streamed",
        lambda *_args, **_kwargs: _FakeStreamResult([], [], None),
    )

    context = SimpleNamespace(session_id="s")
    assert [event async for event in agent_module._agent_events({}, context)] == []
    # create_agent(session_id, mcp) — the healthy servers are the second positional arg.
    assert create_agent.call_args.args[0] == "s"
    assert create_agent.call_args.args[1] == [healthy]
    assert healthy.cache_tools_list is True
    assert all(
        server.tool_filter is tool_filter
        for server, tool_filter in zip(all_servers, tool_filters, strict=True)
    )
    for server in all_servers:
        server.cleanup.assert_awaited_once()


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
    agent = create_agent("test-actor")
    result = await Runner.run(
        agent,
        [{"role": "user", "content": "Reply with the single word: pong"}],
        session=session_store("test-e2e", "test-actor"),
    )
    assert result.final_output
