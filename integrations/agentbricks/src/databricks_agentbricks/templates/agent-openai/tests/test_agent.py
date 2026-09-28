"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, event serialization) — no
Databricks auth needed, so they run anywhere. The live test builds the full agent and calls the
model; it is skipped unless a workspace profile is configured.
"""

import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent.agent import resume_agent
from agent.tools import all_tools
from agents import FunctionTool
from runtime.adapter import _normalize_item, _serialize_events


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
        self.final_output = None

    async def stream_events(self):
        for event in self._events:
            yield event

    def to_state(self):
        return self._state


@pytest.mark.asyncio
async def test_agent_events_omit_unavailable_mcp_servers(monkeypatch):
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
    monkeypatch.setattr(agent_module, "session_store", lambda _session_id, _actor: None)
    monkeypatch.setattr(
        agent_module.Runner,
        "run_streamed",
        lambda *_args, **_kwargs: _FakeStreamResult([], [], None),
    )

    async with agent_module.run_agent([], session_id="s", actor="actor") as result:
        assert [event async for event in result.stream_events()] == []
    # create_agent(actor, mcp) — the healthy servers are the second positional arg.
    assert create_agent.call_args.args[1] == [healthy]
    assert healthy.cache_tools_list is True
    assert all(
        server.tool_filter is tool_filter
        for server, tool_filter in zip(all_servers, tool_filters, strict=True)
    )
    for server in all_servers:
        server.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_events_propagate_request_user_mcp_failure(monkeypatch):
    import agent.agent as agent_module

    server = MagicMock(name="request-user")
    server.name = "request-user"
    server._agentbricks_request_user = True
    server.connect = AsyncMock(side_effect=PermissionError("request-user denied"))
    server.cleanup = AsyncMock()

    async def mcp_servers(_extra):
        return [server]

    monkeypatch.setattr(agent_module, "mcp_servers", mcp_servers)
    monkeypatch.setattr(agent_module, "build_mcp_servers", lambda: [])
    monkeypatch.setattr(agent_module, "create_agent", MagicMock(return_value=object()))
    monkeypatch.setattr(agent_module, "session_store", lambda _session_id, _actor: None)
    monkeypatch.setattr(
        agent_module.Runner,
        "run_streamed",
        lambda *_args, **_kwargs: _FakeStreamResult([], [], None),
    )

    with pytest.raises(PermissionError, match="request-user denied"):
        async with agent_module.run_agent([], session_id="s", actor="actor"):
            pass


@pytest.mark.asyncio
async def test_serialize_events_relays_interrupt_as_native_event():
    approval = _FakeToolApproval("send_message", '{"recipient": "x", "body": "y"}', "call-1")
    sentinel_state = object()
    result = _FakeStreamResult([], [approval], sentinel_state)

    events = [e async for e in _serialize_events(result)]

    assert events == [
        {
            "type": "interrupt",
            "id": "call-1",
            "value": {
                "action_requests": [
                    {"name": "send_message", "args": {"recipient": "x", "body": "y"}}
                ]
            },
        }
    ]


def test_resume_agent_approves_pending_run(monkeypatch):
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
    resume_agent("sess-2", {"decisions": [{"type": "approve"}]})
    assert approved == ["item-a"]
    assert "sess-2" not in _pending_runs  # popped so it can't be resumed twice


def test_resume_agent_without_pending_run_raises():
    with pytest.raises(RuntimeError, match="No paused run"):
        resume_agent("never-started", {"decisions": [{"type": "approve"}]})


def test_configure_raises_clear_error_without_auth(monkeypatch):
    from agent.agent import configure

    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/nonexistent-databrickscfg")
    with pytest.raises(RuntimeError, match="Databricks auth is not configured"):
        configure()


def test_configure_routes_openai_client_to_workspace(monkeypatch):
    import agent.agent as agent_module
    import agents

    workspace = object()
    created = object()
    client = MagicMock(return_value=created)
    set_client = MagicMock()
    set_api = MagicMock()

    monkeypatch.setattr(agent_module, "workspace_client", lambda: workspace)
    monkeypatch.setattr(
        agent_module,
        "workspace_headers",
        lambda: {"X-Databricks-Org-Id": "123"},
    )
    monkeypatch.setattr(agent_module, "AsyncDatabricksOpenAI", client)
    monkeypatch.setattr(agent_module, "configure_tracing", lambda: None)
    monkeypatch.setattr(agents, "set_default_openai_client", set_client)
    monkeypatch.setattr(agents, "set_default_openai_api", set_api)

    agent_module.configure()

    client.assert_called_once_with(
        workspace_client=workspace,
        default_headers={"X-Databricks-Org-Id": "123"},
        use_ai_gateway=True,
    )
    set_client.assert_called_once_with(created)
    set_api.assert_called_once_with("chat_completions")


def test_session_store_defaults_to_in_process(monkeypatch):
    import databricks_agentkit.openai.sessions as ss

    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    ss._local_sessions.clear()
    # In-process default: same session id returns the same cached SQLiteSession (multi-turn works).
    assert ss.session_store("abc-123") is ss.session_store("abc-123")


def test_session_store_selects_durable_store(monkeypatch):
    import databricks_agentkit.openai.sessions as ss

    monkeypatch.setenv("AGENT_SESSION_STORE", "my-store")
    monkeypatch.setattr(ss, "SessionStoreClient", lambda *a, **k: _FakeStoreClient())
    store = ss.session_store("abc-123")
    assert isinstance(store, ss.DatabricksSessionStore)


class _FakeStoreClient:
    def set_session_store(self, name):
        return self


@pytest.mark.asyncio
async def test_adapter_recovery_marks_replayed_agent_input(monkeypatch):
    import runtime.adapter as adapter

    calls = []

    @asynccontextmanager
    async def fake_run_agent(agent_input, **kwargs):
        calls.append((agent_input, kwargs))
        yield _FakeStreamResult([], [], None)

    monkeypatch.setattr(adapter, "run_agent", fake_run_agent)
    payload = {"session_id": "session-1", "messages": [{"role": "user", "content": "hi"}]}
    context = SimpleNamespace(session_id="runtime-session", emit=AsyncMock())

    await adapter.invoke(payload, context)
    await adapter.recover(payload, context)

    assert calls == [
        (payload["messages"], {"session_id": "session-1", "actor": "session-1", "model": None}),
        (
            [
                {"role": "developer", "content": adapter._RECOVERY_INSTRUCTION},
                *payload["messages"],
            ],
            {"session_id": "session-1", "actor": "session-1", "model": None},
        ),
    ]
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


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
    from agent.agent import configure, run_agent

    configure()
    async with run_agent(
        [{"role": "user", "content": "Reply with the single word: pong"}],
        session_id="test-e2e",
        actor="test-actor",
    ) as result:
        _ = [event async for event in result.stream_events()]
        assert result.final_output
