"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, event serialization) — no
Databricks auth needed, so they run anywhere. The live test builds the full agent and calls the
model; it is skipped unless a workspace profile is configured.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agent.tools import all_tools
from langchain_core.tools import BaseTool
from runtime.adapter import _serialize_events

from databricks_mason.langgraph.session_store import checkpointer, thread_config


def test_tools_autoregister():
    tools = all_tools()
    assert tools, "expected the sample tool to auto-register"
    assert all(isinstance(t, BaseTool) for t in tools)
    assert {"get_current_time", "send_message"} <= {t.name for t in tools}


def test_gated_tool_is_in_require_approval():
    # The gated demo tool must exist and be listed for approval, or the HITL demo does nothing.
    from agent.agent import REQUIRE_APPROVAL

    assert REQUIRE_APPROVAL.get("send_message")
    assert "send_message" in {t.name for t in all_tools()}


class _FakeInterrupt:
    def __init__(self, value, id):  # mirrors langgraph.types.Interrupt's `.value` / `.id`
        self.value, self.id = value, id


async def _aiter(events):
    for e in events:
        yield e


@pytest.mark.asyncio
async def test_serialize_events_relays_interrupt_as_native_event():
    hitl = {"action_requests": [{"name": "send_message", "args": {"recipient": "x", "body": "y"}}]}
    stream = _aiter([("updates", {"__interrupt__": (_FakeInterrupt(hitl, "int-1"),)})])
    events = [e async for e in _serialize_events(stream)]
    assert events == [{"type": "interrupt", "id": "int-1", "value": hitl}]


def test_configure_raises_clear_error_without_auth(monkeypatch):
    from agent.agent import configure

    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/nonexistent-databrickscfg")
    with pytest.raises(RuntimeError, match="Databricks auth is not configured"):
        configure()


def test_chat_model_forwards_account_routing_header(monkeypatch):
    from agent.agent import _RoutedChatDatabricks

    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "123456")
    model = _RoutedChatDatabricks(endpoint="test-endpoint")

    assert model._get_client_kwargs()["default_headers"] == {"X-Databricks-Org-Id": "123456"}


def test_thread_config_from_session_id():
    # actor_id rides alongside thread_id — the durable saver maps it onto the Session's actor.
    assert thread_config("abc-123") == {
        "configurable": {"thread_id": "abc-123", "actor_id": "abc-123"}
    }


def test_thread_config_uses_supplied_actor():
    # A caller-supplied actor (e.g. the signed-in user) partitions the durable store per user.
    assert thread_config("abc-123", "alice") == {
        "configurable": {"thread_id": "abc-123", "actor_id": "alice"}
    }


def test_checkpointer_is_shared(monkeypatch):
    # In-memory by default (no AGENT_SESSION_STORE); built once and shared so multi-turn works.
    import databricks_mason.langgraph.session_store as ss

    monkeypatch.setattr(ss, "_saver", None)  # reset the process-wide saver
    assert checkpointer() is checkpointer()


def test_session_store_selects_durable_saver(monkeypatch):
    # AGENT_SESSION_STORE must route to the durable Session Store saver, not stay in-memory. Stub the
    # REST client so it stays hermetic (no network); the saver builds without touching the API.
    import databricks_mason.langgraph.session_store as ss

    monkeypatch.setattr(ss, "_saver", None)
    monkeypatch.setenv("AGENT_SESSION_STORE", "my-store")
    monkeypatch.setattr(ss, "SessionStoreClient", lambda *a, **k: _FakeStoreClient())
    saver = checkpointer()
    assert isinstance(saver, ss.DatabricksSessionStoreSaver)


class _FakeStoreClient:
    def set_session_store(self, name):
        return self


@pytest.mark.asyncio
async def test_recovery_input_resumes_current_checkpoint(monkeypatch):
    import agent.agent as agent_module

    class Saver:
        async def aget_tuple(self, config):
            return SimpleNamespace(metadata={"databricks_mason.invocation_id": "inv-1"})

    monkeypatch.setattr(agent_module, "checkpointer", lambda: Saver())
    original = {"messages": [{"role": "user", "content": "hi"}]}
    recovered = await agent_module.recovery_input(
        original,
        session_id="session-1",
        actor="actor-1",
        invocation_id="inv-1",
    )

    assert recovered is None


@pytest.mark.asyncio
async def test_recovery_replays_input_without_current_checkpoint(monkeypatch):
    import agent.agent as agent_module

    class Saver:
        async def aget_tuple(self, config):
            return None

    monkeypatch.setattr(agent_module, "checkpointer", lambda: Saver())
    original = {"messages": [{"role": "user", "content": "hi"}]}
    recovered = await agent_module.recovery_input(
        original,
        session_id="session-1",
        actor="actor-1",
        invocation_id="inv-1",
    )

    assert recovered is original


@pytest.mark.asyncio
async def test_adapter_calls_same_run_agent_for_invoke_and_recovery(monkeypatch):
    import runtime.adapter as adapter

    calls = []

    async def fake_run_agent(agent_input, **kwargs):
        calls.append((agent_input, kwargs))
        if False:
            yield None

    async def fake_recovery_input(agent_input, **_kwargs):
        return None

    monkeypatch.setattr(adapter, "run_agent", fake_run_agent)
    monkeypatch.setattr(adapter, "recovery_input", fake_recovery_input)
    context = SimpleNamespace(
        invocation_id="inv-1",
        session_id="runtime-session",
        emit=AsyncMock(),
    )
    payload = {"session_id": "session-1", "messages": [{"role": "user", "content": "hi"}]}

    await adapter.invoke(payload, context)
    await adapter.recover(payload, context)

    assert calls[0][0] == {"messages": payload["messages"]}
    assert calls[1][0] is None
    assert calls[0][1] == calls[1][1]


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
    events = [
        event
        async for event in run_agent(
            {"messages": [{"role": "user", "content": "Reply with the single word: pong"}]},
            session_id="test-e2e",
            actor="test-actor",
        )
    ]
    assert events
