"""Hermetic smoke tests for the generated agent."""

from types import SimpleNamespace

import pytest
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from agent.agent import _serialize_events, agent, app
from agent.session_store import checkpointer, thread_config
from agent.tools.sample_tool import get_current_time
from databricks_ai_bridge.durable_runtime import InMemoryDurabilityStore


class _Message:
    def model_dump(self):
        return {"type": "ai", "content": "pong"}


async def _aiter(events):
    for event in events:
        yield event


def test_sample_tool_is_registered():
    assert isinstance(get_current_time, BaseTool)


def test_runtime_defaults_to_process_memory_locally():
    assert isinstance(app.runtime.durability_store, InMemoryDurabilityStore)


@pytest.mark.asyncio
async def test_serialize_events_relays_messages_and_deltas():
    stream = _aiter(
        [
            ("updates", {"agent": {"messages": [_Message()]}}),
            ("messages", (SimpleNamespace(content="ignored"), {})),
        ]
    )
    events = [event async for event in _serialize_events(stream)]
    assert events == [{"type": "message", "message": {"type": "ai", "content": "pong"}}]


def test_configure_raises_clear_error_without_auth(monkeypatch):
    from agent.agent import configure

    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/nonexistent-databrickscfg")
    with pytest.raises(RuntimeError, match="Databricks auth is not configured"):
        configure()


def test_thread_config_from_session_id():
    assert thread_config("abc-123") == {"configurable": {"thread_id": "abc-123"}}


@pytest.mark.asyncio
async def test_checkpointer_is_shared(monkeypatch):
    import agent.session_store as session_store

    monkeypatch.setattr(session_store, "_saver", None)
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    assert isinstance(await checkpointer(), InMemorySaver)
    assert await checkpointer() is await checkpointer()


@pytest.mark.asyncio
async def test_session_store_selects_durable_checkpointer(monkeypatch):
    import agent.session_store as session_store

    monkeypatch.setattr(session_store, "_saver", None)
    selected = False

    async def fake_durable():
        nonlocal selected
        selected = True
        return InMemorySaver()

    monkeypatch.setenv("AGENT_SESSION_STORE", "my-store")
    monkeypatch.setattr(session_store, "_durable_checkpointer", fake_durable)
    await checkpointer()
    assert selected


@pytest.mark.asyncio
async def test_entrypoint_emits_stream_events(monkeypatch):
    class Graph:
        def astream(self, **kwargs):
            return _aiter([("updates", {"agent": {"messages": [_Message()]}})])

    async def fake_graph():
        return Graph()

    emitted = []

    async def emit(event):
        emitted.append(event)
        return len(emitted)

    monkeypatch.setattr("agent.agent.create_agent_graph", fake_graph)
    context = SimpleNamespace(session_id="session-1", emit=emit)

    result = await agent({"input": [{"role": "user", "content": "ping"}]}, context)

    assert emitted == [{"type": "message", "message": {"type": "ai", "content": "pong"}}]
    assert result == {
        "output": [{"type": "ai", "content": "pong"}],
        "session_id": "session-1",
        "status": "completed",
    }
