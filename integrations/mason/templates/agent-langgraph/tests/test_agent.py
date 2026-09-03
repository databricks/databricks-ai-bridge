"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, event serialization) — no
Databricks auth needed, so they run anywhere. The live test builds the full agent and calls the
model; it is skipped unless a workspace profile is configured.
"""

import os
from types import SimpleNamespace

import pytest
from agent.agent import _serialize_events
from agent.tools import all_tools
from databricks_mason.langgraph import checkpointer, thread_config
from langchain_core.tools import BaseTool


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

    assert model._get_client_kwargs()["default_headers"] == {
        "X-Databricks-Org-Id": "123456"
    }


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

    monkeypatch.setattr(ss, "_saver", None)
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


@pytest.mark.asyncio
async def test_invoke_starts_a_turn_and_recover_resumes_the_checkpoint(monkeypatch):
    import agent.agent as agent_module

    calls = []

    async def fake_run_agent(agent_input, context):
        calls.append(agent_input)
        return {"output": []}

    class Saver:
        async def aget_tuple(self, config):
            return SimpleNamespace(metadata={"databricks_mason.run_id": "run-1"})

    monkeypatch.setattr(agent_module, "_run_agent", fake_run_agent)
    monkeypatch.setattr(agent_module, "checkpointer", lambda: Saver())
    context = SimpleNamespace(run_id="run-1", session_id="session-1", actor="alice")
    messages = [{"role": "user", "content": "hello"}]

    await agent_module.invoke({"input": messages}, context)
    await agent_module.recover({"input": messages}, context)

    assert calls[0] == {"messages": messages}
    assert calls[1] is None


@pytest.mark.asyncio
async def test_recover_replays_input_when_no_checkpoint_exists(monkeypatch):
    import agent.agent as agent_module

    calls = []

    async def fake_run_agent(agent_input, context):
        calls.append(agent_input)
        return {"output": []}

    class Saver:
        async def aget_tuple(self, config):
            return None

    monkeypatch.setattr(agent_module, "_run_agent", fake_run_agent)
    monkeypatch.setattr(agent_module, "checkpointer", lambda: Saver())
    messages = [{"role": "user", "content": "hello"}]

    await agent_module.recover(
        {"input": messages},
        SimpleNamespace(run_id="run-1", session_id="session-1", actor="alice"),
    )

    assert calls == [{"messages": messages}]


@pytest.mark.asyncio
async def test_recover_replays_input_when_checkpoint_belongs_to_an_older_run(monkeypatch):
    import agent.agent as agent_module

    calls = []

    async def fake_run_agent(agent_input, context):
        calls.append(agent_input)
        return {"output": []}

    class Saver:
        async def aget_tuple(self, config):
            return SimpleNamespace(metadata={"databricks_mason.run_id": "run-0"})

    monkeypatch.setattr(agent_module, "_run_agent", fake_run_agent)
    monkeypatch.setattr(agent_module, "checkpointer", lambda: Saver())
    messages = [{"role": "user", "content": "hello"}]

    await agent_module.recover(
        {"input": messages},
        SimpleNamespace(run_id="run-1", session_id="session-1", actor="alice"),
    )

    assert calls == [{"messages": messages}]


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
    from agent.agent import configure, create_agent_graph

    configure()
    agent = await create_agent_graph("test-actor")
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Reply with the single word: pong"}]},
        config=thread_config("test-e2e", "test-actor"),
    )
    assert result["messages"][-1].content
