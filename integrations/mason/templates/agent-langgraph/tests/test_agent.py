"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, event serialization) — no
Databricks auth needed, so they run anywhere. The live test builds the full agent and calls the
model; it is skipped unless a workspace profile is configured.
"""

import os
from unittest.mock import patch

import pytest
from agent.agent import _serialize_events, _session_id, _workspace_client
from agent.mason.session_store import checkpointer, thread_config
from agent.tools import all_tools
from fastapi.testclient import TestClient
from langchain_core.tools import BaseTool
from runtime.runtime import build_app


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


def test_workspace_client_prefers_named_profile(monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "test-profile")
    monkeypatch.setenv("DATABRICKS_HOST", "https://inherited.example.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "inherited-token")

    with patch("agent.agent.WorkspaceClient") as workspace_client:
        _workspace_client()

    workspace_client.assert_called_once_with(profile="test-profile")
    assert "DATABRICKS_HOST" not in os.environ
    assert "DATABRICKS_TOKEN" not in os.environ


def test_thread_config_from_session_id():
    # actor_id rides alongside thread_id — the durable saver maps it onto the Session's actor.
    assert thread_config("abc-123") == {"configurable": {"thread_id": "abc-123", "actor_id": "abc-123"}}


def test_thread_config_uses_configured_actor(monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_ACTOR_ID", "alice")
    assert thread_config("abc-123") == {"configurable": {"thread_id": "abc-123", "actor_id": "alice"}}


def test_checkpointer_is_shared(monkeypatch):
    # In-memory by default (no AGENT_SESSION_STORE); built once and shared so multi-turn works.
    import agent.mason.session_store as ss

    monkeypatch.setattr(ss, "_saver", None)  # reset the process-wide saver
    assert checkpointer() is checkpointer()


def test_session_store_selects_durable_saver(monkeypatch):
    # AGENT_SESSION_STORE must route to the durable Session Store saver, not stay in-memory. Stub the
    # REST client so it stays hermetic (no network); the saver builds without touching the API.
    import agent.mason.session_store as ss

    monkeypatch.setattr(ss, "_saver", None)
    monkeypatch.setenv("AGENT_SESSION_STORE", "my-store")
    monkeypatch.setattr(ss, "SessionStoreClient", lambda *a, **k: _FakeStoreClient())
    saver = checkpointer()
    assert isinstance(saver, ss.DatabricksSessionStoreSaver)


class _FakeStoreClient:
    def set_session_store(self, name):
        return self


def test_session_id_from_request():
    request = {"input": [{"role": "user", "content": "hi"}], "session_id": "abc-123"}
    assert _session_id(request) == "abc-123"


def test_session_id_generated_when_absent():
    generated = _session_id({"input": [{"role": "user", "content": "hi"}]})
    assert generated and generated != _session_id({"input": [{"role": "user", "content": "hi"}]})


def test_runtime_passes_same_body_session_id_to_resume_request():
    captured = {}

    async def invoke_handler(request):
        captured.update(request)
        return {"output": [], "session_id": request["session_id"], "status": "completed"}

    async def stream_handler(request):
        if False:
            yield request

    client = TestClient(build_app(invoke_handler, stream_handler))
    response = client.post(
        "/invocations",
        json={
            "session_id": "same-session-id",
            "resume": {"decisions": [{"type": "approve"}]},
        },
    )

    assert response.status_code == 200
    assert captured == {
        "resume": {"decisions": [{"type": "approve"}]},
        "session_id": "same-session-id",
    }


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
    agent = await create_agent_graph()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Reply with the single word: pong"}]},
        config=thread_config("test-e2e"),
    )
    assert result["messages"][-1].content
