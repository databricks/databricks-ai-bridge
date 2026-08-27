"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, wire) — no Databricks auth
needed, so they run anywhere. The live test builds the full agent and calls the model; it is
skipped unless a workspace profile is configured.
"""

import os

import pytest
from langchain_core.tools import BaseTool

from agent.mason.session_store import checkpointer, thread_config
from agent.mason.wire.inbound import get_resume, get_session_id
from agent.mason.wire.outbound import process_agent_astream_events
from agent.tools import all_tools


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


def test_get_resume_reads_native_payload():
    assert get_resume({"resume": {"decisions": [{"type": "approve"}]}}) == {"decisions": [{"type": "approve"}]}
    assert get_resume({"input": [{"role": "user", "content": "hi"}]}) is None


class _FakeInterrupt:
    def __init__(self, value, id):  # mirrors langgraph.types.Interrupt's `.value` / `.id`
        self.value, self.id = value, id


async def _aiter(events):
    for e in events:
        yield e


@pytest.mark.asyncio
async def test_outbound_relays_interrupt_as_native_event():
    hitl = {"action_requests": [{"name": "send_message", "args": {"recipient": "x", "body": "y"}}]}
    stream = _aiter([("updates", {"__interrupt__": (_FakeInterrupt(hitl, "int-1"),)})])
    events = [e async for e in process_agent_astream_events(stream)]
    assert events == [{"type": "interrupt", "id": "int-1", "value": hitl}]


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


def test_checkpointer_is_shared():
    # Cached so multi-turn history is preserved in-process across requests.
    assert checkpointer() is checkpointer()


def test_session_id_from_request():
    request = {"input": [{"role": "user", "content": "hi"}], "session_id": "abc-123"}
    assert get_session_id(request) == "abc-123"


def test_session_id_generated_when_absent():
    generated = get_session_id({"input": [{"role": "user", "content": "hi"}]})
    assert generated and generated != get_session_id({"input": [{"role": "user", "content": "hi"}]})


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
