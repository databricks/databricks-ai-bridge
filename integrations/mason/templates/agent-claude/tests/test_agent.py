"""Smoke tests for the Claude agent.

Hermetic tests import only leaf modules and mock the Anthropic client (no live API, no auth), so they
run anywhere. The live test builds the full agent and calls Claude; it is skipped unless an Anthropic
credential is configured.
"""

import os

import pytest
from anthropic.lib.tools import BetaFunctionTool

from agent.agent import (
    REQUIRE_APPROVAL,
    _agent_events,
    _normalize,
    _run_agent,
    invoke,
    on_recovery,
)
from agent.tools import all_tools


class _Block:
    def __init__(self, type, *, text=None, name=None, input=None, id=None):
        self.type, self.text, self.name, self.input, self.id = type, text, name, input, id


class _Msg:
    def __init__(self, content, *, id="m1", stop_reason="end_turn"):
        self.content, self.id, self.stop_reason = content, id, stop_reason


class _Ctx:
    session_id = "sess"

    async def emit(self, event):
        return 0


def _fake_client(messages):
    """A stand-in anthropic client whose tool_runner yields ``messages``."""
    from types import SimpleNamespace

    def tool_runner(**_kwargs):
        return iter(messages)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(tool_runner=tool_runner)))
    return lambda: client


def test_tools_autoregister():
    tools = all_tools()
    assert tools, "expected the sample tools to auto-register"
    assert all(isinstance(t, BetaFunctionTool) for t in tools)
    assert {"get_current_time", "send_message"} <= {t.name for t in tools}


def test_gated_tool_is_listed_for_approval():
    assert "send_message" in REQUIRE_APPROVAL
    assert "send_message" in {t.name for t in all_tools()}


def test_normalize_text_and_tool_calls():
    content = [
        _Block("text", text="hi"),
        _Block("tool_use", name="send_message", input={"recipient": "x", "body": "y"}, id="c1"),
    ]
    assert _normalize(content) == {
        "role": "assistant",
        "content": "hi",
        "tool_calls": [{"name": "send_message", "args": {"recipient": "x", "body": "y"}}],
    }


@pytest.mark.asyncio
async def test_agent_events_emit_delta_and_message(monkeypatch):
    import agent.agent as agent_module

    monkeypatch.setattr(agent_module, "mcp_servers", lambda *_a: [])
    monkeypatch.setattr(agent_module, "create_tools", lambda _actor: [])
    monkeypatch.setattr(agent_module, "claude_client", _fake_client([_Msg([_Block("text", text="hello")])]))

    events = [
        e async for e in _agent_events({"messages": [{"role": "user", "content": "hi"}]}, "s1", "a")
    ]
    assert {"type": "delta", "content": "hello", "id": "m1"} in events
    assert {"type": "message", "message": {"role": "assistant", "content": "hello"}} in events


@pytest.mark.asyncio
async def test_gated_tool_surfaces_interrupt(monkeypatch):
    import agent.agent as agent_module

    monkeypatch.setattr(agent_module, "mcp_servers", lambda *_a: [])
    monkeypatch.setattr(agent_module, "create_tools", lambda _actor: [])
    tool_use = _Block("tool_use", name="send_message", input={"recipient": "x", "body": "y"}, id="call-1")
    monkeypatch.setattr(agent_module, "claude_client", _fake_client([_Msg([tool_use])]))

    result = await _run_agent({"session_id": "s2", "messages": [{"role": "user", "content": "msg x"}]}, _Ctx())
    assert result["status"] == "interrupted"
    assert result["output"][-1] == {
        "type": "interrupt",
        "id": "call-1",
        "value": {"action_requests": [{"name": "send_message", "args": {"recipient": "x", "body": "y"}}]},
    }


@pytest.mark.asyncio
async def test_run_agent_completed_envelope(monkeypatch):
    import agent.agent as agent_module

    monkeypatch.setattr(agent_module, "mcp_servers", lambda *_a: [])
    monkeypatch.setattr(agent_module, "create_tools", lambda _actor: [])
    monkeypatch.setattr(agent_module, "claude_client", _fake_client([_Msg([_Block("text", text="done")])]))

    result = await _run_agent({"session_id": "s3", "messages": [{"role": "user", "content": "hi"}]}, _Ctx())
    assert result["session_id"] == "s3"
    assert result["status"] == "completed"
    assert result["output"] == [{"role": "assistant", "content": "done"}]


def test_memory_tools_present_when_store_configured(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_STORE", "mem-store-1")
    from databricks_mason.claude import memory_tools

    tools = memory_tools("actor-1")
    assert [t.name for t in tools] == ["remember", "recall"]
    assert all(isinstance(t, BetaFunctionTool) for t in tools)


def test_memory_tools_absent_without_store(monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    from databricks_mason.claude import memory_tools

    assert memory_tools("actor-1") == []


def test_configure_raises_clear_error_without_credentials(monkeypatch):
    from agent.agent import configure

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
    with pytest.raises(RuntimeError, match="Anthropic auth is not configured"):
        configure()


@pytest.mark.asyncio
async def test_invoke_and_recovery_use_same_application_payload(monkeypatch):
    import agent.agent as agent_module

    calls = []

    async def fake_run_agent(payload, context):
        calls.append((payload, context))
        return {"output": []}

    monkeypatch.setattr(agent_module, "_run_agent", fake_run_agent)
    payload = {"session_id": "session-1", "messages": [{"role": "user", "content": "hi"}]}
    context = object()

    await invoke(payload, context)
    await on_recovery(payload, context)
    assert calls == [(payload, context), (payload, context)]


def _has_anthropic_auth() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_CODE_USE_BEDROCK"))


@pytest.mark.skipif(not _has_anthropic_auth(), reason="no Anthropic credential; skipping live call")
@pytest.mark.asyncio
async def test_agent_responds_end_to_end():
    from agent.agent import _run_agent, configure

    configure()
    result = await _run_agent(
        {"session_id": "e2e", "messages": [{"role": "user", "content": "Reply with: pong"}]},
        _Ctx(),
    )
    assert result["status"] == "completed"
    assert result["output"]
