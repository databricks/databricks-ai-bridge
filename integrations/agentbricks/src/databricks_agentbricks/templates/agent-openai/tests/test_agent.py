"""Smoke tests for the agent.

Hermetic tests import only the leaf modules (tools, session store, event serialization) — no
Databricks auth needed, so they run anywhere. The live test builds the full agent and calls the
model; it is skipped unless a workspace profile is configured.
"""

import json
import os
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from agent.agent import resume_agent
from agent.tools import all_tools
from agents import FunctionTool
from runtime.adapter import _approval_value, _normalize_item


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


@pytest.mark.parametrize("text", ["", "I checked the result."])
def test_normalize_reasoning_item_preserves_summary_and_metadata(text):
    from agents.items import ReasoningItem
    from openai.types.responses import ResponseReasoningItem

    item = object.__new__(ReasoningItem)
    item.raw_item = ResponseReasoningItem(
        id="r1",
        type="reasoning",
        summary=[{"type": "summary_text", "text": text, "signature": "opaque-signature"}],
        encrypted_content="opaque-content",
    )
    assert _normalize_item(item) == {
        "role": "assistant",
        "content": [item.raw_item.model_dump()],
    }


class _FakeToolApproval:
    def __init__(self, name, args, call_id):
        self.tool_name, self.arguments, self.call_id = name, args, call_id


class _FakeStreamResult:
    """Minimal RunResultStreaming stand-in: a delta, a message, then a pending interruption."""

    def __init__(self, events, interruptions, state):
        self._events, self.interruptions, self._state = events, interruptions, state
        self.final_output = None
        self.is_complete = True

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


def test_approval_value_preserves_tool_arguments():
    approval = _FakeToolApproval("send_message", '{"recipient": "x", "body": "y"}', "call-1")
    second = _FakeToolApproval("send_message", "{}", "call-2")
    assert _approval_value([approval, second]) == {
        "action_requests": [
            {"name": "send_message", "args": {"recipient": "x", "body": "y"}, "call_id": "call-1"},
            {"name": "send_message", "args": {}, "call_id": "call-2"},
        ]
    }


@pytest.mark.parametrize(
    "decisions",
    [
        None,
        [],
        {},
        [None],
        [{"type": "approve"}],
        [{"type": "invalid", "call_id": "call-1"}],
        [{"type": "reject", "call_id": "call-1", "message": 5}],
        [{"type": "approve", "call_id": "stale"}],
        [{"type": "approve"}, {"type": "approve"}],
    ],
)
def test_resume_agent_invalid_then_valid_does_not_consume_state(decisions):
    state = MagicMock()
    item = _FakeToolApproval("send_message", "{}", "call-1")
    state.get_interruptions.return_value = [item]
    with pytest.raises(ValueError):
        resume_agent(state, {"decisions": decisions})
    state.approve.assert_not_called()
    state.reject.assert_not_called()
    assert resume_agent(state, {"decisions": [{"type": "approve", "call_id": "call-1"}]}) is state
    state.approve.assert_called_once_with(item)


def test_resume_agent_validates_all_decisions_before_mutating():
    state = MagicMock()
    state.get_interruptions.return_value = [
        _FakeToolApproval("send_message", "{}", "call-1"),
        _FakeToolApproval("send_message", "{}", "call-2"),
    ]
    with pytest.raises(ValueError):
        resume_agent(
            state,
            {
                "decisions": [
                    {"type": "approve", "call_id": "call-1"},
                    {"type": "invalid", "call_id": "call-2"},
                ]
            },
        )
    state.approve.assert_not_called()
    state.reject.assert_not_called()
    resume_agent(
        state,
        {
            "decisions": [
                {"type": "approve", "call_id": "call-1"},
                {"type": "reject", "call_id": "call-2", "message": "No"},
            ]
        },
    )
    state.reject.assert_called_once_with(state.get_interruptions()[1], rejection_message="No")


@pytest.fixture
def approval_run(monkeypatch):
    import agent.agent as agent_module
    from agents import Agent, RunConfig, SQLiteSession, function_tool
    from agents.models.interface import Model
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    executed, models = [], []
    control = SimpleNamespace(model_calls=0, fail_after_tool=False, second_approval=False)
    store = {"state": None}
    session = SQLiteSession("approval-test")

    @function_tool(needs_approval=True)
    def send_message() -> str:
        """Send the approved message."""
        assert store["state"]["status"] == "consumed"
        executed.append("sent")
        return "sent"

    class ApprovalModel(Model):
        async def get_response(self, *args, **kwargs):
            raise AssertionError("expected streaming")

        async def stream_response(self, instructions, input, *args, **kwargs):
            control.model_calls += 1
            outputs = [item for item in input if item.get("type") == "function_call_output"]
            if not outputs or (control.second_approval and len(outputs) == 1):
                output = ResponseFunctionToolCall(
                    id=f"tool-{len(outputs) + 1}",
                    type="function_call",
                    call_id=f"call-{len(outputs) + 1}",
                    name="send_message",
                    arguments="{}",
                )
            else:
                if control.fail_after_tool:
                    control.fail_after_tool = False
                    raise RuntimeError("model failed after tool execution")
                output = ResponseOutputMessage(
                    id="message-1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="Done", annotations=[])],
                )
            response = Response(
                id="response-1",
                created_at=0,
                model="test",
                object="response",
                output=[output],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            )
            yield ResponseCompletedEvent(
                type="response.completed", response=response, sequence_number=0
            )

    def create_agent(actor, mcp, model=None, **kwargs):
        models.append(model)
        return Agent(name="Agent", model=ApprovalModel(), tools=[send_message])

    async def save(value):
        store["state"] = json.loads(json.dumps(value))

    save_state = AsyncMock(side_effect=save)
    import runtime.adapter as adapter

    checkpoints = SimpleNamespace(
        load=AsyncMock(side_effect=lambda: store["state"]), save=save_state
    )
    monkeypatch.setattr(adapter, "Checkpoints", lambda *_args: checkpoints)
    monkeypatch.setattr(agent_module, "create_agent", create_agent)
    monkeypatch.setattr(agent_module, "build_mcp_servers", lambda: [])
    monkeypatch.setattr(agent_module, "mcp_servers", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_module, "session_store", lambda *_args: session)
    monkeypatch.setattr(agent_module, "start_trace", lambda **_kwargs: nullcontext(None))
    run_streamed = agent_module.Runner.run_streamed
    monkeypatch.setattr(
        agent_module.Runner,
        "run_streamed",
        lambda *args, **kwargs: run_streamed(
            *args, **kwargs, run_config=RunConfig(tracing_disabled=True)
        ),
    )

    async def run(**kwargs):
        async with agent_module.run_agent(
            [{"role": "user", "content": "Send a message"}],
            session_id="approval-test",
            actor="actor",
            load_state=AsyncMock(side_effect=lambda: store["state"]),
            save_state=save_state,
            **kwargs,
        ) as result:
            async for _ in result.stream_events():
                pass
        return result

    return SimpleNamespace(
        run=run,
        control=control,
        store=store,
        executed=executed,
        models=models,
        session=session,
        save_state=save_state,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_approval_restores_sdk_state_original_model_and_session(approval_run, decision):
    run = approval_run
    await run.run(model="original-model", invocation_id="start")
    assert run.executed == []
    replayed = await run.run(invocation_id="start")
    assert [item.call_id for item in replayed.interruptions] == ["call-1"]
    assert run.control.model_calls == 1
    snapshot = json.loads(json.dumps(run.store["state"]))
    with pytest.raises(ValueError):
        await run.run(
            resume={"decisions": [{"type": "invalid", "call_id": "call-1"}]},
            invocation_id="invalid",
        )
    assert run.store["state"] == snapshot
    result = await run.run(
        model="changed-model",
        invocation_id="resume",
        resume={"decisions": [{"type": decision, "call_id": "call-1"}]},
    )
    assert run.models[-1] == "original-model"
    assert run.executed == (["sent"] if decision == "approve" else [])
    assert result.final_output == "Done"
    assert run.store["state"] == {"status": "consumed", "actor": "actor", "invocation_id": "resume"}
    history = await run.session.get_items()
    assert any(item.get("type") == "function_call_output" for item in history)
    assert history[-1]["content"][0]["text"] == "Done"


@pytest.mark.asyncio
async def test_approval_storage_failure_happens_before_tool_execution(approval_run):
    run = approval_run
    await run.run(model="original-model", invocation_id="start")
    snapshot = json.loads(json.dumps(run.store["state"]))
    run.save_state.side_effect = OSError("store unavailable")
    with pytest.raises(OSError, match="store unavailable"):
        await run.run(
            resume={"decisions": [{"type": "approve", "call_id": "call-1"}]}, invocation_id="resume"
        )
    assert run.store["state"] == snapshot
    assert run.executed == []


@pytest.mark.asyncio
async def test_approval_model_failure_refuses_replay_without_repeating_tool(approval_run):
    run = approval_run
    decisions = {"decisions": [{"type": "approve", "call_id": "call-1"}]}
    await run.run(model="original-model", invocation_id="start")
    run.control.fail_after_tool = True
    with pytest.raises(RuntimeError, match="model failed"):
        await run.run(resume=decisions, invocation_id="failed-resume")
    assert run.store["state"]["status"] == "consumed"
    snapshot = json.loads(json.dumps(run.store["state"]))
    calls = run.control.model_calls
    for invocation_id in ("failed-resume", "retry"):
        with pytest.raises(RuntimeError, match="already accepted"):
            await run.run(resume=decisions, invocation_id=invocation_id)
        assert run.store["state"] == snapshot
    assert run.control.model_calls == calls
    assert run.executed == ["sent"]


def _approval_context():
    return SimpleNamespace(
        session_id="approval-test", invocation_id="start", attempt=1, emit=AsyncMock()
    )


@pytest.mark.asyncio
async def test_adapter_refuses_consumed_approval_but_allows_new_turn(approval_run):
    import runtime.adapter as adapter

    run = approval_run
    context = _approval_context()
    await adapter.invoke(
        {
            "messages": [{"role": "user", "content": "Send a message"}],
            "actor": "actor",
            "model": "original-model",
        },
        context,
    )
    writes = run.save_state.await_count
    context.invocation_id = "invalid"
    with pytest.raises(ValueError):
        await adapter.invoke({"actor": "actor", "resume": {"decisions": []}}, context)
    assert run.save_state.await_count == writes
    context.invocation_id = "resume"
    payload = {
        "actor": "actor",
        "resume": {"decisions": [{"type": "approve", "call_id": "call-1"}]},
    }
    completed = await adapter.invoke(payload, context)
    calls, history = run.control.model_calls, await run.session.get_items()
    assert completed["status"] == "completed"
    with pytest.raises(RuntimeError, match="already accepted"):
        await adapter.recover(payload, context)
    assert run.control.model_calls == calls
    assert run.executed == ["sent"]
    assert await run.session.get_items() == history
    context.invocation_id = "duplicate"
    with pytest.raises(RuntimeError):
        await adapter.invoke(payload, context)
    assert run.control.model_calls == calls
    context.invocation_id = "new-turn"
    result = await adapter.invoke(
        {"actor": "actor", "messages": [{"role": "user", "content": "Hi"}]}, context
    )
    assert result["status"] == "completed"
    assert run.executed == ["sent"]


@pytest.mark.asyncio
async def test_initial_pause_storage_failure_does_not_emit_interrupt(approval_run):
    import runtime.adapter as adapter

    run, context = approval_run, _approval_context()
    run.save_state.side_effect = OSError("store unavailable")
    with pytest.raises(OSError, match="store unavailable"):
        await adapter.invoke(
            {"messages": [{"role": "user", "content": "Send"}], "actor": "actor"}, context
        )
    assert not any(call.args[0]["type"] == "interrupt" for call in context.emit.await_args_list)
    assert run.executed == []


@pytest.mark.asyncio
async def test_recovered_resume_reemits_next_pause_without_reapplying_old_decision(approval_run):
    run = approval_run
    run.control.second_approval = True
    await run.run(model="original-model", invocation_id="start")
    decisions = {"decisions": [{"type": "approve", "call_id": "call-1"}]}
    await run.run(resume=decisions, invocation_id="resume")
    recovered = await run.run(resume=decisions, invocation_id="resume")
    assert [item.call_id for item in recovered.interruptions] == ["call-2"]
    assert run.executed == ["sent"]
    assert run.control.model_calls == 2
    with pytest.raises(ValueError, match="mismatched"):
        await run.run(resume=decisions, invocation_id="stale-decision")
    await run.run(
        resume={"decisions": [{"type": "approve", "call_id": "call-2"}]},
        invocation_id="second-resume",
    )
    assert run.executed == ["sent", "sent"]
    assert run.store["state"]["status"] == "consumed"


@pytest.mark.asyncio
@pytest.mark.parametrize("completed", [False, True])
async def test_transport_failure_keeps_uncommitted_continuation_claimed(
    approval_run, monkeypatch, completed
):
    import agent.agent as agent_module

    run = approval_run
    await run.run(model="original-model", invocation_id="start")
    active = _FakeStreamResult([], [], None)
    active.is_complete = completed
    active.final_output = "Done" if completed else None
    active.cancel = MagicMock(side_effect=lambda: setattr(active, "is_complete", True))
    monkeypatch.setattr(agent_module.Runner, "run_streamed", lambda *_args, **_kwargs: active)
    with pytest.raises(OSError, match="transport lost"):
        async with agent_module.run_agent(
            [],
            session_id="approval-test",
            actor="actor",
            load_state=AsyncMock(side_effect=lambda: run.store["state"]),
            save_state=run.save_state,
            invocation_id="resume",
            resume={"decisions": [{"type": "approve", "call_id": "call-1"}]},
        ):
            raise OSError("transport lost")
    assert active.cancel.call_count == (0 if completed else 1)
    assert run.store["state"]["status"] == "consumed"
    assert run.store["state"]["invocation_id"] == "resume"
    for invocation_id in ("resume", "duplicate"):
        with pytest.raises(RuntimeError, match="already accepted"):
            await run.run(
                invocation_id=invocation_id,
                resume={"decisions": [{"type": "approve", "call_id": "call-1"}]},
            )


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
    payload = {"session_id": "ignored", "messages": [{"role": "user", "content": "hi"}]}
    context = SimpleNamespace(
        session_id="runtime-session", emit=AsyncMock(), invocation_id="invocation", attempt=1
    )
    checkpoints = SimpleNamespace(load=AsyncMock(return_value=None), save=AsyncMock())
    monkeypatch.setattr(adapter, "Checkpoints", lambda *_args: checkpoints)

    response = await adapter.invoke(payload, context)
    recovered = await adapter.recover(payload, context)

    kwargs = {
        "session_id": "runtime-session",
        "actor": "runtime-session",
        "model": None,
        "resume": None,
        "load_state": checkpoints.load,
        "save_state": ANY,
        "invocation_id": context.invocation_id,
    }
    assert calls == [
        (payload["messages"], kwargs),
        (
            [{"role": "developer", "content": adapter._RECOVERY_INSTRUCTION}, *payload["messages"]],
            kwargs,
        ),
    ]
    assert "session_id" not in response
    assert "session_id" not in recovered
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
