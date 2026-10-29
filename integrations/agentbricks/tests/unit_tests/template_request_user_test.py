"""Exercise the generated runtime/agent boundary without a live model."""

import importlib
import importlib.util
import json
import re
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from template_fixtures import (
    TEMPLATES,
    durable_app,
    forget_template,
    import_template,
    runtime_client,
    scripted_model,
)


@pytest.fixture(params=["langgraph", "openai"])
def template(request, monkeypatch):
    framework = request.param
    dependencies = (
        ("databricks_langchain", "langgraph", "langchain", "langchain_mcp_adapters")
        if framework == "langgraph"
        else ("agents", "databricks_openai")
    )
    for dependency in dependencies:
        pytest.importorskip(dependency, reason=f"Requires databricks-agentbricks[{framework}]")
    adapter, agent = import_template(monkeypatch, TEMPLATES / f"agent-{framework}")
    yield framework, adapter, agent
    forget_template()


class RequestAuth:
    def __init__(self, owner, requires_user=True):
        self.owner = owner
        self.client_for = Mock()

    def namespace(self, kind, value):
        return f"private:{self.owner}:{kind}:{value}"


def test_documented_memory_payload_keeps_actor_stable_across_conversations(template):
    framework, adapter, _agent = template
    reference = TEMPLATES / f"agent-{framework}" / "AGENTKIT_CONTRACT.md"
    example = re.search(r"```json\n(.*?)\n```", reference.read_text(), re.DOTALL)
    assert example is not None
    invocation = json.loads(example.group(1))
    payload = adapter._payload(invocation["input"])
    for session_id in (invocation["session_id"], "memory-recall"):
        assert adapter._actor(payload, session_id) == invocation["input"]["actor"]
        shorthand = adapter._payload(invocation["input"]["messages"])
        assert adapter._actor(shorthand, session_id) == session_id


@pytest.mark.asyncio
async def test_adapter_uses_context_session_and_namespaces_untrusted_actor(template, monkeypatch):
    framework, adapter, _agent = template
    calls = []

    async def stream(*args, **kwargs):
        calls.append((args, kwargs))
        if False:
            yield

    @asynccontextmanager
    async def result(*args, **kwargs):
        calls.append((args, kwargs))
        yield SimpleNamespace(stream_events=stream, interruptions=[])

    monkeypatch.setattr(adapter, "run_agent", stream if framework == "langgraph" else result)
    payload = {"session_id": "ignored", "actor": "victim", "messages": []}
    for owner in ("alice", "bob"):
        auth = RequestAuth(owner)
        context = SimpleNamespace(
            request_auth=auth,
            session_id=f"private:{owner}:session:public",
            invocation_id="run",
            emit=AsyncMock(),
            attempt=1,
        )
        response = await adapter.invoke(payload, context)
        kwargs = calls[0][1]
        assert kwargs["session_id"] == f"private:{owner}:session:public"
        assert kwargs["actor"] == f"private:{owner}:actor:victim"
        assert kwargs["workspace_client_for"] is auth.client_for
        assert "request_auth" not in kwargs
        assert "session_id" not in response
        assert "private:" not in repr(response)
        calls.clear()


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize("overlay", [False, True])
def test_main_lets_durable_agent_server_infer_auth_policy_after_configure(framework, overlay):
    root = TEMPLATES / "ui" if overlay else TEMPLATES
    source = (root / f"agent-{framework}" / "runtime/main.py").read_text()
    assert "app = DurableAgentServer()" in source
    assert "InvocationAuthPolicy" not in source
    assert source.index("configure()", source.index("load_dotenv(")) < source.index(
        "app = DurableAgentServer"
    )
    assert "\napp.recover(recover)\n" in source
    assert "if not app.auth_policy.requires_user" not in source


@pytest.mark.asyncio
@pytest.mark.parametrize("actor", [None, ""])
async def test_context_session_and_default_actor_are_already_private(template, monkeypatch, actor):
    framework, adapter, _agent = template
    calls = []

    async def stream(*args, **kwargs):
        if kwargs:
            calls.append(kwargs)
        if False:
            yield

    @asynccontextmanager
    async def result(*args, **kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(stream_events=stream, interruptions=[])

    monkeypatch.setattr(adapter, "run_agent", stream if framework == "langgraph" else result)
    context = SimpleNamespace(
        request_auth=RequestAuth("alice"),
        session_id="already-private",
        invocation_id="public-run",
        emit=AsyncMock(),
        attempt=1,
    )
    payload = {"messages": []}
    if actor is not None:
        payload["actor"] = actor
    response = await adapter.invoke(payload, context)
    assert calls[0]["session_id"] == "already-private"
    assert calls[0]["actor"] == "already-private"
    assert "session_id" not in response


@pytest.mark.asyncio
async def test_nested_session_does_not_replace_missing_context_session(template, monkeypatch):
    _framework, adapter, _agent = template
    run = Mock(side_effect=AssertionError("must reject before running"))
    monkeypatch.setattr(adapter, "run_agent", run)
    context = SimpleNamespace(
        request_auth=None,
        session_id=None,
        invocation_id="run",
        emit=AsyncMock(),
    )

    with pytest.raises(ValueError, match="top-level invocation field"):
        await adapter.invoke({"session_id": "legacy", "messages": []}, context)

    run.assert_not_called()


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize("user_auth", [False, True])
@pytest.mark.asyncio
async def test_ui_runtime_capabilities_ignore_tool_auth_policy(framework, user_auth, monkeypatch):
    import httpx
    from fastapi import FastAPI

    path = TEMPLATES / "ui" / f"agent-{framework}" / "runtime/ui.py"
    spec = importlib.util.spec_from_file_location(f"ui_{framework}", path)
    assert spec is not None and spec.loader is not None
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    monkeypatch.setattr(ui, "_default_model", lambda: "model")
    monkeypatch.setattr(ui, "_memory_store", lambda: "memory")
    monkeypatch.setattr(ui, "_session_store", lambda: "sessions")
    monkeypatch.setattr(ui, "runtime_store_is_persistent_environment", lambda: True)
    app = FastAPI()
    monkeypatch.setattr(app, "auth_policy", SimpleNamespace(requires_user=user_auth), raising=False)
    ui.install_ui(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        config = (await client.get("/api/ui/config?session_id=public")).json()
    assert config["background"] == {
        "enabled": True,
        "persistent": True,
        "mode": "Runtime Store",
    }
    assert config["streaming"]["enabled"] is True
    assert config["streaming"]["persistent"] is True
    assert config["streaming"]["mode"] == "Runtime Store"
    assert config["session"]["history"] is True
    assert config["memory"]["enabled"] is True


@pytest.mark.asyncio
async def test_openai_request_exit_cancels_native_run(template, monkeypatch):
    framework, _adapter, agent = template
    if framework != "openai":
        pytest.skip("OpenAI native streaming task")

    async def events():
        if False:
            yield

    monkeypatch.setattr(agent, "mcp_servers", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent, "create_agent", Mock(return_value=object()))
    monkeypatch.setattr(agent, "session_store", Mock(return_value=None))
    monkeypatch.setattr(
        agent, "start_trace", Mock(return_value=__import__("contextlib").nullcontext())
    )
    result = SimpleNamespace(is_complete=False, cancel=Mock(), stream_events=events)
    monkeypatch.setattr(agent.Runner, "run_streamed", Mock(return_value=result))
    with pytest.raises(RuntimeError, match="disconnected"):
        async with agent.run_agent([], session_id="private", workspace_client_for=Mock()):
            raise RuntimeError("disconnected")
    result.cancel.assert_called_once_with()


@pytest.fixture
def hitl_app(template, monkeypatch):
    """Real HTTP runtime, graph and middleware; only model/tool I/O is local."""
    from contextlib import nullcontext

    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    from databricks_agentkit.runtime.auth import InvocationAuthPolicy
    from databricks_agentkit.runtime.store import InMemoryRuntimeStore

    _framework, adapter, agent = template
    monkeypatch.setenv("DATABRICKS_APP_NAME", "hitl-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.delenv("DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL", raising=False)
    # Keep SDK discovery offline while exercising the real request-bound credential resolver.
    monkeypatch.setattr(
        "databricks.sdk.WorkspaceClient",
        lambda **kwargs: SimpleNamespace(config=SimpleNamespace(token=kwargs["token"])),
    )
    saver = InMemorySaver()
    writes = []
    credentials = []

    async def create_graph(actor, model=None, *, workspace_client_for=None):
        assert workspace_client_for is not None
        credentials.append(workspace_client_for.__self__)

        @tool
        async def record_action(text: str) -> str:
            """Record the approved action locally."""
            client = workspace_client_for("user")
            writes.append((text, client.config.token))
            return "Action recorded"

        return create_agent(
            scripted_model(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "record_action",
                            "args": {"text": "original"},
                            "id": "action-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Review handled"),
            ),
            [record_action],
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"record_action": True})],
            checkpointer=saver,
        )

    monkeypatch.setattr(agent, "create_agent_graph", create_graph)
    monkeypatch.setattr(agent, "start_trace", lambda **kwargs: nullcontext())
    store = InMemoryRuntimeStore()
    app = durable_app(
        adapter, runtime_store=store, auth_policy=InvocationAuthPolicy(user_required=True)
    )
    return SimpleNamespace(
        app=app, store=store, saver=saver, writes=writes, credentials=credentials
    )


def hitl_headers(token, user="alice"):
    return {"x-forwarded-user": user, "x-forwarded-access-token": token}


def hitl_request(value, *, stream=False):
    from uuid import uuid4

    return {"id": str(uuid4()), "session_id": "review-session", "input": value, "stream": stream}


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["langgraph"], indirect=True)
@pytest.mark.parametrize("decision", ["approve", "edit", "reject"])
@pytest.mark.parametrize("stream", [False, True])
async def test_langgraph_user_hitl_resumes_with_current_request_credentials(
    hitl_app, decision, stream
):
    from databricks_agentkit.runtime.auth import AuthError

    state = hitl_app
    async with runtime_client(state.app) as client:
        pause = await client.post(
            "/api/invocations",
            json=hitl_request(
                {"messages": [{"role": "user", "content": "Record an action"}]}, stream=stream
            ),
            headers=hitl_headers("before-approval-token"),
        )
        assert pause.status_code == 200
        if stream:
            assert "event: interrupt" in pause.text
            assert "event: run.failed" not in pause.text
        else:
            assert pause.json()["output"]["status"] == "interrupted"
        assert state.writes == []
        with pytest.raises(AuthError, match="no longer active"):
            state.credentials[0].client_for("user")

        approval = {"type": decision}
        if decision == "edit":
            approval["edited_action"] = {"name": "record_action", "args": {"text": "edited"}}
        body = hitl_request({"resume": {"decisions": [approval]}}, stream=stream)
        resume = await client.post(
            "/api/invocations", json=body, headers=hitl_headers("after-approval-token")
        )
        assert resume.status_code == 200
        if stream:
            assert "event: run.completed" in resume.text
            assert "event: run.failed" not in resume.text
        else:
            assert resume.json()["output"]["status"] == "completed"
        replay = await client.post(
            "/api/invocations", json=body, headers=hitl_headers("replay-token")
        )
        assert replay.status_code == 200
        assert replay.text == resume.text

    expected = (
        []
        if decision == "reject"
        else [("edited" if decision == "edit" else "original", "after-approval-token")]
    )
    assert state.writes == expected
    assert len(state.credentials) == 2
    for auth in state.credentials:
        with pytest.raises(AuthError, match="no longer active"):
            auth.client_for("user")
    persisted = repr(
        (state.store.states, state.store.persisted_events, state.saver.storage, state.saver.writes)
    )
    for token in ("before-approval-token", "after-approval-token", "replay-token"):
        assert token not in persisted


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["langgraph"], indirect=True)
async def test_langgraph_user_hitl_requires_authenticated_session_owner(hitl_app):
    state = hitl_app
    async with runtime_client(state.app) as client:
        pause = await client.post(
            "/api/invocations",
            json=hitl_request({"messages": [{"role": "user", "content": "Record an action"}]}),
            headers=hitl_headers("owner-token"),
        )
        assert pause.json()["output"]["status"] == "interrupted"
        approval = {"resume": {"decisions": [{"type": "approve"}]}}
        missing = await client.post("/api/invocations", json=hitl_request(approval))
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "MCP_USER_AUTH_REQUIRED"
        other = await client.post(
            "/api/invocations",
            json=hitl_request(approval),
            headers=hitl_headers("other-token", "bob"),
        )
        assert state.writes == []
        assert "Action recorded" not in other.text
        owner = await client.post(
            "/api/invocations",
            json=hitl_request(approval),
            headers=hitl_headers("owner-resume-token"),
        )
        assert owner.status_code == 200
        assert state.writes == [("original", "owner-resume-token")]


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["langgraph"], indirect=True)
@pytest.mark.parametrize(
    "decisions",
    [
        [],
        [{"type": "invalid"}],
        [{"type": "edit"}],
        [{"type": "respond"}],
        [{"type": "approve"}, {"type": "approve"}],
    ],
)
async def test_langgraph_user_hitl_invalid_decision_does_not_execute_or_consume_approval(
    hitl_app, decisions
):
    state = hitl_app
    async with runtime_client(state.app) as client:
        pause = await client.post(
            "/api/invocations",
            json=hitl_request({"messages": [{"role": "user", "content": "Record an action"}]}),
            headers=hitl_headers("pause-token"),
        )
        assert pause.json()["output"]["status"] == "interrupted"
        invalid = await client.post(
            "/api/invocations",
            json=hitl_request({"resume": {"decisions": decisions}}),
            headers=hitl_headers("invalid-decision-token"),
        )
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "INVALID_APPROVAL"
        assert state.writes == []
        valid = await client.post(
            "/api/invocations",
            json=hitl_request({"resume": {"decisions": [{"type": "approve"}]}}),
            headers=hitl_headers("approved-token"),
        )
        assert valid.status_code == 200
    assert state.writes == [("original", "approved-token")]


@pytest.fixture(params=["memory-function", "managed-function", "memory-mcp", "managed-mcp"])
def openai_hitl_app(template, request, monkeypatch):
    """Real SDK serialization, tool dispatch and checkpoint adapters with local I/O."""
    from contextlib import nullcontext

    from agents import Agent, Model, RunConfig, function_tool
    from agents.mcp import MCPServer
    from mcp.types import CallToolResult, TextContent, Tool
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )
    from template_history_fixtures import SessionStoreTransport

    from databricks_agentkit.openai import sessions
    from databricks_agentkit.runtime.auth import InvocationAuthPolicy
    from databricks_agentkit.runtime.store import InMemoryRuntimeStore

    _framework, adapter, agent = template
    checkpoints = importlib.import_module("agent.checkpoints")
    mode, tool_kind = request.param.split("-")
    transport = SessionStoreTransport()
    store_name = "hitl" if mode == "managed" else None
    monkeypatch.setattr(checkpoints, "resolve_session_store", lambda: store_name)
    monkeypatch.setattr(checkpoints, "SessionStoreClient", lambda: transport)
    monkeypatch.setattr(
        "databricks_agentkit.runtime.tool_manifest.resolve_session_store", lambda *_args: store_name
    )
    monkeypatch.setattr(sessions, "SessionStoreClient", lambda *_args: transport)
    monkeypatch.setattr(sessions, "_local_sessions", {})
    monkeypatch.setenv("DATABRICKS_APP_NAME", "hitl-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.delenv("DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL", raising=False)
    monkeypatch.setattr(
        "databricks.sdk.WorkspaceClient",
        lambda **kwargs: SimpleNamespace(config=SimpleNamespace(token=kwargs["token"])),
    )
    writes, credentials, servers = [], [], []

    class ApprovalModel(Model):
        async def get_response(self, *_args, **_kwargs):
            raise AssertionError("Expected native streaming")

        async def stream_response(self, system_instructions, input, *_args, **_kwargs):
            if any(item.get("type") == "function_call_output" for item in input):
                output = ResponseOutputMessage(
                    id="answer",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(
                            text="Review handled", type="output_text", annotations=[]
                        )
                    ],
                )
            else:
                output = ResponseFunctionToolCall(
                    id="action",
                    type="function_call",
                    call_id="action-call",
                    name="record_action",
                    arguments='{"text":"original"}',
                )
            yield ResponseCompletedEvent(
                type="response.completed",
                sequence_number=0,
                response=Response(
                    id="reply",
                    created_at=0,
                    model="test",
                    object="response",
                    output=[output],
                    parallel_tool_calls=False,
                    tool_choice="auto",
                    tools=[],
                ),
            )

    class UserMCP(MCPServer):
        tool_filter = None
        cache_tools_list = True
        _agentbricks_request_user = True

        def __init__(self, resolver, require_approval):
            super().__init__(require_approval=require_approval)
            self.resolver = resolver
            self.closed = False

        @property
        def name(self):
            return "request-user-tools"

        async def connect(self):
            pass

        async def cleanup(self):
            self.closed = True

        async def list_tools(self, *_args, **_kwargs):
            return [
                Tool(
                    name="record_action",
                    description="Record an approved action",
                    inputSchema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                )
            ]

        async def call_tool(self, tool_name, arguments, meta=None):
            assert not self.closed and tool_name == "record_action"
            writes.append((arguments["text"], self.resolver("user").config.token))
            return CallToolResult(content=[TextContent(type="text", text="Action recorded")])

        async def list_prompts(self):
            raise NotImplementedError

        async def get_prompt(self, name, arguments=None):
            raise NotImplementedError

    async def mcp_servers(_configured, *, workspace_client_for, require_approval):
        assert require_approval == {"record_action": True}
        if tool_kind == "function":
            return []
        server = UserMCP(workspace_client_for, require_approval)
        servers.append(server)
        return [server]

    def create_agent(actor, mcp, model=None, *, workspace_client_for):
        credentials.append(workspace_client_for.__self__)

        @function_tool(needs_approval=True)
        async def record_action(text: str) -> str:
            """Record an approved action locally."""
            writes.append((text, workspace_client_for("user").config.token))
            return "Action recorded"

        return Agent(
            name="Agent",
            model=ApprovalModel(),
            mcp_servers=mcp,
            tools=[record_action] if tool_kind == "function" else [],
        )

    monkeypatch.setattr(agent, "create_agent", create_agent)
    monkeypatch.setattr(agent, "mcp_servers", mcp_servers)
    monkeypatch.setattr(agent, "REQUIRE_APPROVAL", {"record_action"})
    monkeypatch.setattr(agent, "start_trace", lambda **_kwargs: nullcontext())
    run_streamed = agent.Runner.run_streamed
    monkeypatch.setattr(
        agent.Runner,
        "run_streamed",
        lambda *args, **kwargs: run_streamed(
            *args, **kwargs, run_config=RunConfig(tracing_disabled=True)
        ),
    )
    runtime = InMemoryRuntimeStore()
    app = durable_app(
        adapter, runtime_store=runtime, auth_policy=InvocationAuthPolicy(user_required=True)
    )
    yield SimpleNamespace(
        app=app,
        store=runtime,
        agent=agent,
        checkpoints=checkpoints,
        transport=transport,
        writes=writes,
        credentials=credentials,
        servers=servers,
        mode=mode,
    )
    for session in sessions._local_sessions.values():
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["openai"], indirect=True)
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("stream", [False, True])
async def test_openai_user_hitl_resumes_with_current_request_credentials(
    openai_hitl_app, decision, stream
):
    from databricks_agentkit.runtime.auth import AuthError

    state = openai_hitl_app
    async with runtime_client(state.app) as client:
        pause = await client.post(
            "/api/invocations",
            headers=hitl_headers("before-approval-token"),
            json=hitl_request(
                {"messages": [{"role": "user", "content": "Record an action"}]}, stream=stream
            ),
        )
        assert pause.status_code == 200
        if stream:
            assert "event: interrupt" in pause.text and "event: run.failed" not in pause.text
        else:
            assert pause.json()["output"]["status"] == "interrupted"
        assert state.writes == []
        assert all(server.closed for server in state.servers)
        with pytest.raises(AuthError, match="no longer active"):
            state.credentials[0].client_for("user")
        # Only JSON snapshots are retained; there is no live RunState in the agent cache.
        assert state.agent._pending_runs == {}
        persisted = (
            state.transport.items if state.mode == "managed" else state.checkpoints._local_items
        )
        encoded = json.dumps(persisted)
        assert "run_state" in encoded and "before-approval-token" not in encoded
        if state.mode == "managed":
            # Clear process-local caches; the next invocation must load the managed snapshot.
            state.checkpoints._local_items.clear()
        body = hitl_request(
            {"resume": {"decisions": [{"type": decision, "call_id": "action-call"}]}}, stream=stream
        )
        resume = await client.post(
            "/api/invocations", json=body, headers=hitl_headers("after-approval-token")
        )
        assert resume.status_code == 200
        if stream:
            assert "event: run.completed" in resume.text and "event: run.failed" not in resume.text
        else:
            assert resume.json()["output"]["status"] == "completed"
        replay = await client.post(
            "/api/invocations", json=body, headers=hitl_headers("replay-token")
        )
        assert replay.status_code == 200 and replay.text == resume.text
        assert len(state.credentials) == 2
        consumed = await client.post(
            "/api/invocations",
            json=hitl_request(body["input"]),
            headers=hitl_headers("consumed-token"),
        )
        assert consumed.status_code == 500  # Existing consumed-approval contract.
    assert state.writes == ([] if decision == "reject" else [("original", "after-approval-token")])
    assert all(server.closed for server in state.servers)
    for auth in state.credentials:
        with pytest.raises(AuthError, match="no longer active"):
            auth.client_for("user")
    persisted = repr(
        (
            state.store.states,
            state.store.persisted_events,
            state.checkpoints._local_items,
            state.transport.items,
        )
    )
    for token in (
        "before-approval-token",
        "after-approval-token",
        "replay-token",
        "consumed-token",
    ):
        assert token not in persisted


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["openai"], indirect=True)
async def test_openai_user_hitl_requires_authenticated_session_owner(openai_hitl_app):
    state = openai_hitl_app
    async with runtime_client(state.app) as client:
        pause = await client.post(
            "/api/invocations",
            headers=hitl_headers("owner-token"),
            json=hitl_request({"messages": [{"role": "user", "content": "Record an action"}]}),
        )
        assert pause.json()["output"]["status"] == "interrupted"
        approval = {"resume": {"decisions": [{"type": "approve", "call_id": "action-call"}]}}
        missing = await client.post("/api/invocations", json=hitl_request(approval))
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "MCP_USER_AUTH_REQUIRED"
        other = await client.post(
            "/api/invocations",
            json=hitl_request(approval),
            headers=hitl_headers("other-token", "bob"),
        )
        assert other.status_code == 500 and "Action recorded" not in other.text
        assert state.writes == []
        owner = await client.post(
            "/api/invocations",
            json=hitl_request(approval),
            headers=hitl_headers("owner-resume-token"),
        )
        assert owner.status_code == 200
        assert state.writes == [("original", "owner-resume-token")]


@pytest.mark.asyncio
@pytest.mark.parametrize("template", ["openai"], indirect=True)
@pytest.mark.parametrize(
    "decisions",
    [
        [],
        [{"type": "invalid", "call_id": "action-call"}],
        [{"type": "approve", "call_id": "stale"}],
        [{"type": "approve"}],
        [{"type": "edit", "call_id": "action-call"}],
        [{"type": "reject", "call_id": "action-call", "message": 5}],
    ],
)
async def test_openai_user_hitl_invalid_decision_keeps_approval(openai_hitl_app, decisions):
    state = openai_hitl_app
    async with runtime_client(state.app) as client:
        pause = await client.post(
            "/api/invocations",
            headers=hitl_headers("pause-token"),
            json=hitl_request({"messages": [{"role": "user", "content": "Record an action"}]}),
        )
        assert pause.json()["output"]["status"] == "interrupted"
        before = json.dumps((state.checkpoints._local_items, state.transport.items), sort_keys=True)
        invalid = await client.post(
            "/api/invocations",
            headers=hitl_headers("invalid-decision-token"),
            json=hitl_request({"resume": {"decisions": decisions}}),
        )
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "INVALID_APPROVAL"
        assert state.writes == []
        assert (
            json.dumps((state.checkpoints._local_items, state.transport.items), sort_keys=True)
            == before
        )
        valid = await client.post(
            "/api/invocations",
            headers=hitl_headers("approved-token"),
            json=hitl_request(
                {"resume": {"decisions": [{"type": "approve", "call_id": "action-call"}]}}
            ),
        )
        assert valid.status_code == 200
        assert state.writes == [("original", "approved-token")]
