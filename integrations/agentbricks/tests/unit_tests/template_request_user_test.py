"""Exercise the generated runtime/agent boundary without a live model."""

import importlib
import importlib.util
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"


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
    monkeypatch.syspath_prepend(str(TEMPLATES / f"agent-{framework}"))
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, name)
    adapter = importlib.import_module("runtime.adapter")
    agent = importlib.import_module("agent.agent")
    yield framework, adapter, agent
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            sys.modules.pop(name, None)


class RequestAuth:
    def __init__(self, owner, requires_user=True):
        self.owner = owner
        self.client_for = Mock()

    def namespace(self, kind, value):
        return f"private:{self.owner}:{kind}:{value}"


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [None, "off"])
async def test_template_prompt_caching_default_and_opt_out(template, monkeypatch, policy):
    framework, _adapter, agent = template
    if policy is None:
        monkeypatch.delenv("DATABRICKS_CLAUDE_PROMPT_CACHING", raising=False)
    else:
        monkeypatch.setenv("DATABRICKS_CLAUDE_PROMPT_CACHING", policy)
    monkeypatch.setattr(agent, "workspace_client", Mock())
    if framework == "langgraph":
        model = Mock()
        monkeypatch.setattr(agent, "_RoutedChatDatabricks", model)
        monkeypatch.setattr(agent, "mcp_tools", AsyncMock(return_value=[]))
        monkeypatch.setattr(agent, "memory_tools", lambda *_args: [])
        monkeypatch.setattr(agent, "genie_tools", lambda **_kwargs: [])
        monkeypatch.setattr(agent, "checkpointer", Mock())
        monkeypatch.setattr(agent, "create_agent", Mock())
        await agent.create_agent_graph("actor")
    else:
        import agents

        model = Mock()
        monkeypatch.setattr(agent, "ClaudeCachingOpenAI", model)
        monkeypatch.setattr(agent, "_check_databricks_auth", Mock())
        monkeypatch.setattr(agent, "configure_tracing", Mock())
        monkeypatch.setattr(agents, "set_default_openai_client", Mock())
        monkeypatch.setattr(agents, "set_default_openai_api", Mock())
        agent.configure()
    assert "prompt_caching" not in model.call_args.kwargs
    assert model.call_args.kwargs["use_ai_gateway"] is True
    if framework == "openai":
        assert model.call_args.kwargs["cache_policy"] == (policy or "auto")


def _load_custom_agent(framework, monkeypatch):
    dependency = "databricks_langchain" if framework == "langgraph" else "databricks_openai"
    pytest.importorskip(dependency, reason=f"Requires databricks-agentbricks[{framework}]")
    path = TEMPLATES / f"custom-agent-{framework}" / "agent/agent.py"
    spec = importlib.util.spec_from_file_location(f"custom_agent_{framework}", path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, agent)
    spec.loader.exec_module(agent)
    return agent


@pytest.mark.parametrize("policy", [None, "off"])
@pytest.mark.parametrize("headers", [{}, {"X-Test-Routing": "request"}])
def test_custom_openai_prompt_caching_with_and_without_headers(monkeypatch, policy, headers):
    agent = _load_custom_agent("openai", monkeypatch)
    if policy is None:
        monkeypatch.delenv("DATABRICKS_CLAUDE_PROMPT_CACHING", raising=False)
    else:
        monkeypatch.setenv("DATABRICKS_CLAUDE_PROMPT_CACHING", policy)
    workspace = Mock()
    monkeypatch.setattr(agent, "workspace_client", Mock(return_value=workspace))
    monkeypatch.setattr(agent, "workspace_headers", Mock(return_value=headers))
    client = Mock()
    monkeypatch.setattr(agent, "ClaudeCachingOpenAI", client)
    configure = Mock()
    monkeypatch.setattr(agent, "set_default_openai_client", configure)
    monkeypatch.setattr(agent, "set_default_openai_api", Mock())
    agent._configure_client()
    assert "prompt_caching" not in client.call_args.kwargs
    assert client.call_args.kwargs["workspace_client"] is workspace
    assert client.call_args.kwargs.get("default_headers", {}) == headers
    configure.assert_called_once_with(client.return_value)
    assert client.call_args.kwargs["cache_policy"] == (policy or "auto")
    agent.set_default_openai_api.assert_called_once_with("chat_completions")


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [None, "off"])
async def test_custom_langgraph_prompt_caching_default_and_opt_out(monkeypatch, policy):
    agent = _load_custom_agent("langgraph", monkeypatch)
    if policy is None:
        monkeypatch.delenv("DATABRICKS_CLAUDE_PROMPT_CACHING", raising=False)
    else:
        monkeypatch.setenv("DATABRICKS_CLAUDE_PROMPT_CACHING", policy)
    workspace = Mock()
    monkeypatch.setattr(agent, "workspace_client", Mock(return_value=workspace))
    model = Mock()
    monkeypatch.setattr(agent, "_RoutedChatDatabricks", model)
    output = {"role": "assistant", "content": "OK"}
    graph = Mock(ainvoke=AsyncMock(return_value={"messages": [output]}))
    monkeypatch.setattr(agent, "create_agent", Mock(return_value=graph))
    messages = [{"role": "user", "content": "Hello"}]
    assert await agent.invoke(messages) == {"output": [output]}
    assert "prompt_caching" not in model.call_args.kwargs
    assert model.call_args.kwargs["workspace_client"] is workspace
    graph.ainvoke.assert_awaited_once_with({"messages": messages})


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


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["resume", "approval", "approvals"])
async def test_user_hitl_input_rejected_before_framework_execution(template, monkeypatch, key):
    _framework, adapter, _agent = template
    from databricks_agentkit.runtime.auth import AuthError

    run = Mock(side_effect=AssertionError("must reject before running"))
    monkeypatch.setattr(adapter, "run_agent", run)
    context = SimpleNamespace(
        request_auth=RequestAuth("alice"), session_id="public", emit=AsyncMock()
    )
    with pytest.raises(AuthError) as raised:
        await adapter.invoke({key: {}}, context)
    assert raised.value.code == "MCP_USER_AUTH_HITL_UNSUPPORTED"
    run.assert_not_called()


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
async def test_openai_does_not_retain_request_bound_run_state(template, monkeypatch):
    framework, _adapter, agent = template
    if framework != "openai":
        pytest.skip("OpenAI RunState custody")
    from databricks_agentkit.runtime.auth import AuthError

    monkeypatch.setattr(agent, "mcp_servers", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent, "create_agent", Mock(return_value=object()))
    monkeypatch.setattr(agent, "session_store", Mock(return_value=None))
    monkeypatch.setattr(
        agent, "start_trace", Mock(return_value=__import__("contextlib").nullcontext())
    )
    result = SimpleNamespace(
        interruptions=[object()], to_state=Mock(), final_output=None, is_complete=True
    )
    monkeypatch.setattr(agent.Runner, "run_streamed", Mock(return_value=result))
    with pytest.raises(AuthError) as raised:
        async with agent.run_agent([], session_id="private", workspace_client_for=Mock()):
            pass
    assert raised.value.code == "MCP_USER_AUTH_HITL_UNSUPPORTED"
    result.to_state.assert_not_called()
    assert "private" not in agent._pending_runs
    assert "context" not in agent.Runner.run_streamed.call_args.kwargs


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


@pytest.mark.parametrize("policy", ["auto", "off"])
@pytest.mark.parametrize(
    "model", ["system.ai.claude-sonnet-4-6", "system.ai.claude-sonnet-4-5", "system.ai.gpt-5-2"]
)
@pytest.mark.parametrize("custom", [False, True])
def test_generated_langgraph_prepares_claude_controls(template, monkeypatch, policy, model, custom):
    framework, _, agent = template
    if framework != "langgraph":
        pytest.skip("LangGraph preparation only")
    if custom:
        agent = _load_custom_agent("langgraph", monkeypatch)
    from unittest.mock import MagicMock

    from databricks.sdk import WorkspaceClient
    from langchain_core.messages import HumanMessage, SystemMessage

    workspace = MagicMock(spec=WorkspaceClient)
    workspace.config.host = "https://test.databricks.com"
    monkeypatch.setenv("DATABRICKS_CLAUDE_PROMPT_CACHING", policy)
    llm = agent._RoutedChatDatabricks(model=model, workspace_client=workspace, use_ai_gateway=True)
    request = llm._prepare_inputs([SystemMessage(content="Stable"), HumanMessage(content="Hello")])
    if policy == "auto" and "claude" in model:
        assert request["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert ("cache_control" in request.get("extra_body", {})) == model.endswith("4-6")
    else:
        assert request["messages"][0]["content"] == "Stable"
        assert "cache_control" not in request.get("extra_body", {})
