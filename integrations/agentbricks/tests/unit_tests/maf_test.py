"""Native MAF execution against managed state with only external transports replaced."""

import importlib
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from template_history_fixtures import SessionStoreTransport

pytest.importorskip("agent_framework")

from agent_framework import (
    Agent,
    AgentSession,
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)

from databricks_agentkit.maf import sessions
from databricks_agentkit.maf.sessions import DatabricksHistoryProvider

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"


class ReplyClient(BaseChatClient):
    """Deterministic model transport; Agent and its HistoryProvider execute normally."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.options = []

    def _inner_get_response(self, *, messages, stream=False, options, **kwargs):
        self.calls.append(list(messages))
        self.options.append(dict(options))
        response = ChatResponse(messages=[Message("assistant", ["hello"], message_id="answer")])
        if not stream:

            async def reply():
                return response

            return reply()

        async def updates():
            yield ChatResponseUpdate(
                role="assistant", contents=[Content.from_text("hello")], message_id="answer"
            )

        return ResponseStream(updates(), finalizer=lambda _: response)


@pytest.mark.asyncio
async def test_native_agent_restores_managed_history_after_new_provider(monkeypatch):
    transport = SessionStoreTransport()
    monkeypatch.setattr(sessions, "resolve_session_store", lambda _: "managed")
    first = Agent(
        client=ReplyClient(),
        context_providers=[DatabricksHistoryProvider("alice", client=transport)],
    )
    await first.run("first turn", session=AgentSession(session_id="conversation"))
    second_client = ReplyClient()
    second = Agent(
        client=second_client,
        context_providers=[DatabricksHistoryProvider("alice", client=transport)],
    )
    await second.run("second turn", session=AgentSession(session_id="conversation"))
    assert [message.text for message in second_client.calls[0]] == [
        "first turn",
        "hello",
        "second turn",
    ]
    assert len(transport.items["conversation"]) == 4
    provider = DatabricksHistoryProvider("alice", client=transport)
    native = Message(
        "assistant", [Content.from_function_call(call_id="call", name="sample", arguments={"n": 1})]
    )
    await provider.save_messages("conversation", [native])
    restored = await DatabricksHistoryProvider("alice", client=transport).get_messages(
        "conversation"
    )
    assert restored[-1].to_dict() == native.to_dict()
    with pytest.raises(ValueError, match="different actor"):
        await DatabricksHistoryProvider("bob", client=transport).get_messages("conversation")


@pytest.mark.asyncio
async def test_local_history_is_partitioned_by_actor_and_session(monkeypatch):
    monkeypatch.setattr(sessions, "resolve_session_store", lambda _: None)
    monkeypatch.setattr(sessions, "_local_messages", {})
    provider = DatabricksHistoryProvider("alice")
    await provider.save_messages("conversation", [Message("user", ["private"])])
    assert (await DatabricksHistoryProvider("alice").get_messages("conversation"))[
        0
    ].text == "private"
    assert await DatabricksHistoryProvider("bob").get_messages("conversation") == []
    assert await provider.get_messages("other") == []
    with pytest.raises(ValueError, match="session_id"):
        await provider.get_messages(None)


@pytest.fixture
def template(monkeypatch):
    monkeypatch.syspath_prepend(str(TEMPLATES / "agent-maf"))
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, name)
    agent = importlib.import_module("agent.agent")
    adapter = importlib.import_module("runtime.adapter")
    monkeypatch.setattr(sessions, "resolve_session_store", lambda _: None)
    monkeypatch.setattr(sessions, "_local_messages", {})
    monkeypatch.setattr(agent, "mcp_tools", lambda: [])
    monkeypatch.setattr(agent, "start_trace", lambda *_args, **_kwargs: nullcontext())
    clients = []

    def create_agent(actor, **_kwargs):
        client = ReplyClient()
        clients.append(client)
        return Agent(
            client=client,
            instructions=_kwargs.get("instructions"),
            context_providers=[DatabricksHistoryProvider(actor)],
        )

    monkeypatch.setattr(agent, "create_agent", create_agent)
    yield agent, adapter, clients
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            sys.modules.pop(name, None)


@pytest.mark.asyncio
async def test_template_runs_real_native_stream_through_http_runtime(template):
    from databricks_agentkit import DurableAgentServer

    agent, adapter, clients = template
    app = DurableAgentServer()
    app.invoke(adapter.invoke)
    app.recover(adapter.recover)
    body = {
        "id": str(uuid4()),
        "session_id": "conversation",
        "input": {"messages": [{"role": "user", "content": "first"}]},
    }
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = await client.post("/api/invocations", json=body)
        assert first.status_code == 200, first.text
        assert first.json()["output"]["output"] == [{"role": "assistant", "content": "hello"}]
        body.update(id=str(uuid4()), stream=True)
        body["input"]["messages"][0]["content"] = "second"
        streamed = await client.post("/api/invocations", json=body)
        assert streamed.status_code == 200, streamed.text
        assert '"type":"delta"' in streamed.text.replace(" ", "")
        assert '"type":"message"' in streamed.text.replace(" ", "")
    assert [message.text for message in clients[-1].calls[0]] == ["first", "hello", "second"]
    history = await agent.read_history("conversation")
    assert len(history["session_items"]) == 4


@pytest.mark.asyncio
async def test_recovery_instruction_and_unsupported_approval(template):
    _, adapter, clients = template
    context = SimpleNamespace(session_id="s", emit=AsyncMock())
    payload = {"messages": [{"role": "user", "content": "continue"}]}
    await adapter.recover(payload, context)
    assert "side effects" in clients[0].options[0]["instructions"]
    await adapter.recover(payload, context)
    history = await DatabricksHistoryProvider("s").get_messages("s")
    assert [message.role for message in history] == ["user", "assistant", "user", "assistant"]
    with pytest.raises(ValueError, match="approval continuation"):
        await adapter.invoke({"resume": {"decisions": []}}, context)


@pytest.mark.asyncio
async def test_recovery_replays_a_turn_already_committed_to_history(template):
    """History commit and runtime completion are separate; replay has at-least-once semantics."""
    _, adapter, clients = template

    async def stop_before_completion(event):
        if event["type"] == "message":
            raise RuntimeError("worker stopped after the transcript committed")

    payload = {"messages": [{"role": "user", "content": "original turn"}]}
    context = SimpleNamespace(session_id="conversation", emit=stop_before_completion)
    with pytest.raises(RuntimeError, match="worker stopped"):
        await adapter.invoke(payload, context)
    context.emit = AsyncMock()
    await adapter.recover(payload, context)
    transcript = [
        message for message in clients[1].calls[0] if message.role in {"user", "assistant"}
    ]
    assert [(message.role, message.text) for message in transcript[:2]] == [
        ("user", "original turn"),
        ("assistant", "hello"),
    ]
    assert transcript[2].text == "original turn"


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["system", "developer", "assistant", "tool"])
async def test_public_payload_accepts_only_new_user_messages(template, role):
    _, adapter, clients = template
    context = SimpleNamespace(session_id="s", emit=AsyncMock())
    with pytest.raises(ValueError, match="role='user'"):
        await adapter.invoke({"messages": [{"role": role, "content": "untrusted"}]}, context)
    assert clients == []


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
async def test_browser_config_and_native_history_endpoints(template, monkeypatch, managed):
    from databricks_agentkit import DurableAgentServer

    agent, _, _ = template
    transport = SessionStoreTransport()
    monkeypatch.setattr(sessions, "resolve_session_store", lambda _: "managed" if managed else None)
    monkeypatch.setattr(sessions, "SessionStoreClient", lambda: transport)
    await DatabricksHistoryProvider("agent").save_messages(
        "conversation", [Message("user", ["visible"]), Message("assistant", ["reply"])]
    )
    spec = importlib.util.spec_from_file_location(
        "maf_ui", TEMPLATES / "ui/agent-openai/runtime/ui.py"
    )
    assert spec and spec.loader
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    monkeypatch.setattr(ui, "_session_store", lambda: "managed" if managed else None)
    monkeypatch.setattr(ui, "_memory_store", lambda: None)
    app = DurableAgentServer()
    ui.install_ui(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        config = await client.get("/api/ui/config?session_id=conversation")
        assert config.status_code == 200, config.text
        assert config.json()["models"]["default"] == agent.MODEL
        history = await client.get("/api/demo/session/items?session_id=conversation")
        assert history.status_code == 200, history.text
        assert [item["data"]["content"] for item in history.json()["session_items"]] == [
            "visible",
            "reply",
        ]


def test_tool_messages_keep_native_call_and_result_content():
    from databricks_agentkit.maf.messages import serialize_message

    call = serialize_message(
        Message(
            "assistant",
            [Content.from_function_call(call_id="call", name="sample", arguments='{"n":1}')],
        )
    )
    result = serialize_message(
        Message("tool", [Content.from_function_result(call_id="call", result="finished")])
    )
    assert call["tool_calls"] == [{"id": "call", "name": "sample", "args": {"n": 1}}]
    assert result == {"role": "tool", "content": "finished"}


def test_declared_user_tools_fail_before_client_construction(monkeypatch):
    from databricks_agentkit.maf import mcp
    from databricks_agentkit.runtime.tool_manifest import ToolManifestError, ToolRecord

    monkeypatch.setattr(
        mcp,
        "load_tools",
        lambda **_: (ToolRecord("private", "mcp", service="service", auth="user"),),
    )
    monkeypatch.setattr(
        mcp, "workspace_client", lambda: pytest.fail("auth rejected before client construction")
    )
    with pytest.raises(ToolManifestError, match="only App-authenticated"):
        mcp.mcp_tools()


def test_sandbox_policy_overrides_all_untrusted_metadata():
    from databricks_agentkit.maf.mcp import _ConfiguredMCPTool

    connection = _ConfiguredMCPTool(
        name="sandbox", url="https://example.test/mcp", protected_meta={"policy": "trusted"}
    )
    connection._tool_call_meta_by_name["execute"] = {"policy": "server-controlled"}
    _, meta = connection._prepare_call_kwargs(
        "execute", {"_meta": {"policy": "caller-controlled", "other": "kept"}}
    )
    assert meta == {"policy": "trusted", "other": "kept"}


def test_approval_tools_fail_before_model_configuration(template, monkeypatch):
    from agent_framework import tool

    agent, _, _ = template
    # The fixture replaces only model construction for native HTTP tests; reload the original factory.
    agent = importlib.reload(agent)

    @tool(approval_mode="always_require")
    def side_effect() -> str:
        """A side effect requiring a decision."""
        pytest.fail("approval tool must not execute")

    monkeypatch.setattr(agent, "all_tools", lambda: [side_effect])
    monkeypatch.setattr(agent, "memory_tools", lambda _: [])
    monkeypatch.setattr(agent, "genie_tools", lambda: [])
    monkeypatch.setattr(
        agent, "workspace_client", lambda: pytest.fail("validate before model configuration")
    )
    with pytest.raises(ValueError, match="approval-requiring tools"):
        agent.create_agent("alice")


def test_configuration_resolves_model_binding_for_browser(template, monkeypatch):
    agent, _, _ = template
    monkeypatch.setattr(agent, "resolve_model_service", lambda: "catalog.schema.bound-model")
    monkeypatch.setattr(agent, "workspace_client", lambda: None)
    monkeypatch.setattr(agent, "configure_tracing", lambda: None)
    agent.configure()
    assert agent.MODEL == "catalog.schema.bound-model"
