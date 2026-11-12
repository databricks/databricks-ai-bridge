"""First-party adapters translate native events and persist between worker lifetimes."""

import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from databricks_agentkit.sandbox.adapters.common import (
    app_workspace_client,
    configure_project,
    shutdown_session,
    text_content,
    trace_result,
    workspace_client_for,
)


@pytest.fixture(autouse=True)
def isolate_catalog_setting(monkeypatch):
    # Project configuration sets a process-wide worker default.
    monkeypatch.setenv("MLFLOW_MODEL_CATALOG_URI", os.getenv("MLFLOW_MODEL_CATALOG_URI", ""))


def project_modules(monkeypatch, **attributes):
    agent = ModuleType("agent")
    agent.__path__ = []
    module = ModuleType("agent.agent")
    module.MODEL = "default-model"
    module.configure = lambda: None
    for name, value in attributes.items():
        setattr(module, name, value)
    mcps = ModuleType("agent.mcps")
    mcps.build_mcp_servers = lambda: []
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.agent", module)
    monkeypatch.setitem(sys.modules, "agent.mcps", mcps)


@pytest.mark.asyncio
async def test_openai_translates_only_output_text_and_reopens_sqlite_history(tmp_path, monkeypatch):
    agents = pytest.importorskip("agents")
    pytest.importorskip("databricks_openai")
    from agents.stream_events import RawResponsesStreamEvent
    from openai.types.responses import ResponseTextDeltaEvent

    from databricks_agentkit.sandbox.adapters.openai import invoke

    histories = []
    configure = Mock()
    project_modules(monkeypatch, create_agent=lambda *args, **kwargs: object(), configure=configure)
    monkeypatch.setattr("databricks_agentkit.openai.mcp_servers", AsyncMock(return_value=[]))

    class Result:
        final_output = "Hello 🌍"
        interruptions = []
        new_items = []

        def __init__(self, message, session):
            self.message, self.session = message, session

        async def stream_events(self):
            histories.append(await self.session.get_items())
            await self.session.add_items(
                [
                    {"role": "user", "content": self.message},
                    {"role": "assistant", "content": self.final_output},
                ]
            )
            for kind, delta in [
                ("response.reasoning_text.delta", "private"),
                ("response.output_text.delta", "Hello "),
                ("response.output_text.delta", "🌍"),
            ]:
                if kind == "response.output_text.delta":
                    yield RawResponsesStreamEvent(
                        data=ResponseTextDeltaEvent(
                            type=kind,
                            delta=delta,
                            content_index=0,
                            output_index=0,
                            item_id="message",
                            sequence_number=1,
                            logprobs=[],
                        )
                    )
                else:
                    yield SimpleNamespace(
                        type="raw_response_event", data=SimpleNamespace(type=kind, delta=delta)
                    )

    monkeypatch.setattr(
        agents.Runner,
        "run_streamed",
        lambda agent, message, *, session, max_turns: Result(message, session),
    )
    chunks = []
    for message in ["first turn", "second turn"]:
        result = await invoke(
            {"session_id": "s", "input": {"message": message, "model": "test-model"}},
            tmp_path,
            emit=chunks.append,
        )
        assert result["reply"] == "Hello 🌍"
        assert result["model"] == "test-model"
    assert chunks == ["Hello ", "🌍"] * 2
    assert histories[0] == []
    assert histories[1][0] == {"role": "user", "content": "first turn"}
    assert (tmp_path / "chat.sqlite").exists()
    configure.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
async def test_langgraph_streams_native_chunks_and_reopens_sqlite_checkpoints(
    tmp_path, monkeypatch, managed
):
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.graph import END, START, MessagesState, StateGraph

    from databricks_agentkit.sandbox.adapters.langgraph import invoke

    histories = []
    if managed:
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        monkeypatch.setenv("AGENT_SESSION_STORE", "bound-store")

        def bound_checkpointer(store):
            assert store == "bound-store"
            return saver

        monkeypatch.setattr("databricks_agentkit.langgraph.checkpointer", bound_checkpointer)
    else:
        monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)

    async def create_graph(actor, *, model, workspace_client_for, checkpointer_override):
        assert model == "test-model"
        # A real native graph/model; no network, harness event shapes stay native.
        llm = FakeListChatModel(responses=["Hello 🌍"])

        async def call(state):
            histories.append([message.content for message in state["messages"]])
            return {"messages": [await llm.ainvoke(state["messages"])]}

        builder = StateGraph(MessagesState)
        builder.add_node("agent", call)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        return builder.compile(checkpointer=checkpointer_override)

    configure = Mock()
    project_modules(monkeypatch, create_agent_graph=create_graph, configure=configure)
    for turn in ["first turn", "second turn"]:
        chunks = []
        result = await invoke(
            {"session_id": "s", "input": {"message": turn, "model": "test-model"}},
            tmp_path,
            emit=chunks.append,
        )
        assert result["reply"] == "Hello 🌍"
        assert "".join(chunks) == result["reply"]
        assert not result["approval_required"]
        assert result["session_backend"]["mode"] == ("managed" if managed else "sandbox-sqlite")
    assert histories == [["first turn"], ["first turn", "Hello 🌍", "second turn"]]
    configure.assert_called_once()
    # Same sandbox, another session directory: no history leakage.
    await invoke(
        {"session_id": "other", "input": {"message": "isolated", "model": "test-model"}},
        tmp_path / "other",
    )
    assert histories[-1] == ["isolated"]
    assert (tmp_path / "chat.sqlite").exists() is not managed


def test_app_auth_does_not_silently_substitute_for_request_user():
    with pytest.raises(RuntimeError, match="supports app-auth tools only"):
        workspace_client_for("user")


def test_app_client_is_reused_only_for_app_auth(monkeypatch):
    app_workspace_client.cache_clear()
    factory = Mock(return_value=object())
    monkeypatch.setattr("databricks_agentkit.workspace_client", factory)
    try:
        first = workspace_client_for("app")
        assert workspace_client_for("app") is first
        with pytest.raises(RuntimeError, match="supports app-auth tools only"):
            workspace_client_for("user")
        factory.assert_called_once()
    finally:
        app_workspace_client.cache_clear()


def test_failed_startup_configuration_is_not_cached():
    project = ModuleType("agent.project")
    project.configure = Mock(side_effect=[RuntimeError("not ready"), None])
    with pytest.raises(RuntimeError, match="not ready"):
        configure_project(project)
    configure_project(project)
    configure_project(project)
    assert project.configure.call_count == 2


def test_default_trace_cost_uses_bundled_catalog_without_network(monkeypatch):
    from mlflow.utils import providers

    monkeypatch.delenv("MLFLOW_MODEL_CATALOG_URI", raising=False)
    fetch = Mock(side_effect=AssertionError("Span completion must not fetch prices"))
    monkeypatch.setattr(providers, "cloud_storage_http_request", fetch)
    project = ModuleType("agent.catalog")
    project.configure = lambda: None
    configure_project(project)
    cost = providers.cost_per_token(
        "unknown-model", prompt_tokens=100, completion_tokens=20, custom_llm_provider="openai"
    )
    assert cost is None  # Missing prices never trigger a remote fallback.
    fetch.assert_not_called()


@pytest.mark.parametrize("configured_by", ["environment", "project"])
def test_explicit_trace_catalog_is_preserved(monkeypatch, configured_by):
    uri = "file:///opt/company-prices"
    monkeypatch.delenv("MLFLOW_MODEL_CATALOG_URI", raising=False)
    project = ModuleType("agent.catalog")
    project.configure = lambda: None
    if configured_by == "environment":
        monkeypatch.setenv("MLFLOW_MODEL_CATALOG_URI", uri)
    else:
        project.configure = lambda: monkeypatch.setenv("MLFLOW_MODEL_CATALOG_URI", uri)
    configure_project(project)
    assert os.environ["MLFLOW_MODEL_CATALOG_URI"] == uri


def test_langgraph_text_blocks_exclude_reasoning_and_tool_payloads():
    assert (
        text_content(
            [
                {"type": "reasoning", "text": "private"},
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "input": "args"},
            ]
        )
        == "Hello"
    )


@pytest.mark.parametrize("summary,signature", [("", ""), ("", "signed"), ("summary", "")])
def test_langgraph_legacy_json_reasoning_is_not_answer_text(summary, signature):
    block = json.dumps(
        [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": summary, "signature": signature}],
            }
        ]
    )
    assert text_content(block) == ""
    assert text_content(block + block + "Hello 🌍") == "Hello 🌍"
    assert text_content('{"count": 1}') == '{"count": 1}'
    ordinary = '[{"type": "reasoning", "text": "ordinary JSON"}]'
    assert text_content(ordinary) == ordinary


@pytest.mark.asyncio
async def test_legacy_langgraph_stream_and_reply_hide_reasoning_but_checkpoints_keep_it(
    tmp_path, monkeypatch
):
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langgraph.graph import END, START, MessagesState, StateGraph

    from databricks_agentkit.sandbox.adapters.langgraph import invoke

    encoded = json.dumps(
        [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "", "signature": "test-signature"}],
            }
        ]
    )
    histories = []
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)

    class LegacyModel(FakeListChatModel):
        async def _astream(self, messages, **kwargs):
            # 0.21.0 emits a complete JSON string per reasoning delta.
            for chunk in [encoded, "Hello ", "🌍"]:
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

    async def create_graph(actor, *, model, workspace_client_for, checkpointer_override):
        llm = LegacyModel(responses=[encoded + "Hello 🌍"])

        async def call(state):
            histories.append([message.content for message in state["messages"]])
            return {"messages": [await llm.ainvoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        return graph.compile(checkpointer=checkpointer_override)

    project_modules(monkeypatch, create_agent_graph=create_graph)
    for message in ["first", "second"]:
        chunks = []
        result = await invoke(
            {"session_id": "s", "input": {"message": message}}, tmp_path, emit=chunks.append
        )
        assert chunks == ["Hello ", "🌍"]
        assert result["reply"] == "Hello 🌍"
    assert histories[1] == ["first", encoded + "Hello 🌍", "second"]


def test_trace_link_failure_retains_identity_without_failing_completed_agent(monkeypatch):
    def fail():
        raise RuntimeError("export unavailable")

    monkeypatch.setattr("databricks_agentkit.sandbox.adapters.common.app_workspace_client", fail)
    assert trace_result(SimpleNamespace(trace_id="tr-current")) == {
        "trace_id": "tr-current",
        "url": None,
        "export_status": "unverified",
    }


def test_trace_export_does_not_block_turn_and_drains_on_worker_shutdown(monkeypatch, tmp_path):
    flush = Mock()
    monkeypatch.setattr("mlflow.flush_trace_async_logging", flush)
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.adapters.common.app_workspace_client",
        lambda: SimpleNamespace(config=SimpleNamespace(host="https://workspace")),
    )
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "123")
    assert trace_result(SimpleNamespace(trace_id="tr-current")) == {
        "trace_id": "tr-current",
        "url": "https://workspace/ml/experiments/123/traces/tr-current",
    }
    flush.assert_not_called()
    shutdown_session(tmp_path)
    flush.assert_called_once()


@pytest.mark.asyncio
async def test_langgraph_reports_completed_native_tool_calls(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    from databricks_agentkit.sandbox.adapters.langgraph import invoke

    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)

    @tool
    def clock() -> str:
        """Return the test clock's reading."""
        return "14:00"

    async def create_graph(actor, *, model, workspace_client_for, checkpointer_override):
        def call(state):
            if isinstance(state["messages"][-1], ToolMessage):
                return {"messages": [AIMessage(content="It is 14:00.")]}
            return {
                "messages": [
                    AIMessage(content="", tool_calls=[{"name": "clock", "args": {}, "id": "tc-1"}])
                ]
            }

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call)
        graph.add_node("tools", ToolNode([clock]))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=checkpointer_override)

    project_modules(monkeypatch, create_agent_graph=create_graph)
    result = await invoke({"session_id": "tools", "input": {"message": "What time?"}}, tmp_path)
    assert result["reply"] == "It is 14:00."
    assert result["tool_calls"] == ["clock"]
    assert len(result["tool_events"]) == 1
    event = result["tool_events"][0]
    assert (event["call_id"], event["name"], event["status"]) == ("tc-1", "clock", "completed")
