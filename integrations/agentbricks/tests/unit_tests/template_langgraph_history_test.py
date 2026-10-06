"""Read template history through real checkpoints, without live tool discovery."""

import importlib
import importlib.util
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import httpx
import pytest

from databricks_agentkit.runtime.session_store_client import (
    Session,
    SessionItem,
    SessionStoreClient,
)

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"


class SessionStoreTransport(SessionStoreClient):
    """Replace REST transport only; retain managed checkpoint serialization."""

    def __init__(self):
        self.items = {}
        self._store_name = None

    def get_session(self, *, session_id: str) -> Session:
        return Session("test", session_id, "actor")

    def append_items(self, session: Session, *, items: Sequence[Any]) -> None:
        self.items.setdefault(session.session_id, []).extend(json.loads(json.dumps(items)))

    def list_items(self, session: Session, *, order_by: str | None = None) -> Iterator[SessionItem]:
        assert order_by == "create_time asc"
        for index, item in enumerate(self.items.get(session.session_id, [])):
            yield SessionItem(str(index), item)


@pytest.fixture(params=["memory", "managed"])
def template(request, monkeypatch):
    for dependency in ("databricks_langchain", "langgraph", "langchain", "langchain_mcp_adapters"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[langgraph]")
    from langgraph.checkpoint.memory import InMemorySaver

    from databricks_agentkit.langgraph import session_store

    monkeypatch.syspath_prepend(str(TEMPLATES / "agent-langgraph"))
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, name)
    agent = importlib.import_module("agent.agent")
    adapter = importlib.import_module("runtime.adapter")
    spec = importlib.util.spec_from_file_location(
        "history_ui", TEMPLATES / "ui/agent-langgraph/runtime/ui.py"
    )
    assert spec and spec.loader
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    saver = (
        InMemorySaver()
        if request.param == "memory"
        else session_store.DatabricksSessionStoreSaver("test", client=SessionStoreTransport())
    )
    monkeypatch.setattr(session_store, "checkpointer", lambda: saver)
    monkeypatch.setattr(agent, "checkpointer", lambda: saver)
    monkeypatch.setattr(agent, "start_trace", lambda **_kwargs: nullcontext())

    async def unavailable(*_args, **_kwargs):
        from databricks_agentkit.runtime.auth import AuthError

        raise AuthError(
            "MCP_USER_AUTHORIZATION_MISSING", "Live tools must not initialize for history"
        )

    monkeypatch.setattr(agent, "create_agent_graph", unavailable)
    yield ui, agent, adapter, saver
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            sys.modules.pop(name, None)


def graph_builder():
    from langgraph.graph import MessagesState, StateGraph

    return StateGraph(cast(Any, MessagesState))


@pytest.mark.asyncio
async def test_history_reads_completed_messages_without_live_tools(template):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.graph import END, START

    from databricks_agentkit.langgraph import thread_config

    ui, _agent, _adapter, saver = template

    def answer(_state):
        return {
            "messages": [
                AIMessage(
                    content="", id="call", tool_calls=[{"name": "lookup", "args": {}, "id": "t"}]
                ),
                ToolMessage(content="71", tool_call_id="t", id="result"),
                AIMessage(content="The total is 71.", id="answer"),
            ]
        }

    builder = graph_builder()
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=saver)
    config = thread_config("conversation", "alice")
    await graph.ainvoke(
        {"messages": [HumanMessage(content="Find the total.", id="prompt")]}, config
    )
    expected = await graph.aget_state(config)
    result = await ui._checkpoint_history("conversation", "alice")
    assert [item["data"] for item in result["session_items"]] == [
        message.model_dump() for message in expected.values["messages"]
    ]
    assert result["interrupts"] == []
    assert (await ui._checkpoint_history("unused", "alice"))["session_items"] == []


@pytest.mark.asyncio
async def test_history_reduces_pending_writes_and_preserves_interrupts(template):
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START
    from langgraph.types import interrupt

    from databricks_agentkit.langgraph import thread_config

    ui, _agent, _adapter, saver = template
    executions = []

    def revise(_state):
        executions.append("revise")
        return {"messages": [AIMessage(content="revised answer", id="answer")]}

    def approve(_state):
        executions.append("approve")
        interrupt({"action_requests": [{"name": "send_message", "args": {"text": "hello"}}]})
        raise AssertionError("History must not resume approvals")

    builder = graph_builder()
    builder.add_node("revise", revise)
    builder.add_node("approve", approve)
    builder.add_edge(START, "revise")
    builder.add_edge(START, "approve")
    builder.add_edge("revise", END)
    builder.add_edge("approve", END)
    graph = builder.compile(checkpointer=saver)
    config = thread_config("conversation", "alice")
    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="hello", id="prompt"),
                AIMessage(content="draft", id="answer"),
            ]
        },
        config,
    )
    saved = await saver.aget_tuple(config)
    assert saved and any(channel == "messages" for _, channel, _ in saved.pending_writes)
    expected = await graph.aget_state(config)
    before = list(executions)
    for _attempt in range(2):
        result = await ui._checkpoint_history("conversation", "alice")
        assert [item["data"] for item in result["session_items"]] == [
            message.model_dump() for message in expected.values["messages"]
        ]
        assert result["interrupts"] == [
            {"id": item.id, "value": item.value} for item in expected.interrupts
        ]
    assert [item["data"]["content"] for item in result["session_items"]] == [
        "hello",
        "revised answer",
    ]
    assert executions == before


@pytest.mark.asyncio
async def test_obo_history_matches_invocation_namespace_and_isolates_users(template, monkeypatch):
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START

    from databricks_agentkit import DurableAgentServer
    from databricks_agentkit.runtime.auth import InvocationAuthPolicy
    from databricks_agentkit.runtime.store import InMemoryRuntimeStore

    ui, agent, adapter, saver = template
    monkeypatch.setenv("DATABRICKS_APP_NAME", "history-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    builder = graph_builder()
    builder.add_node(
        "answer", lambda _: {"messages": [AIMessage(content="saved answer", id="answer")]}
    )
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=saver)

    async def create_graph(*_args, **_kwargs):
        return graph

    app = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(),
        auth_policy=InvocationAuthPolicy(user_tools=("genie",)),
    )
    app.invoke(adapter.invoke)
    ui.install_ui(app)
    headers = {
        "x-forwarded-user": "alice-id",
        "x-forwarded-email": "alice@example.com",
        "x-forwarded-access-token": "test-token",
    }
    await app._runtime.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="https://test"
        ) as client:
            with monkeypatch.context() as turn:
                turn.setattr(agent, "create_agent_graph", create_graph)
                invocation = await client.post(
                    "/api/invocations",
                    headers=headers,
                    json={
                        "id": str(uuid4()),
                        "session_id": "public",
                        "input": {
                            "actor": "alice@example.com",
                            "messages": [{"role": "user", "content": "saved prompt"}],
                        },
                    },
                )
            assert invocation.status_code == 200, invocation.text
            result = await client.get("/api/demo/session/items?session_id=public", headers=headers)
            assert result.status_code == 200, result.text
            assert result.json()["session_id"] == "public"
            assert [item["data"]["content"] for item in result.json()["session_items"]] == [
                "saved prompt",
                "saved answer",
            ]
            other = await client.get(
                "/api/demo/session/items?session_id=public",
                headers={**headers, "x-forwarded-user": "bob-id"},
            )
            assert other.status_code == 200, other.text
            assert other.json()["session_items"] == []
            missing = await client.get("/api/demo/session/items?session_id=public")
            assert missing.status_code == 401
    finally:
        await app._runtime.stop()
