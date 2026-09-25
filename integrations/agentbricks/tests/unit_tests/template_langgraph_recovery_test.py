"""Recover template responses from real LangGraph checkpoints without a model service."""

import importlib
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from databricks_agentkit.runtime.session_store_client import (
    Session,
    SessionItem,
    SessionStoreClient,
)


class SessionStoreTransport(SessionStoreClient):
    """Replace only the remote Session Store transport; run the real managed checkpointer."""

    def __init__(self):
        self.items = []
        self._store_name = None

    def get_session(self, *, session_id: str) -> Session:
        return Session("test", session_id, "conversation")

    def append_items(self, session: Session, *, items: Sequence[Any]) -> None:
        self.items.extend(json.loads(json.dumps(items)))

    def list_items(self, session: Session, *, order_by: str | None = None) -> Iterator[SessionItem]:
        assert order_by == "create_time asc"
        for index, item in enumerate(self.items):
            yield SessionItem(str(index), item)


def messages_graph():
    from langgraph.graph import MessagesState, StateGraph

    # This is LangGraph's canonical TypedDict schema; ty does not match its structural StateLike
    # bound yet. Keep the cast on the schema, not on graph operations or node callbacks.
    return StateGraph(cast(Any, MessagesState))


@pytest.fixture(params=["memory", "managed"])
def template(request, monkeypatch):
    for dependency in ("databricks_langchain", "langgraph", "langchain", "langchain_mcp_adapters"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[langgraph]")
    from langgraph.checkpoint.memory import InMemorySaver

    from databricks_agentkit.langgraph.session_store import DatabricksSessionStoreSaver

    template_path = (
        Path(__file__).parents[2] / "src/databricks_agentbricks/templates/agent-langgraph"
    )
    monkeypatch.syspath_prepend(str(template_path))
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, name)
    adapter = importlib.import_module("runtime.adapter")
    agent = importlib.import_module("agent.agent")
    saver = (
        InMemorySaver()
        if request.param == "memory"
        else DatabricksSessionStoreSaver("test", client=SessionStoreTransport())
    )
    monkeypatch.setattr(agent, "checkpointer", lambda: saver)
    monkeypatch.setattr(agent, "start_trace", lambda **_kwargs: nullcontext())

    def install_graph(builder):
        graph = builder.compile(checkpointer=saver)

        async def create_graph(*_args, **_kwargs):
            return graph

        # Substitute the external model/tool construction, not graph execution or recovery.
        monkeypatch.setattr(agent, "create_agent_graph", create_graph)
        return graph

    yield adapter, install_graph
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            sys.modules.pop(name, None)


def context(invocation_id):
    events = []

    async def emit(event):
        events.append(event)

    return SimpleNamespace(
        invocation_id=invocation_id,
        session_id="conversation",
        emit=emit,
        events=events,
    )


def message_input(content):
    return {"messages": [{"role": "user", "content": content}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier_turn", [False, True])
async def test_completed_recovery_restores_only_current_invocation(template, earlier_turn):
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START

    adapter, install_graph = template
    calls = []

    def answer(state):
        content = state["messages"][-1].content
        calls.append(content)
        return {"messages": [AIMessage(content=f"answer:{content}", id=f"answer-{content}")]}

    graph = messages_graph()
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    install_graph(graph)
    if earlier_turn:
        await adapter.invoke(message_input("earlier"), context("previous"))

    payload = message_input("current")
    original = await adapter.invoke(payload, context("current"))
    assert [item["content"] for item in original["output"]] == ["answer:current"]
    for _attempt in range(2):
        recovered_context = context("current")
        recovered = await adapter.recover(payload, recovered_context)
        assert recovered == original
        assert recovered_context.events == []
    assert calls == (["earlier", "current"] if earlier_turn else ["current"])


@pytest.mark.asyncio
async def test_partial_recovery_keeps_checkpointed_and_new_output(template):
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START

    adapter, install_graph = template
    calls = []

    def first(state):
        calls.append("first")
        return {"messages": [AIMessage(content="first result", id="first")]}

    def second(state):
        calls.append("second")
        if calls.count("second") == 1:
            raise RuntimeError("worker failed after first checkpoint")
        return {"messages": [AIMessage(content="second result", id="second")]}

    graph = messages_graph()
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    install_graph(graph)
    payload = message_input("current")
    with pytest.raises(RuntimeError, match="worker failed"):
        await adapter.invoke(payload, context("current"))

    recovered_context = context("current")
    recovered = await adapter.recover(payload, recovered_context)
    assert [item["content"] for item in recovered["output"]] == [
        "first result",
        "second result",
    ]
    assert recovered["status"] == "completed"
    assert calls == ["first", "second", "second"]
    assert [
        event["message"]["content"]
        for event in recovered_context.events
        if event["type"] == "message"
    ] == ["second result"]


@pytest.mark.asyncio
async def test_recovery_preserves_pause_and_completed_resume_response(template):
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START
    from langgraph.types import interrupt

    adapter, install_graph = template
    side_effects = []

    def prepare(state):
        return {"messages": [AIMessage(content="waiting for approval", id="prepare")]}

    def approval(state):
        approved = interrupt("Approve the action?")
        side_effects.append(approved)
        return {"messages": [AIMessage(content="action completed", id="approved")]}

    graph = messages_graph()
    graph.add_node("prepare", prepare)
    graph.add_node("approval", approval)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "approval")
    graph.add_edge("approval", END)
    install_graph(graph)
    payload = message_input("current")
    paused = await adapter.invoke(payload, context("pause"))
    assert paused["status"] == "interrupted"
    assert await adapter.recover(payload, context("pause")) == paused
    assert side_effects == []

    resumed = await adapter.invoke({"resume": True}, context("resume"))
    assert [item["content"] for item in resumed["output"]] == ["action completed"]
    recovered = await adapter.recover({"resume": True}, context("resume"))
    assert recovered == resumed
    assert side_effects == [True]


@pytest.mark.asyncio
async def test_parallel_recovery_reuses_pending_writes_without_duplicate_output(template):
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START

    adapter, install_graph = template
    calls = []

    def left(state):
        calls.append("left")
        return {"messages": [AIMessage(content="left result", id="left")]}

    def right(state):
        calls.append("right")
        if calls.count("right") == 1:
            raise RuntimeError("worker failed during parallel step")
        return {"messages": [AIMessage(content="right result", id="right")]}

    graph = messages_graph()
    graph.add_node("left", left)
    graph.add_node("right", right)
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge("left", END)
    graph.add_edge("right", END)
    install_graph(graph)
    payload = message_input("current")
    with pytest.raises(RuntimeError, match="worker failed"):
        await adapter.invoke(payload, context("current"))

    recovered = await adapter.recover(payload, context("current"))
    assert sorted(item["content"] for item in recovered["output"]) == [
        "left result",
        "right result",
    ]
    assert calls.count("left") == 1
    assert calls.count("right") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier_turn", [False, True])
async def test_recovery_without_matching_checkpoint_runs_original_input(template, earlier_turn):
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START

    adapter, install_graph = template
    calls = []

    def answer(state):
        content = state["messages"][-1].content
        calls.append(content)
        return {"messages": [AIMessage(content=f"answer:{content}", id=f"answer-{content}")]}

    graph = messages_graph()
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    install_graph(graph)
    if earlier_turn:
        await adapter.invoke(message_input("earlier"), context("previous"))

    response = await adapter.recover(message_input("current"), context("current"))
    assert [item["content"] for item in response["output"]] == ["answer:current"]
    assert calls == (["earlier", "current"] if earlier_turn else ["current"])
