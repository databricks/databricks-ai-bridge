"""State views use the agent's backend and request identity, including API-created turns.

These tests exercise state HTTP routing and real framework state. Only the managed Session
Store transport is replaced; no model or workspace is required for deterministic regressions.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest

from databricks_agentkit import DurableAgentServer
from databricks_agentkit.runtime.auth import InvocationAuthPolicy, RequestAuthContext
from databricks_agentkit.runtime.session_store_client import (
    Session,
    SessionItem,
    SessionStoreClient,
)
from databricks_agentkit.runtime.store import RUNTIME_STORE_LOCAL_ENV, InMemoryRuntimeStore


@pytest.fixture(params=["openai", "langgraph"])
def ui(request, monkeypatch):
    framework = request.param
    pytest.importorskip("agents" if framework == "openai" else "langgraph")
    path = (
        Path(__file__).parents[2]
        / f"src/databricks_agentbricks/templates/ui/agent-{framework}/runtime/ui.py"
    )
    spec = importlib.util.spec_from_file_location(f"state_ui_{framework}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    monkeypatch.delenv(RUNTIME_STORE_LOCAL_ENV, raising=False)
    monkeypatch.setenv("DATABRICKS_APP_NAME", "state-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://state-test.example")
    return framework, module


class _SessionTransport(SessionStoreClient):
    """In-memory remote transport, retaining serialized items across saver instances."""

    def __init__(self):
        self.items = {}

    def get_session(self, *, session_id):
        return Session("test", session_id, "actor")

    def append_items(self, session, *, items):
        self.items.setdefault(session.session_id, []).extend(json.loads(json.dumps(items)))

    def list_items(self, session, *, order_by=None):
        assert order_by == "create_time asc"
        return iter(
            SessionItem(str(i), item)
            for i, item in enumerate(self.items.get(session.session_id, []))
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("user_auth", [False, True])
async def test_api_turns_and_ui_read_the_same_ordered_history(ui, monkeypatch, user_auth):
    framework, module = ui
    session_id = str(uuid4())
    headers = {
        "x-forwarded-user": "user-id",
        "x-forwarded-email": "alice",
        "x-forwarded-access-token": "test-token",
    }
    if framework == "langgraph":
        from langchain_core.runnables import RunnableConfig
        from langgraph.graph import END, START, MessagesState, StateGraph

        import databricks_agentkit.langgraph.session_store as stores

        # Exercise serialized managed checkpoints, not browser-written duplicate message items.
        transport = _SessionTransport()
        saver = stores.DatabricksSessionStoreSaver("test", client=transport)
        monkeypatch.setattr(stores, "checkpointer", lambda: saver)
        monkeypatch.setenv("AGENT_SESSION_STORE", "test")
        monkeypatch.setattr(
            module,
            "_state_client",
            Mock(side_effect=AssertionError("history must read checkpoints")),
        )
        builder = StateGraph(cast(Any, MessagesState))
        builder.add_node(
            "reply",
            lambda state: {"messages": [("assistant", "reply " + state["messages"][-1].content)]},
        )
        builder.add_edge(START, "reply")
        builder.add_edge("reply", END)
        graph = builder.compile(checkpointer=saver)
        _install_graph(monkeypatch, graph)

        async def save(context, text, actor):
            await graph.ainvoke(
                {"messages": [("user", text)]},
                cast(RunnableConfig, stores.thread_config(context.session_id, actor)),
            )
    else:
        from databricks_agentkit.openai.sessions import session_store

        async def save(context, text, actor):
            await session_store(context.session_id, actor).add_items(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": "reply " + text},
                ]
            )

    app = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(),
        auth_policy=InvocationAuthPolicy(user_required=user_auth),
    )

    # Persist turns through the real framework backend with the runtime's documented mapping.
    actor = "alice"
    effective_session = session_id
    if user_auth:
        auth = RequestAuthContext.from_headers(headers)
        actor = auth.namespace("actor", actor)
        effective_session = auth.namespace("session", session_id)
        auth.close()
    context = SimpleNamespace(session_id=effective_session)
    for text in ("first API turn", "second API turn"):
        await save(context, text, actor)
    module.install_ui(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
    ) as client:
        result = await client.get("/api/demo/session/items", params={"session_id": session_id})
    assert result.status_code == 200, result.text
    assert [item["data"]["content"] for item in result.json()["session_items"]] == [
        "first API turn",
        "reply first API turn",
        "second API turn",
        "reply second API turn",
    ]
    assert result.json()["session_id"] == session_id


def test_managed_history_paginates_in_write_order(ui):
    _, module = ui
    client = object.__new__(module._ManagedStateClient)
    client._do = Mock(
        side_effect=[
            {"session_items": [{"item_id": "1"}], "next_page_token": "page-2"},
            {"session_items": [{"item_id": "2"}]},
        ]
    )
    assert client.list_session_items("conversation")["session_items"] == [
        {"item_id": "1"},
        {"item_id": "2"},
    ]
    assert client._do.call_args_list[-1].kwargs["query"] == {
        "order_by": "create_time asc",
        "page_size": 100,
        "page_token": "page-2",
    }


@pytest.mark.asyncio
async def test_user_state_routes_require_identity_and_preserve_public_session_ids(ui, monkeypatch):
    _, module = ui
    app = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(), auth_policy=InvocationAuthPolicy(user_required=True)
    )
    module.install_ui(app)
    monkeypatch.setenv("AGENT_SESSION_STORE", "test")
    monkeypatch.setenv("AGENT_MEMORY_STORE", "test")
    headers = {
        "x-forwarded-user": "user-id",
        "x-forwarded-email": "alice",
        "x-forwarded-access-token": "test-token",
    }
    auth = RequestAuthContext.from_headers(headers)
    effective_session = auth.namespace("session", "public-session")
    effective_actor = auth.namespace("actor", "alice")
    auth.close()
    state = Mock()
    state.ensure_session.return_value = {
        "session_id": effective_session,
        "actor_id": effective_actor,
    }
    state.get_session.return_value = state.ensure_session.return_value
    state.list_sessions.return_value = {
        "sessions": [
            {
                **state.ensure_session.return_value,
                "metadata": {"public_session_id": "public-session"},
            }
        ]
    }
    state.list_memory_entries.return_value = {"managed_memory_entries": []}
    monkeypatch.setattr(module, "_state_client", lambda: state)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        params={"session_id": "public-session"},
    ) as client:
        denied = await client.get("/api/demo/session")
        assert denied.status_code == 401
        state.get_session.assert_not_called()
        client.headers.update(headers)
        created = await client.post("/api/demo/sessions")
        assert created.json()["session_id"] == "public-session"
        state.ensure_session.assert_called_once_with(
            effective_actor, effective_session, "public-session"
        )
        listed = await client.get("/api/demo/sessions")
        assert listed.json()["sessions"][0]["session_id"] == "public-session"
        opened = await client.post("/api/demo/sessions/public-session/open")
        assert opened.json()["session_id"] == "public-session"
        state.get_session.assert_called_once_with(effective_session)
        await client.get("/api/demo/memory/entries")
        state.list_memory_entries.assert_called_once_with(effective_actor, None)
        state.get_session.return_value = {"actor_id": "another-actor"}
        assert (await client.post("/api/demo/sessions/public-session/open")).status_code == 403


def _install_graph(monkeypatch, graph):
    async def create_graph(actor, **kwargs):
        return graph

    monkeypatch.setitem(sys.modules, "agent", ModuleType("agent"))
    monkeypatch.setitem(
        sys.modules, "agent.agent", SimpleNamespace(create_agent_graph=create_graph)
    )


@pytest.mark.asyncio
async def test_langgraph_history_includes_messages_pending_parallel_approval(ui, monkeypatch):
    framework, module = ui
    if framework != "langgraph":
        pytest.skip("LangGraph checkpoint representation")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import interrupt

    builder = StateGraph(cast(Any, MessagesState))
    builder.add_node("reply", lambda state: {"messages": [("assistant", "parallel reply")]})
    builder.add_node("approval", lambda state: interrupt({"approval": True}))
    builder.add_edge(START, "reply")
    builder.add_edge(START, "approval")
    builder.add_edge("reply", END)
    builder.add_edge("approval", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    _install_graph(monkeypatch, graph)
    assert await module._checkpoint_history("new", "alice") == {
        "session_id": "new",
        "session_items": [],
        "interrupts": [],
    }
    await graph.ainvoke(
        {"messages": [("user", "hello")]},
        {"configurable": {"thread_id": "new", "actor_id": "alice"}},
    )
    result = await module._checkpoint_history("new", "alice")
    assert [item["data"]["content"] for item in result["session_items"]] == [
        "hello",
        "parallel reply",
    ]
    assert [item["value"] for item in result["interrupts"]] == [{"approval": True}]


@pytest.mark.asyncio
async def test_langgraph_history_errors_are_actionable_without_raw_exception(ui, monkeypatch):
    framework, module = ui
    if framework != "langgraph":
        pytest.skip("LangGraph checkpoint representation")
    from databricks.sdk.errors import PermissionDenied

    async def unavailable(*args, **kwargs):
        raise PermissionDenied("secret-sentinel", error_code="PERMISSION_DENIED")

    monkeypatch.setattr(module, "_checkpoint_history", unavailable)
    app = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(), auth_policy=InvocationAuthPolicy()
    )
    module.install_ui(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/demo/session/items", params={"session_id": "test"})
    assert response.status_code == 502
    assert "PERMISSION_DENIED" in response.text
    assert "Check access to the bound session store" in response.text
    assert "secret-sentinel" not in response.text


def test_user_session_listing_maps_current_api_session_without_ui_metadata(ui):
    _, module = ui
    assert module._chat_sessions(
        {"sessions": [{"session_id": "private-current"}, {"session_id": "private-unmapped"}]},
        user_scoped=True,
        current_effective_id="private-current",
        current_public_id="public-current",
    ) == [{"session_id": "public-current"}]
