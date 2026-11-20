"""DatabricksMemoryStore inside real compiled LangGraph runs: identity, concurrency, strict mode.

Each test compiles a StateGraph with ``store=`` and runs it with ``thread_config``, so the store
reads the actor from the run config the way an agent does. Only the workspace transport is faked
(``_memory_api_fake``). Async run-config tests need Python 3.11+: on 3.10 async code can't read the
run config, so the matching ``actor=`` variants run on every Python.
"""

# No `from __future__ import annotations` here: LangGraph injects `runtime: Runtime` by annotation.
import asyncio
import contextvars
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict, cast

import pytest

pytest.importorskip("langgraph")

from _memory_api_fake import STORE, install  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402
from langgraph.store.base import BaseStore, Item  # noqa: E402

from databricks_agentkit.langgraph import store as store_mod  # noqa: E402
from databricks_agentkit.langgraph.session_store import thread_config  # noqa: E402

needs_3_11 = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.10: async code can't read the run config; the actor= variant covers it",
)


class State(TypedDict, total=False):
    text: str
    found: object
    user: str
    op: str


@pytest.fixture
def api(monkeypatch):
    return install(monkeypatch)


def _chain(store, save, load):
    builder = StateGraph(cast(Any, State))
    builder.add_node("save", save)
    builder.add_node("load", load)
    builder.add_edge(START, "save")
    builder.add_edge("save", "load")
    builder.add_edge("load", END)
    return builder.compile(store=store)


def _store_of(runtime: Runtime) -> BaseStore:
    assert runtime.store is not None, "the graph was compiled without a store"
    return runtime.store


def save(state: State, runtime: Runtime) -> dict:
    if state.get("text"):
        _store_of(runtime).put(("notes",), "note", {"text": state["text"]})
    return {}


def load(state: State, runtime: Runtime) -> dict:
    item = _store_of(runtime).get(("notes",), "note")
    return {"found": item.value if item is not None else None}


async def async_save(state: State, runtime: Runtime) -> dict:
    if state.get("text"):
        await _store_of(runtime).aput(("notes",), "note", {"text": state["text"]})
    return {}


async def async_load(state: State, runtime: Runtime) -> dict:
    item = await _store_of(runtime).aget(("notes",), "note")
    return {"found": item.value if item is not None else None}


def _creates(api):
    return [call for call in api.calls if call.route == "create"]


# --- two users, one store ----------------------------------------------------------


def test_two_users_share_one_store_and_see_only_their_own_notes(api):
    graph = _chain(store_mod.DatabricksMemoryStore(store=STORE), save, load)

    assert graph.invoke({"text": "alice note"}, config=thread_config("alice-1", "alice")) == {
        "text": "alice note",
        "found": {"text": "alice note"},
    }
    assert graph.invoke({}, config=thread_config("bob-1", "bob"))["found"] is None
    # A new session for alice still sees her memory: memories outlive one conversation.
    assert graph.invoke({}, config=thread_config("alice-2", "alice"))["found"] == {
        "text": "alice note"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [pytest.param("run_config", marks=needs_3_11), "actor_kwarg"],
)
async def test_two_users_share_one_store_async(api, mode):
    if mode == "run_config":
        alice_graph = bob_graph = _chain(
            store_mod.DatabricksMemoryStore(store=STORE), async_save, async_load
        )
    else:
        alice_graph = _chain(
            store_mod.DatabricksMemoryStore(store=STORE, actor="alice"), async_save, async_load
        )
        bob_graph = _chain(
            store_mod.DatabricksMemoryStore(store=STORE, actor="bob"), async_save, async_load
        )

    alice = await alice_graph.ainvoke(
        {"text": "alice note"}, config=thread_config("alice-1", "alice")
    )
    bob = await bob_graph.ainvoke({}, config=thread_config("bob-1", "bob"))
    alice_again = await alice_graph.ainvoke({}, config=thread_config("alice-2", "alice"))

    assert alice["found"] == {"text": "alice note"}
    assert bob["found"] is None
    assert alice_again["found"] == {"text": "alice note"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [pytest.param("run_config", marks=needs_3_11), "actor_kwarg"],
)
async def test_fifty_concurrent_async_runs_do_not_mix_users(api, mode):
    users = [f"user-{i:02d}" for i in range(50)]
    runs = []
    for user in users:
        store = (
            store_mod.DatabricksMemoryStore(store=STORE)
            if mode == "run_config"
            else store_mod.DatabricksMemoryStore(store=STORE, actor=user)
        )
        graph = _chain(store, async_save, async_load)
        runs.append(
            graph.ainvoke({"text": f"note of {user}"}, config=thread_config(f"s-{user}", user))
        )

    results = await asyncio.gather(*runs)

    assert [result["found"] for result in results] == [
        {"text": f"note of {user}"} for user in users
    ]
    assert {call.actor_id for call in _creates(api)} == {f"{user}:langgraph" for user in users}
    assert len(api.rows()) == 50


# --- identity checks in a run -----------------------------------------------------------


@pytest.mark.parametrize(
    ("run_actor", "expect_error"),
    [
        pytest.param("alice", False, id="matching-actor"),
        pytest.param("bob", True, id="different-actor"),
        pytest.param(None, True, id="session-defaulted-actor"),
    ],
)
def test_explicit_actor_is_checked_against_the_run_actor(api, run_actor, expect_error):
    graph = _chain(store_mod.DatabricksMemoryStore(store=STORE, actor="alice"), save, load)
    config = thread_config("s1", run_actor)  # run_actor=None: the actor defaults to "s1"

    if expect_error:
        with pytest.raises(ValueError, match="built for actor 'alice'"):
            graph.invoke({"text": "x"}, config=config)
        assert api.calls == []
        return
    graph.invoke({"text": "x"}, config=config)
    assert {call.actor_id for call in _creates(api)} == {"alice:langgraph"}


def test_warns_once_when_the_actor_defaults_to_the_session_id(api, caplog, monkeypatch):
    monkeypatch.setattr(store_mod, "_warned", set())
    graph = _chain(store_mod.DatabricksMemoryStore(store=STORE), save, load)

    with caplog.at_level(logging.WARNING, logger=store_mod.logger.name):
        graph.invoke({"text": "a"}, config=thread_config("s9"))
        graph.invoke({"text": "b"}, config=thread_config("s9"))

    warned = [r for r in caplog.records if "session id" in r.getMessage()]
    assert len(warned) == 1
    assert {call.actor_id for call in _creates(api)} == {"s9:langgraph"}


@pytest.mark.parametrize(
    ("langgraph_user_id", "error"),
    [
        pytest.param(None, "langgraph_user_id", id="key-missing"),
        pytest.param("bob", "refusing to read or write another user's memory", id="mismatch"),
        pytest.param("alice", None, id="match"),
    ],
)
def test_strict_mode_needs_the_user_key_to_match_the_actor(api, langgraph_user_id, error):
    store = store_mod.DatabricksMemoryStore(store=STORE, strict_user_key="langgraph_user_id")
    graph = _chain(store, save, load)
    config = thread_config("s1", "alice")
    if langgraph_user_id is not None:
        config["configurable"]["langgraph_user_id"] = langgraph_user_id

    if error is not None:
        with pytest.raises(ValueError, match=error):
            graph.invoke({"text": "mine"}, config=config)
        assert api.calls == []
        return
    assert graph.invoke({"text": "mine"}, config=config)["found"] == {"text": "mine"}
    assert {call.actor_id for call in api.calls} == {"alice:langgraph"}


def test_outside_a_run_the_actor_kwarg_is_used_and_no_actor_raises(api):
    store = store_mod.DatabricksMemoryStore(store=STORE, actor="alice")
    store.put(("ns",), "k", {"v": 1})
    item = store.get(("ns",), "k")
    assert item is not None and item.value == {"v": 1}
    assert [call.route for call in api.calls] == ["create", "list"]
    assert {call.actor_id for call in api.calls} == {"alice:langgraph"}

    with pytest.raises(ValueError, match="needs a run config"):
        store_mod.DatabricksMemoryStore(store=STORE).put(("ns",), "k", {"v": 2})
    # Strict mode needs a run config to verify the user, so an actor= alone is not enough.
    strict = store_mod.DatabricksMemoryStore(
        store=STORE, actor="alice", strict_user_key="langgraph_user_id"
    )
    with pytest.raises(ValueError, match="needs a run config"):
        strict.get(("ns",), "k")
    assert len(api.calls) == 2


# --- threads and context -------------------------------------------------------------


@pytest.mark.parametrize(
    ("use_copy_context", "expected"),
    [
        pytest.param(False, "copy_context", id="plain-thread-pool"),
        pytest.param(True, {"text": "mine"}, id="copied-context"),
    ],
)
def test_a_thread_started_in_a_node_needs_a_copied_context(api, use_copy_context, expected):
    def fetch_in_pool(state: State, runtime: Runtime) -> dict:
        store = _store_of(runtime)
        ctx = contextvars.copy_context()  # taken on the node's thread, where the run config lives

        def read() -> Item | None:
            return store.get(("notes",), "note")

        def read_in_copied_context() -> Item | None:
            return ctx.run(read)

        target = read_in_copied_context if use_copy_context else read
        with ThreadPoolExecutor(max_workers=1) as pool:
            item = pool.submit(target).result()
        return {"found": item.value if item is not None else None}

    graph = _chain(store_mod.DatabricksMemoryStore(store=STORE), save, fetch_in_pool)
    config = thread_config("s1", "alice")

    if use_copy_context:
        assert graph.invoke({"text": "mine"}, config=config)["found"] == expected
        return
    with pytest.raises(ValueError, match=expected):
        graph.invoke({"text": "mine"}, config=config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [pytest.param("run_config", marks=needs_3_11), "actor_kwarg"],
)
async def test_abatch_reads_the_actor_on_the_event_loop_not_in_the_worker(api, monkeypatch, mode):
    config_reads = []
    real_get_config = store_mod.get_config

    def recording_get_config():
        config_reads.append(threading.get_ident())
        return real_get_config()

    monkeypatch.setattr(store_mod, "get_config", recording_get_config)
    loop_thread = []

    async def record_and_save(state: State, runtime: Runtime) -> dict:
        loop_thread.append(threading.get_ident())
        await _store_of(runtime).aput(("notes",), "note", {"text": "mine"})
        return {}

    store = (
        store_mod.DatabricksMemoryStore(store=STORE)
        if mode == "run_config"
        else store_mod.DatabricksMemoryStore(store=STORE, actor="alice")
    )
    graph = _chain(store, record_and_save, async_load)
    result = await graph.ainvoke({}, config=thread_config("s1", "alice"))

    assert result["found"] == {"text": "mine"}
    assert set(config_reads) == {loop_thread[0]}
    # The service call itself ran on a worker thread, so the actor had to be read before it.
    assert loop_thread[0] not in {call.thread for call in api.calls}


def test_langmem_style_namespace_is_partitioned_by_actor_not_by_label(api):
    store = store_mod.DatabricksMemoryStore(store=STORE, strict_user_key="langgraph_user_id")

    def act(state: State, runtime: Runtime) -> dict:
        # LangMem-style: the namespace is templated on the user id.
        namespace = ("memories", state["user"])
        if state["op"] == "remember":
            _store_of(runtime).put(namespace, "fact", {"text": state["text"]})
            return {}
        return {"found": [item.value for item in _store_of(runtime).search(namespace)]}

    builder = StateGraph(cast(Any, State))
    builder.add_node("act", act)
    builder.add_edge(START, "act")
    builder.add_edge("act", END)
    graph = builder.compile(store=store)

    alice_config = thread_config("s-a", "alice")
    alice_config["configurable"]["langgraph_user_id"] = "alice"
    bob_config = thread_config("s-b", "bob")
    bob_config["configurable"]["langgraph_user_id"] = "bob"

    graph.invoke({"user": "alice", "op": "remember", "text": "likes tea"}, config=alice_config)
    # Bob asks for alice's namespace label: the actor partitions, so he gets nothing of hers.
    assert graph.invoke({"user": "alice", "op": "recall"}, config=bob_config)["found"] == []
    assert graph.invoke({"user": "alice", "op": "recall"}, config=alice_config)["found"] == [
        {"text": "likes tea"}
    ]
