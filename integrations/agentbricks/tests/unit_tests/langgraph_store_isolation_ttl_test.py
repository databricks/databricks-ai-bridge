"""DatabricksMemoryStore: the storage actor, the stored-value envelope and client-side TTL.

Same seam as ``langgraph_store_test.py``: the real AgentKit client over the wire fake
(``_memory_api_fake``). The store's clock is replaced per test, so no test sleeps.
"""

import asyncio
import logging
import uuid
from contextlib import contextmanager
from typing import Any

import pytest

pytest.importorskip("langgraph")

from _memory_api_fake import STORE, install  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.config import var_child_runnable_config  # noqa: E402
from langgraph.store.base import PutOp  # noqa: E402

from databricks_agentkit.langgraph import memory as memory_mod  # noqa: E402
from databricks_agentkit.langgraph import session_store as session_store_mod  # noqa: E402
from databricks_agentkit.langgraph import store as store_mod  # noqa: E402
from databricks_agentkit.langgraph import thread_config  # noqa: E402

T0 = 1_800_000_000_000  # an arbitrary "now", in epoch milliseconds
MINUTE = 60_000


@pytest.fixture
def api(monkeypatch):
    return install(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(store_mod, "_warned", set())


class Clock:
    def __init__(self, now: int = T0) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _store(actor="alice", clock=None, **kwargs):
    store = store_mod.DatabricksMemoryStore(store=STORE, actor=actor, **kwargs)
    if clock is not None:
        store._now_ms = clock
    return store


@contextmanager
def run_as(actor, **configurable):
    config: RunnableConfig = {
        "configurable": {"thread_id": "s1", "actor_id": actor, **configurable}
    }
    token = var_child_runnable_config.set(config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _warnings(caplog, text):
    return [r for r in caplog.records if text in r.getMessage()]


# --- the storage actor --------------------------------------------------------------


def test_every_operation_goes_to_the_storage_actor_never_the_plain_one(api):
    store = _store()
    store.put(("ns",), "k", {"note": "tea"})
    store.put(("ns",), "k", {"note": "green tea"})  # create conflict, then update
    store.get(("ns",), "k")
    store.search(("ns",))
    store.search(("ns",), query="tea")
    store.list_namespaces()
    store.put(("ns",), "k", None)

    actors = {call.actor_id for call in api.calls if call.actor_id is not None}
    assert actors == {"alice:langgraph"}
    assert {"create", "list", "update", "search", "delete"} <= set(api.routes())


def test_storage_actor_composes_with_composite_actor_ids():
    assert store_mod.DatabricksMemoryStore.storage_actor("alice") == "alice:langgraph"
    assert store_mod.DatabricksMemoryStore.storage_actor("acme:alice") == "acme:alice:langgraph"


@pytest.mark.parametrize("actor", ["alice:langgraph", "Alice:LangGraph", "acme:langgraph"])
def test_an_actor_ending_with_the_reserved_suffix_is_refused_everywhere(api, monkeypatch, actor):
    with pytest.raises(ValueError, match="can't end with ':langgraph'"):
        store_mod.DatabricksMemoryStore(store=STORE, actor=actor)
    store = store_mod.DatabricksMemoryStore(store=STORE)
    with run_as(actor):
        with pytest.raises(ValueError, match="can't end with ':langgraph'"):
            store.put(("ns",), "k", {"v": 1})
        with pytest.raises(ValueError, match="can't end with ':langgraph'"):
            store.get(("ns",), "k")
    with pytest.raises(ValueError, match="can't end with ':langgraph'"):
        store.sweep_ttl(actor=actor)
    with pytest.raises(ValueError, match="can't end with ':langgraph'"):
        thread_config("s1", actor)
    with pytest.raises(ValueError, match="can't end with ':langgraph'"):
        thread_config(actor)  # the actor defaults to the session id
    with pytest.raises(ValueError, match="can't end with ':langgraph'"):  # a hand-built config
        session_store_mod._parse_config({"configurable": {"thread_id": "s1", "actor_id": actor}})
    monkeypatch.setenv("AGENT_MEMORY_STORE", STORE)
    with pytest.raises(ValueError, match="can't end with ':langgraph'"):
        memory_mod.memory_tools(actor)
    assert api.calls == []


@pytest.mark.parametrize("actor", ["alice:langgraph", "Alice:LangGraph"])
def test_the_openai_session_and_memory_helpers_refuse_the_reserved_suffix(monkeypatch, actor):
    openai_memory = pytest.importorskip("databricks_agentkit.openai.memory")
    openai_sessions = pytest.importorskip("databricks_agentkit.openai.sessions")
    monkeypatch.setenv("AGENT_MEMORY_STORE", STORE)
    with pytest.raises(ValueError, match="can't end with ':langgraph'"):
        openai_memory.memory_tools(actor)
    with pytest.raises(ValueError, match="can't end with ':langgraph'"):
        openai_sessions.session_store("s1", actor)


def test_actor_ids_that_are_not_strings_pass_through_the_helpers_unchanged():
    actor = uuid.uuid4()
    assert thread_config("s1", actor)["configurable"]["actor_id"] is actor  # ty: ignore[invalid-argument-type]


def test_memory_tools_for_the_same_user_never_see_store_entries(api, monkeypatch):
    monkeypatch.setattr(memory_mod, "workspace_client", api.workspace_client)
    monkeypatch.setenv("AGENT_MEMORY_STORE", STORE)
    _store().put(("prefs",), "style", {"note": "tea, no sugar"})

    tools = {t.name: t for t in memory_mod.memory_tools("alice")}

    recall_args: dict[str, Any] = {"query": "tea"}
    list_args: dict[str, Any] = {}
    assert "tea" not in tools["recall"].invoke(recall_args)
    assert "prefs" not in tools["list_memories"].invoke(list_args)


# --- the envelope -------------------------------------------------------------------------


def test_a_newer_envelope_version_is_skipped_with_one_warning(api, caplog):
    api.seed("/langgraph/ns/k", '{"lg":2,"v":{"note":"tea"},"y":1}')
    store = _store()

    with caplog.at_level(logging.WARNING, logger=store_mod.logger.name):
        assert store.get(("ns",), "k") is None
        assert store.search(("ns",)) == []
        assert store.search(("ns",), query="tea") == []
        assert store.list_namespaces() == []

    assert len(_warnings(caplog, "newer version")) == 1


def test_foreign_entries_warn_once_without_naming_paths_or_values(api, caplog):
    api.seed("/langgraph/users/alice@example.com/a", "User prefers tea")
    api.seed("/langgraph/users/alice@example.com/b", '{"note":"tea"}')
    store = _store()

    with caplog.at_level(logging.WARNING, logger=store_mod.logger.name):
        assert store.search(("users",)) == []
        assert store.list_namespaces() == []

    [warning] = _warnings(caplog, "this store wrote")
    assert "alice" not in warning.getMessage() and "tea" not in warning.getMessage()


def test_a_value_that_is_not_valid_unicode_is_refused_before_any_call(api):
    with pytest.raises(ValueError, match="isn't valid Unicode"):
        _store().put(("ns",), "k", {"text": "\ud800"})
    assert api.calls == []


# --- TTL ----------------------------------------------------------------------------


def test_ttl_sets_an_expiry_from_the_clock_and_expires_at_that_millisecond(api):
    clock = Clock()
    store = _store(clock=clock)
    store.put(("ns",), "k", {"v": 1}, ttl=1.5)

    [row] = api.rows()
    assert row["content"] == store_mod._wrap({"v": 1}, T0 + 90_000)
    clock.now = T0 + 90_000 - 1
    assert store.get(("ns",), "k").value == {"v": 1}
    clock.now = T0 + 90_000
    assert store.get(("ns",), "k") is None


def test_expired_items_are_hidden_from_every_read_and_reads_never_write(api):
    clock = Clock()
    store = _store(clock=clock)
    store.put(("old",), "k", {"note": "tea"}, ttl=1)
    store.put(("new",), "k", {"note": "tea"})
    clock.now += 2 * MINUTE
    api.calls.clear()

    assert store.get(("old",), "k") is None
    assert [item.namespace for item in store.search(())] == [("new",)]
    assert [item.namespace for item in store.search((), query="tea")] == [("new",)]
    assert store.list_namespaces() == [("new",)]

    # get and search default to refresh_ttl=True; this store neither refreshes nor deletes on read.
    assert set(api.routes()) <= {"list", "search"}
    assert len(api.rows()) == 2  # the expired entry stays until a sweep


def test_put_without_ttl_clears_an_earlier_ttl(api):
    clock = Clock()
    store = _store(clock=clock)
    store.put(("ns",), "k", {"v": 1}, ttl=1)
    store.put(("ns",), "k", {"v": 2})
    clock.now += 10 * MINUTE

    assert store.get(("ns",), "k").value == {"v": 2}
    assert '"x"' not in api.rows()[0]["content"]


def test_default_ttl_applies_to_puts_that_do_not_pass_ttl(api):
    clock = Clock()
    store = _store(clock=clock, ttl={"default_ttl": 2})
    store.put(("ns",), "default", {"v": 1})
    store.put(("ns",), "explicit", {"v": 1}, ttl=5)
    store.put(("ns",), "none", {"v": 1}, ttl=None)  # an explicit None means no TTL
    store.batch([PutOp(("ns",), "direct", {"v": 1})])  # a raw PutOp carries its own ttl (None)

    contents = {row["path"].rsplit("/", 1)[-1]: row["content"] for row in api.rows()}
    assert contents == {
        "default": store_mod._wrap({"v": 1}, T0 + 2 * MINUTE),
        "explicit": store_mod._wrap({"v": 1}, T0 + 5 * MINUTE),
        "none": store_mod._wrap({"v": 1}, None),
        "direct": store_mod._wrap({"v": 1}, None),
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"sweep_interval_minutes": 5}, "sweep_interval_minutes isn't supported"),
        ({"refresh_on_read": True}, "refresh_on_read isn't supported"),
        ({"default_ttl": 0}, "ttl must be a positive number of minutes"),
        ({"default_ttl": -1.0}, "ttl must be a positive number of minutes"),
    ],
)
def test_unsupported_ttl_settings_are_refused_at_construction(api, config, message):
    with pytest.raises(ValueError, match=message):
        _store(ttl=config)
    assert api.calls == []


def test_supported_ttl_settings_are_kept_and_omit_expired_false_returns_expired_items(api):
    clock = Clock()
    store = _store(clock=clock, ttl={"refresh_on_read": False, "omit_expired": False})
    assert store.ttl_config == {"refresh_on_read": False, "omit_expired": False}
    store.put(("ns",), "k", {"v": 1}, ttl=1)
    clock.now += 2 * MINUTE

    assert store.get(("ns",), "k").value == {"v": 1}
    assert [item.key for item in store.search(("ns",))] == ["k"]


def test_sweep_deletes_only_one_users_expired_items(api):
    clock = Clock()
    alice, bob = _store("alice", clock=clock), _store("bob", clock=clock)
    alice.put(("ns",), "expired", {"v": 1}, ttl=1)
    alice.put(("ns",), "live", {"v": 2}, ttl=60)
    alice.put(("ns",), "forever", {"v": 3})
    bob.put(("ns",), "expired", {"v": 4}, ttl=1)
    api.seed("/langgraph/ns/foreign", "User likes tea")
    clock.now += 2 * MINUTE

    assert alice.sweep_ttl() == 1
    remaining = sorted((row["actor_id"], row["path"]) for row in api.rows())
    assert remaining == [
        ("alice:langgraph", "/langgraph/ns/foreign"),
        ("alice:langgraph", "/langgraph/ns/forever"),
        ("alice:langgraph", "/langgraph/ns/live"),
        ("bob:langgraph", "/langgraph/ns/expired"),
    ]
    assert bob.sweep_ttl(actor="bob") == 1
    assert alice.sweep_ttl() == 0


def test_sweep_keeps_an_entry_that_a_put_renewed_after_the_listing(api, monkeypatch):
    clock = Clock()
    store = _store(clock=clock)
    store.put(("ns",), "k", {"v": 1}, ttl=1)
    clock.now += 2 * MINUTE
    at = store._at

    def renew_then_reread(handle, partition, path):
        _store(clock=clock).put(("ns",), "k", {"v": 2}, ttl=60)  # another process renews it
        return at(handle, partition, path)

    monkeypatch.setattr(store, "_at", renew_then_reread)

    assert store.sweep_ttl() == 0
    monkeypatch.undo()
    assert store.get(("ns",), "k").value == {"v": 2}


def test_sweep_uses_the_run_actor_and_has_an_async_form(api):
    clock = Clock()
    store = _store(actor=None, clock=clock)
    with run_as("carol"):
        store.put(("ns",), "a", {"v": 1}, ttl=1)
        store.put(("ns",), "b", {"v": 1}, ttl=1)
    clock.now += 2 * MINUTE
    with run_as("carol"):
        assert store.sweep_ttl() == 2

    with run_as("carol"):
        store.put(("ns",), "c", {"v": 1}, ttl=1)
    clock.now += 2 * MINUTE
    assert asyncio.run(store.asweep_ttl(actor="carol")) == 1
    assert api.rows() == []
