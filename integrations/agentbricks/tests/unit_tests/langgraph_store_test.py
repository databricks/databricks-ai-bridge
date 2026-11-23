"""DatabricksMemoryStore: ops, paths, filters, search window, errors and the in-process fallback.

The real store drives the real AgentKitClient, MemoryStores, MemoryStore and Memory wrappers. Only
the workspace transport is faked (``_memory_api_fake``). Run-config and graph behaviour lives in
``langgraph_store_graph_test.py``.
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

pytest.importorskip("langgraph")

from _memory_api_fake import STORE, install  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.config import var_child_runnable_config  # noqa: E402

from databricks_agentbricks.errors import AgentCliError  # noqa: E402
from databricks_agentkit.langgraph import store as store_mod  # noqa: E402


@pytest.fixture
def api(monkeypatch):
    return install(monkeypatch)


def _store(actor="alice", **kwargs):
    return store_mod.DatabricksMemoryStore(store=STORE, actor=actor, **kwargs)


def _env(value, expires_at_ms=None):
    """An entry's content as the store writes it."""
    return store_mod._wrap(value, expires_at_ms)


@contextmanager
def run_as(actor, **configurable):
    """Run the body inside a LangGraph run config for ``actor`` (thread ``s1``)."""
    config: RunnableConfig = {
        "configurable": {"thread_id": "s1", "actor_id": actor, **configurable}
    }
    token = var_child_runnable_config.set(config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


# --- paths and keys ----------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "segment"),
    [(".", "%2E"), ("..", "%2E%2E"), ("a/b", "a%2Fb"), ("%x", "%25x"), ("café", "caf%C3%A9")],
)
def test_keys_round_trip_as_percent_encoded_paths(api, key, segment):
    store = _store()
    store.put(("ns",), key, {"v": key})

    assert [row["path"] for row in api.rows()] == [f"/langgraph/ns/{segment}"]
    assert store.get(("ns",), key).value == {"v": key}
    assert [(item.namespace, item.key) for item in store.search(("ns",))] == [(("ns",), key)]


def test_namespace_labels_with_slash_round_trip(api):
    store = _store()
    namespace = ("a/b", "c d")
    store.put(namespace, "k", {"v": 1})

    assert [row["path"] for row in api.rows()] == ["/langgraph/a%2Fb/c%20d/k"]
    assert store.get(namespace, "k").value == {"v": 1}
    [hit] = store.search(("a/b",))
    assert (hit.namespace, hit.key) == (namespace, "k")
    assert store.list_namespaces() == [namespace]


def test_rejected_inputs_raise_without_touching_the_service(api):
    with pytest.raises(ValueError, match="key must be a non-empty string"):
        _store().put(("ns",), "", {"v": 1})
    with pytest.raises(ValueError, match="root must start with"):
        store_mod.DatabricksMemoryStore(store=STORE, actor="alice", root="/x/")
    for ttl in (0, -1, True, "5"):
        with pytest.raises(ValueError, match="ttl must be a positive number of minutes"):
            _store().put(("ns",), "k", {"v": 1}, ttl=ttl)
    with pytest.raises(NotImplementedError, match="index"):
        _store().put(("ns",), "k", {"v": 1}, index=["text"])
    assert api.calls == []


# --- put, get, overwrite, delete -----------------------------------------------


def test_put_get_and_overwrite_take_the_create_conflict_then_update_path(api):
    store = _store()
    store.put(("ns",), "k", {"msg": "héllo", "n": 1})
    item = store.get(("ns",), "k")

    assert (item.namespace, item.key, item.value) == (("ns",), "k", {"msg": "héllo", "n": 1})
    [row] = api.rows()
    assert row["content"] == '{"lg":1,"v":{"msg":"héllo","n":1}}'
    assert row["actor_id"] == "alice:langgraph"

    api.calls.clear()
    store.put(("ns",), "k", {"msg": "v2"})

    # The create hits ALREADY_EXISTS, then the store looks the entry up and updates it.
    assert api.routes() == ["create", "list", "update"]
    assert store.get(("ns",), "k").value == {"msg": "v2"}
    assert len(api.rows()) == 1


def test_description_is_the_text_of_the_string_values(api):
    store = _store()
    store.put(
        ("ns",), "k", {"title": "Trip", "tags": ["paris", {"note": "  spaced\n out "}], "n": 3}
    )
    store.put(("ns",), "long", {"text": "x" * 600})

    descriptions = {row["path"].rsplit("/", 1)[-1]: row["description"] for row in api.rows()}
    assert descriptions == {"k": "Trip paris spaced out", "long": "x" * 500}


def test_delete_removes_only_the_exact_key_and_a_missing_key_is_a_noop(api):
    store = _store()
    store.put(("ns",), "k", {"v": 1})
    store.put(("ns",), "k2", {"v": 2})

    store.put(("ns",), "k", None)
    assert [row["path"] for row in api.rows()] == ["/langgraph/ns/k2"]
    assert store.get(("ns",), "k") is None
    assert store.get(("ns",), "k2").value == {"v": 2}

    api.calls.clear()
    store.put(("ns",), "never-written", None)
    assert api.routes() == ["list"]  # looked up, found nothing, deleted nothing


@pytest.mark.parametrize(
    ("error_code", "status", "commit", "expect_error"),
    [
        # The entry was already removed (a retried or raced delete): the store treats it as done.
        pytest.param("NOT_FOUND", 404, True, False, id="not-found-is-noop"),
        # Any other error reaches the caller, and the entry is still there.
        pytest.param("PERMISSION_DENIED", 403, False, True, id="other-errors-propagate"),
    ],
)
def test_delete_treats_not_found_as_done_and_propagates_other_errors(
    api, error_code, status, commit, expect_error
):
    store = _store()
    store.put(("ns",), "k", {"v": 1})
    api.fail("delete", error_code, status=status, commit=commit)

    if not expect_error:
        store.put(("ns",), "k", None)
        assert store.get(("ns",), "k") is None
        return
    with pytest.raises(AgentCliError) as exc:
        store.put(("ns",), "k", None)
    assert exc.value.error_code == error_code
    assert store.get(("ns",), "k").value == {"v": 1}


def test_item_times_come_from_the_server(api):
    api.seed(
        "/langgraph/ns/k",
        _env({"v": 1}),
        create_time="2026-01-02T03:04:05.000000Z",
        update_time="2026-02-03T04:05:06.000000Z",
    )
    item = _store().get(("ns",), "k")

    assert item.created_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert item.updated_at == datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc)


# --- what the store must never return ----------------------------------------


@pytest.mark.parametrize("ignore_filters", [False, True], ids=["server-filters", "server-ignores"])
def test_session_foreign_and_non_object_entries_are_invisible(api, ignore_filters):
    api.ignore_filters = ignore_filters
    tea = _env({"color": "tea"})
    api.seed("/langgraph/ns/k", tea)
    api.seed("/langgraph/ns/sess", tea, session_id="s1")
    api.seed("/langgraph/ns/fact", "User likes tea")  # plain text, eg rewritten by a pipeline
    api.seed("/langgraph/ns/bare", '{"color":"tea"}')  # a JSON object without the envelope
    api.seed("/langgraph/ns/arr", '["tea"]')
    api.seed("/langgraph/ns/num", "1")
    api.seed("/langgraph/ns/empty", None)
    api.seed("/langgraph/ns/extra", '{"lg":1,"v":{"color":"tea"},"note":"tea"}')
    api.seed("/langgraph/ns/badx", '{"lg":1,"v":{"color":"tea"},"x":"soon"}')
    api.seed("/langgraph/ns/boolv", '{"lg":true,"v":{"color":"tea"}}')
    api.seed("/langgraph/ns/deep", "[" * 200_000 + "]" * 200_000)  # json.loads recursion
    api.seed("/other/ns/x", tea)  # outside the store's root
    api.seed("/langgraph/sessonly/x", tea, session_id="s1")  # only a session
    api.seed("/langgraph/ns/plain", tea, actor_id="alice")  # the plain actor: not the store's
    store = _store()

    assert store.get(("ns",), "k").value == {"color": "tea"}
    hidden = ("sess", "fact", "bare", "arr", "num", "empty", "extra", "badx", "boolv", "deep")
    for key in hidden + ("plain",):
        assert store.get(("ns",), key) is None
    assert [item.key for item in store.search(("ns",))] == ["k"]
    assert [item.key for item in store.search(("ns",), query="tea")] == ["k"]
    assert store.list_namespaces() == [("ns",)]


# --- search without a query -----------------------------------------------------


def test_search_without_query_returns_the_prefix_with_exact_offset_and_limit(api):
    store = _store()
    for i in range(5):
        store.put(("ns", "sub"), f"k{i}", {"i": i})
    store.put(("ns",), "x", {"i": 9})
    store.put(("nsx",), "y", {"i": 8})  # a sibling namespace must not match the prefix

    assert sorted(item.key for item in store.search(("ns",), limit=10)) == [
        "k0",
        "k1",
        "k2",
        "k3",
        "k4",
        "x",
    ]
    assert [item.key for item in store.search(("ns", "sub"), limit=3, offset=1)] == [
        "k1",
        "k2",
        "k3",
    ]


def test_search_without_query_follows_the_server_pages(api):
    api.server_page_size = 2
    store = _store()
    for i in range(5):
        store.put(("ns",), f"k{i}", {"i": i})
    api.calls.clear()

    assert [item.key for item in store.search(("ns",), limit=10)] == [
        "k0",
        "k1",
        "k2",
        "k3",
        "k4",
    ]
    assert [call.query.get("page_token") for call in api.calls] == [None, "2", "4"]


@pytest.mark.parametrize(
    ("filter_", "expected"),
    [
        ({"score": {"$gt": 15}}, ["c"]),
        ({"score": {"$gte": 15}}, ["b", "c"]),
        ({"score": {"$lt": 15}}, ["a"]),
        ({"score": {"$lte": 15}}, ["a", "b"]),
        ({"score": {"$ne": 15}}, ["a", "c", "d"]),  # "high" is not a number, but $ne still matches
        ({"score": {"$gte": 15}, "kind": "y"}, ["c"]),  # every key must match, not any
    ],
    ids=["gt", "gte", "lt", "lte", "ne", "multi-key-and"],
)
def test_filter_operators_and_boundaries_on_search_without_query(api, filter_, expected):
    store = _store()
    store.put(("ns",), "a", {"score": 10, "kind": "x"})
    store.put(("ns",), "b", {"score": 15, "kind": "x"})
    store.put(("ns",), "c", {"score": 20, "kind": "y"})
    store.put(("ns",), "d", {"score": "high", "kind": "y"})

    assert sorted(item.key for item in store.search(("ns",), filter=filter_)) == expected


# --- search with a query --------------------------------------------------------


def test_query_search_matches_whole_words_and_stays_within_the_actor(api):
    alice, bob = _store("alice"), _store("bob")
    alice.put(("ns",), "tea-house", {"title": "Tea House"})
    alice.put(("ns",), "tealeaf", {"note": "tealeaf"})
    alice.put(("ns",), "green", {"note": "green TEA"})
    bob.put(("ns",), "bobs", {"note": "tea"})

    hits = alice.search(("ns",), query="tea")
    assert {item.key: item.score for item in hits} == {"tea-house": 1.0, "green": 1.0}
    assert [item.key for item in bob.search(("ns",), query="tea")] == ["bobs"]


def test_query_search_passes_the_path_prefix_so_other_namespaces_do_not_use_the_window(api):
    for i in range(150):
        api.seed(f"/langgraph/other/k{i}", _env({"note": "tea"}))
    api.seed("/langgraph/ns/mine", _env({"note": "tea"}))

    results = _store().search(("ns",), query="tea", limit=5)

    assert [item.key for item in results] == ["mine"]
    [search] = [call for call in api.calls if call.route == "search"]
    assert search.body["path_prefix"] == "/langgraph/ns/"
    assert search.body["page_size"] == 55  # offset + limit + 50 headroom


def test_query_search_with_a_filter_requests_the_full_window_and_warns_when_it_is_short(
    api, caplog
):
    store = _store()
    for i in range(10):
        store.put(("ns",), f"blue{i}", {"color": "blue", "note": "tea"})
    for i in range(60):  # these rank first but fail the filter
        store.put(("ns",), f"red{i}", {"color": "red", "note": "tea tea tea"})

    results = store.search(("ns",), query="tea", filter={"color": "blue"}, limit=10)

    assert sorted(item.key for item in results) == sorted(f"blue{i}" for i in range(10))
    [search] = [call for call in api.calls if call.route == "search"]
    assert search.body["page_size"] == 100

    # 120 matches, none passing the filter: the window of 100 is full and the page is short.
    for i in range(120):
        store.put(("full",), f"red{i}", {"color": "red", "note": "tea"})
    with caplog.at_level(logging.WARNING, logger=store_mod.logger.name):
        assert store.search(("full",), query="tea", filter={"color": "blue"}, limit=10) == []
    warned = [r for r in caplog.records if "reached the most results" in r.getMessage()]
    assert len(warned) == 1

    # 70 matches, no filter: the window of 60 covers the page, so there is nothing to warn about.
    caplog.clear()
    for i in range(70):
        store.put(("quiet",), f"q{i}", {"note": "tea"})
    with caplog.at_level(logging.WARNING, logger=store_mod.logger.name):
        assert len(store.search(("quiet",), query="tea", limit=10)) == 10
    assert not [r for r in caplog.records if "reached the most results" in r.getMessage()]


# --- list_namespaces --------------------------------------------------------------


def test_list_namespaces_by_prefix_suffix_and_wildcards(api):
    store = _store()
    for namespace in [("a", "x"), ("b", "x"), ("a", "y", "z"), ("c",)]:
        store.put(namespace, "k", {"v": 1})

    assert store.list_namespaces(prefix=("a",)) == [("a", "x"), ("a", "y", "z")]
    assert store.list_namespaces(suffix=("x",)) == [("a", "x"), ("b", "x")]
    assert store.list_namespaces(prefix=("*", "x")) == [("a", "x"), ("b", "x")]
    assert store.list_namespaces(suffix=("*", "z")) == [("a", "y", "z")]


def test_list_namespaces_truncates_depth_dedupes_and_pages(api):
    store = _store()
    for namespace in [("a", "b"), ("a", "b", "c"), ("a", "d")]:
        store.put(namespace, "k", {"v": 1})
    for i in range(5):
        store.put(("ns", str(i)), "k", {"v": i})

    assert store.list_namespaces(prefix=("a",), max_depth=1) == [("a",)]
    assert store.list_namespaces(prefix=("a",), max_depth=2) == [("a", "b"), ("a", "d")]
    assert store.list_namespaces(prefix=("ns",), limit=2, offset=1) == [("ns", "1"), ("ns", "2")]
    assert store.list_namespaces(prefix=("ns",), limit=2, offset=4) == [("ns", "4")]


# --- service handle, factory and fallback --------------------------------------------


def test_client_is_built_once_on_first_use(api, monkeypatch):
    built = []

    class CountingClient(store_mod.AgentKitClient):
        def __init__(self, *args, **kwargs):
            built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(store_mod, "AgentKitClient", CountingClient)
    store = _store()
    assert built == []

    store.put(("ns",), "k", {"v": 1})
    store.get(("ns",), "k")
    store.search(("ns",))

    assert built == [1]


def test_memory_store_factory_uses_the_explicit_id_then_the_environment(api, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_STORE", "env-store")

    store_mod.memory_store(actor="alice").put(("ns",), "k", {"v": 1})
    store_mod.memory_store("explicit-store", actor="alice").put(("ns",), "j", {"v": 2})

    creates = [call for call in api.calls if call.route == "create"]
    assert [(call.store, call.actor_id) for call in creates] == [
        ("env-store", "alice:langgraph"),
        ("explicit-store", "alice:langgraph"),
    ]


def test_memory_store_without_a_store_is_in_process_partitioned_by_actor_and_strict(
    monkeypatch, caplog, api
):
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    store = store_mod.memory_store()
    with caplog.at_level(logging.WARNING, logger=store_mod.logger.name):
        with run_as("alice"):
            store.put(("ns",), "k", {"v": "alice"})
    assert "No memory store configured" in caplog.text  # warned on first use, not at import

    with run_as("bob"):
        assert store.get(("ns",), "k") is None  # bob can't read alice's entry
    with run_as("alice"):
        item = store.get(("ns",), "k")
        assert item is not None and item.value == {"v": "alice"}

    strict = store_mod.memory_store(strict_user_key="langgraph_user_id")
    with run_as("alice", langgraph_user_id="bob"):
        with pytest.raises(ValueError, match="refusing to read or write another user's memory"):
            strict.put(("ns",), "k", {"v": "x"})
    with run_as("alice", langgraph_user_id="alice"):
        strict.put(("ns",), "k", {"v": "strict"})
        item = strict.get(("ns",), "k")
        assert item is not None and item.value == {"v": "strict"}

    assert api.calls == []  # the fallback never calls the service


def test_value_over_the_size_cap_is_rejected_before_any_call(api):
    store = _store()
    cap = store_mod._REQUEST_MAX_BYTES
    # The request carries the envelope as a JSON string plus a 500-character description.
    base = len(json.dumps({"content": _env({"blob": ""}), "description": "x" * 500}))

    store.put(("ns",), "at-cap", {"blob": "x" * (cap - base)})
    assert api.routes() == ["create"]

    api.calls.clear()
    with pytest.raises(ValueError, match="byte request"):
        store.put(("ns",), "over", {"blob": "x" * (cap - base + 1)})
    assert api.calls == []


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("\u5317" * 250_000, id="non-ascii-escaped-on-the-wire"),
        pytest.param('"' * 300_000, id="quotes-escaped-twice"),
    ],
)
def test_the_size_cap_counts_the_escaped_request_not_the_raw_value(api, text):
    assert len(_env({"t": text}).encode()) < store_mod._REQUEST_MAX_BYTES  # small as raw bytes
    with pytest.raises(ValueError, match="byte request"):
        _store().put(("ns",), "k", {"t": text})
    assert api.calls == []


def test_list_namespaces_ignores_namespaces_with_only_foreign_entries(api):
    api.seed("/langgraph/ghost/fact", "User likes tea")

    assert _store().list_namespaces() == []


# --- known bugs (strict xfail: they flip to XPASS when fixed) ----------------------


@pytest.mark.xfail(
    strict=True,
    raises=AgentCliError,
    reason=(
        "bug: put(value) raises NOT_FOUND when the entry is deleted between the lookup and the "
        "update. MemoryStore._put_memory in memory_store.py retries only the create, so the "
        "update sits outside the retry loop."
    ),
)
def test_put_after_a_concurrent_delete_still_writes_the_value(api):
    store = _store()
    store.put(("ns",), "k", {"v": 1})
    api.hooks["update"] = lambda name: api.entries.pop(name, None)

    store.put(("ns",), "k", {"v": 2})

    assert store.get(("ns",), "k").value == {"v": 2}
