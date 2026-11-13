"""DatabricksMemoryStore edge cases: clocks, envelopes, filters, the search window, actors and inputs."""

import time

import pytest

pytest.importorskip("langgraph")

from _memory_api_fake import STORE, install  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.config import var_child_runnable_config  # noqa: E402

from databricks_agentbricks.errors import AgentCliError  # noqa: E402
from databricks_agentkit._actors import check_actor  # noqa: E402
from databricks_agentkit.langgraph import store as store_mod  # noqa: E402

T0 = 1_800_000_000_000
MINUTE = 60_000


@pytest.fixture
def api(monkeypatch):
    return install(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(store_mod, "_warned", set())


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


def _store(actor="alice", clock=None, **kwargs):
    store = store_mod.DatabricksMemoryStore(store=STORE, actor=actor, **kwargs)
    if clock is not None:
        store._now_ms = clock
    return store


def _keys(items):
    return sorted(item.key for item in items)


def test_default_clock_is_epoch_milliseconds():
    before = time.time() * 1000
    now = store_mod._now_ms()
    assert abs(now - before) < 60_000


def test_an_empty_dict_is_stored_not_deleted(api):
    store = _store()
    store.put(("ns",), "k", {})
    assert store.get(("ns",), "k").value == {}
    assert len(api.rows()) == 1


@pytest.mark.parametrize("content", ['{"lg":1,"v":"tea"}', '{"lg":1}', '{"lg":1,"v":[1]}'])
def test_envelopes_without_an_object_value_are_skipped(api, content):
    api.seed("/langgraph/ns/k", content)
    store = _store()
    assert store.get(("ns",), "k") is None
    assert store.search(("ns",)) == []


def test_nested_and_list_filters_compare_structurally(api):
    store = _store()
    store.put(("ns",), "en", {"meta": {"lang": "en"}, "tags": ["a", "b"]})
    store.put(("ns",), "str", {"meta": "en", "tags": ["a", "b"]})
    store.put(("ns",), "fr", {"meta": {"lang": "fr"}, "tags": ["a"]})
    store.put(("ns",), "partial", {"meta": {"lang": "x"}, "tags": ["a", "x"]})

    assert _keys(store.search(("ns",), filter={"meta": {"lang": "en"}})) == ["en"]
    assert _keys(store.search(("ns",), filter={"tags": ["a", "b"]})) == [
        "en",
        "str",
    ]  # not "partial"
    assert _keys(store.search(("ns",), filter={"tags": ["a"]})) == ["fr"]


def test_query_search_warns_when_expired_entries_fill_the_service_window(api, caplog):
    import logging

    clock = Clock()
    store = _store(clock=clock)
    for i in range(70):
        store.put(("ns",), f"live{i}", {"note": "tea"})
    for i in range(30):
        store.put(("ns",), f"old{i}", {"note": "tea"}, ttl=1)
    clock.now += 2 * MINUTE

    with caplog.at_level(logging.WARNING, logger=store_mod.logger.name):
        assert len(store.search(("ns",), query="tea", limit=100)) == 70
    assert len([r for r in caplog.records if "reached the most results" in r.getMessage()]) == 1


def test_query_search_does_not_warn_on_an_exactly_full_filtered_page(api, caplog):
    import logging

    store = _store()
    for i in range(90):
        store.put(("ns",), f"red{i}", {"color": "red", "note": "tea"})
    for i in range(10):
        store.put(("ns",), f"blue{i}", {"color": "blue", "note": "tea"})

    with caplog.at_level(logging.WARNING, logger=store_mod.logger.name):
        assert len(store.search(("ns",), query="tea", filter={"color": "blue"}, limit=10)) == 10
    assert not [r for r in caplog.records if "reached the most results" in r.getMessage()]


def test_query_search_requests_offset_plus_limit_plus_headroom(api):
    store = _store()
    store.put(("ns",), "k", {"note": "tea"})
    api.calls.clear()
    store.search(("ns",), query="tea", limit=5, offset=20)
    [search] = [call for call in api.calls if call.route == "search"]
    assert search.body["page_size"] == 75


def test_a_store_without_an_id_raises_on_first_use(api, monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    with pytest.raises(ValueError, match="No memory store configured"):
        store_mod.DatabricksMemoryStore(actor="alice").put(("ns",), "k", {"v": 1})
    assert api.calls == []


@pytest.mark.parametrize(
    ("code", "commit", "expect_error", "count"),
    [
        pytest.param("PERMISSION_DENIED", False, True, None, id="other-error-propagates"),
        pytest.param("NOT_FOUND", True, False, 0, id="not-found-is-done"),
    ],
)
def test_sweep_propagates_delete_errors_except_not_found(api, code, commit, expect_error, count):
    clock = Clock()
    store = _store(clock=clock)
    store.put(("ns",), "k", {"v": 1}, ttl=1)
    clock.now += 2 * MINUTE
    api.fail("delete", code, status=403 if code != "NOT_FOUND" else 404, commit=commit)

    if expect_error:
        with pytest.raises(AgentCliError):
            store.sweep_ttl()
        assert len(api.rows()) == 1
    else:
        assert store.sweep_ttl() == count


def test_search_excludes_a_sibling_namespace_when_the_server_ignores_the_prefix(api):
    api.ignore_filters = True
    store = _store()
    store.put(("ns",), "k", {"note": "tea"})
    store.put(("nsx",), "y", {"note": "tea"})

    assert [item.namespace for item in store.search(("ns",), query="tea")] == [("ns",)]
    assert _keys(store.search(("ns",))) == ["k"]


def test_an_empty_run_actor_is_refused(api):
    config: RunnableConfig = {"configurable": {"thread_id": "s1", "actor_id": ""}}
    token = var_child_runnable_config.set(config)
    try:
        with pytest.raises(ValueError, match="needs an actor"):
            store_mod.DatabricksMemoryStore(store=STORE).put(("ns",), "k", {"v": 1})
    finally:
        var_child_runnable_config.reset(token)
    assert api.calls == []


@pytest.mark.parametrize("actor", ["alice-langgraph", "langgraph", "alice:langgraph2"])
def test_actors_that_only_resemble_the_suffix_are_accepted(actor):
    assert check_actor(actor) == actor


def test_paths_with_empty_segments_are_not_read_as_items(api):
    api.seed("/langgraph/ns//k", store_mod._wrap({"color": "tea"}, None))
    store = _store()
    assert store.search(("ns",)) == []
    assert store.list_namespaces() == []


def test_in_process_search_ranks_by_shared_terms_and_drops_non_matches(monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    store = store_mod.DatabricksMemoryStore._local(actor="alice")
    store.put(("ns",), "one", {"note": "tea"})
    store.put(("ns",), "both", {"note": "tea and coffee"})
    store.put(("ns",), "none", {"note": "water"})

    assert [item.key for item in store.search(("ns",), query="tea coffee")] == ["both", "one"]


# --- inputs the service would reject -------------------------------------------------------


@pytest.mark.parametrize("value", [["tea"], "tea", 42, ("a", "b")])
def test_a_value_that_is_not_a_dict_is_refused_before_any_call(api, value):
    with pytest.raises(TypeError, match="must be a dict"):
        _store().put(("ns",), "k", value)
    assert api.calls == []


@pytest.mark.parametrize(
    ("value", "error"),
    [
        pytest.param({"n": float("nan")}, ValueError, id="nan"),
        pytest.param({"n": float("inf")}, ValueError, id="infinity"),
        pytest.param({"when": time.gmtime(0)}, None, id="struct-time-is-a-tuple"),
        pytest.param({"blob": b"x"}, TypeError, id="bytes"),
        pytest.param({"set": {1, 2}}, TypeError, id="set"),
    ],
)
def test_values_must_be_strict_json(api, value, error):
    if error is None:
        _store().put(("ns",), "k", value)
        return
    with pytest.raises(error, match="for key 'k'"):
        _store().put(("ns",), "k", value)
    assert api.calls == []


def test_an_over_long_actor_and_path_is_refused_before_any_call(api):
    with pytest.raises(ValueError, match="take .* bytes together"):
        _store().put(("ns",), "k" * 2_400, {"v": 1})
    assert api.calls == []


def test_a_key_that_is_not_valid_unicode_is_refused(api):
    with pytest.raises(ValueError, match="isn't valid Unicode"):
        _store().put(("ns",), "\ud800", {"v": 1})
    assert api.calls == []


def test_nul_characters_are_kept_in_the_value_but_not_in_the_description(api):
    store = _store()
    store.put(("ns",), "k", {"note": "a\x00b"})
    [create] = [call for call in api.calls if call.route == "create"]
    assert "\x00" not in create.body["description"]
    assert store.get(("ns",), "k").value == {"note": "a\x00b"}


def test_a_blank_query_lists_instead_of_searching(api):
    store = _store()
    store.put(("ns",), "k", {"note": "tea"})
    api.calls.clear()
    assert _keys(store.search(("ns",), query="   ")) == ["k"]
    assert api.routes() == ["list"]


def test_a_search_without_a_query_stops_reading_once_the_page_is_full(api):
    store = _store()
    for i in range(250):
        store.put(("ns",), f"k{i:03}", {"n": i})
    api.calls.clear()

    assert len(store.search(("ns",), limit=10)) == 10
    assert api.routes() == ["list"]  # one page of 100, not all three


# --- explicit actors -----------------------------------------------------------------------


@pytest.mark.parametrize("actor", ["", 7])
def test_an_explicit_actor_must_be_a_non_empty_string(api, actor):
    with pytest.raises(ValueError, match="non-empty string"):
        store_mod.DatabricksMemoryStore(store=STORE, actor=actor)
    with pytest.raises(ValueError, match="non-empty string"):
        _store().sweep_ttl(actor=actor)
    assert api.calls == []


# --- TTL configuration ---------------------------------------------------------------------


def test_reads_never_refresh_so_the_ttl_config_says_so():
    store = _store(ttl={"default_ttl": 5})
    assert store.ttl_config == {"default_ttl": 5, "refresh_on_read": False}


@pytest.mark.parametrize(
    ("ttl", "match"),
    [
        pytest.param({"default_tll": 5}, "Unknown ttl keys", id="typo"),
        pytest.param({"default_ttl": float("inf")}, "positive number", id="infinite"),
        pytest.param({"default_ttl": float("nan")}, "positive number", id="nan"),
        pytest.param({"default_ttl": 10**12}, "at most", id="absurd"),
        pytest.param({"default_ttl": 1e-9}, "one millisecond", id="rounds-to-zero"),
    ],
)
def test_bad_ttl_configs_are_refused(ttl, match):
    with pytest.raises(ValueError, match=match):
        _store(ttl=ttl)


# --- no store bound ------------------------------------------------------------------------


def test_memory_store_resolves_the_store_on_first_use_not_when_built(api, monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    store = store_mod.memory_store(actor="alice")  # eg built at import, before the env is set
    monkeypatch.setenv("AGENT_MEMORY_STORE", STORE)
    store.put(("ns",), "k", {"v": 1})
    assert api.routes() == ["create"]


def test_memory_store_in_a_deployed_app_without_a_store_raises(api, monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    monkeypatch.setenv("DATABRICKS_APP_NAME", "my-agent")
    with pytest.raises(ValueError, match="A deployed App needs a store"):
        store_mod.memory_store(actor="alice").put(("ns",), "k", {"v": 1})
    assert api.calls == []
