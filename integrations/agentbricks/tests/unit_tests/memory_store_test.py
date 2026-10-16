"""Tests for typed managed-memory resources."""

import warnings
from datetime import datetime, timezone
from typing import Optional

import pytest
from resource_test_fixtures import (
    MEMORY_ID,
    MEMORY_STORE_NAME,
    STORE_ID,
    mem_store_payload,
    memory_payload,
    resource_client,
)

from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit import MemoryStore


def test_create_parses_epoch_millis_timestamps() -> None:
    client, api = resource_client()
    api.create_memory_store.return_value = mem_store_payload()

    store = client.memory_stores.create(display_name="coding_agent_memory", description="d")

    assert isinstance(store, MemoryStore)
    assert store.id == STORE_ID
    assert store.create_time == datetime.fromtimestamp(1770000000, tz=timezone.utc)
    assert store.update_time == datetime.fromtimestamp(1770000600, tz=timezone.utc)
    api.create_memory_store.assert_called_once_with("coding_agent_memory", "d")


def test_list_stores_consumes_pages() -> None:
    client, api = resource_client()
    api.list_memory_stores.side_effect = [
        {
            "managed_memory_stores": [mem_store_payload(display_name="first")],
            "next_page_token": "p2",
        },
        {"managed_memory_stores": [mem_store_payload(display_name="second")]},
    ]

    stores = client.memory_stores.list(page_size=25)

    assert [store.display_name for store in stores] == ["first", "second"]
    assert api.list_memory_stores.call_args_list[1].kwargs == {
        "page_size": 25,
        "page_token": "p2",
    }


def test_get_by_id() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()

    store = client.memory_stores.get(STORE_ID)

    assert store.id == STORE_ID
    api.get_memory_store.assert_called_once_with(STORE_ID)


@pytest.mark.parametrize("store_id", [STORE_ID, MEMORY_STORE_NAME])
def test_handle_addresses_a_store_without_fetching_it(store_id: str) -> None:
    client, api = resource_client()

    store = client.memory_stores._handle(store_id)

    assert store.id == STORE_ID
    assert store.name == MEMORY_STORE_NAME
    api.get_memory_store.assert_not_called()


def test_bound_store_update_and_delete() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.update_memory_store.return_value = mem_store_payload(description="updated")
    store = client.memory_stores.get(STORE_ID)

    updated = store.update(description="updated")
    updated.delete()

    assert updated.description == "updated"
    api.update_memory_store.assert_called_once_with(
        STORE_ID,
        display_name=None,
        description="updated",
    )
    api.delete_memory_store.assert_called_once_with(STORE_ID)


def test_grant_permission() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    store = client.memory_stores.get(STORE_ID)

    store.grant_permission("sp-123", permission="READ")

    api.grant_memory_store_permission.assert_called_once_with(STORE_ID, "sp-123", permission="READ")


def test_bound_store_add_memory() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.create_memory_entry.return_value = memory_payload()
    store = client.memory_stores.get(STORE_ID)

    memory = store.add(
        actor_id="alice",
        session_id="s1",
        path="/m/p.md",
        content="c",
        description="desc",
    )

    assert memory.id == MEMORY_ID
    api.create_memory_entry.assert_called_once_with(
        STORE_ID,
        "alice",
        "/m/p.md",
        content="c",
        description="desc",
        session_id="s1",
        source_type=None,
    )


def test_add_without_write_mode_does_not_warn() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.create_memory_entry.return_value = memory_payload()
    store = client.memory_stores.get(STORE_ID)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        store.add(actor_id="alice", path="/m/p.md", content="c")


def _already_exists() -> AgentCliError:
    return AgentCliError("already exists", error_code="ALREADY_EXISTS")


def _store_with_conflict(*existing: dict, create_side_effect=None):
    """A store whose create conflicts and whose path lookup returns ``existing`` on one page."""
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.create_memory_entry.side_effect = create_side_effect or _already_exists()
    api.list_memory_entries.return_value = {"managed_memory_entries": list(existing)}
    api.update_memory_entry.side_effect = lambda store_id, memory_id, **kwargs: memory_payload(
        name=f"{MEMORY_STORE_NAME}/entries/{memory_id}", content=kwargs["content"] or "unchanged"
    )
    return client.memory_stores.get(STORE_ID), api


def test_put_creates_when_path_is_free() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.create_memory_entry.return_value = memory_payload(content="fact")
    store = client.memory_stores.get(STORE_ID)

    memory = store.put(actor_id="alice", path="/m/p.md", content="fact", session_id="s1")

    assert memory.content == "fact"
    api.create_memory_entry.assert_called_once_with(
        STORE_ID,
        "alice",
        "/m/p.md",
        content="fact",
        description=None,
        session_id="s1",
        source_type=None,
    )
    api.list_memory_entries.assert_not_called()
    api.update_memory_entry.assert_not_called()


def test_put_overwrites_existing_and_keeps_description_when_omitted() -> None:
    store, api = _store_with_conflict(memory_payload(session_id=None, content="old"))

    memory = store.put(actor_id="alice", path="/m/p.md", content="new")

    assert memory.content == "new"
    api.list_memory_entries.assert_called_once_with(
        STORE_ID,
        "alice",
        path_prefix=None,
        session_id=None,
        page_size=None,
        page_token=None,
        read_mask=None,
        path="/m/p.md",
    )
    api.update_memory_entry.assert_called_once_with(
        STORE_ID, MEMORY_ID, content="new", description=None
    )


def test_put_overwrite_replaces_description_when_given() -> None:
    store, api = _store_with_conflict(memory_payload(session_id=None, content="old"))

    store.put(actor_id="alice", path="/m/p.md", content="new", description="fresh")

    api.update_memory_entry.assert_called_once_with(
        STORE_ID, MEMORY_ID, content="new", description="fresh"
    )


@pytest.mark.parametrize(
    ("existing", "expected"),
    [("first", "first\nsecond"), ("", "second")],
    ids=["joins-with-newline", "onto-empty-content"],
)
def test_put_append_matches_server_append(existing: str, expected: str) -> None:
    store, api = _store_with_conflict(memory_payload(session_id=None, content=existing))

    store.put(actor_id="alice", path="/m/p.md", content="second", append=True)

    api.update_memory_entry.assert_called_once_with(
        STORE_ID, MEMORY_ID, content=expected, description=None
    )


@pytest.mark.parametrize("content", ["second", "second\nthird"], ids=["line", "line-run"])
def test_put_append_skips_content_already_present(content: str) -> None:
    store, api = _store_with_conflict(
        memory_payload(session_id=None, content="first\nsecond\nthird")
    )

    memory = store.put(actor_id="alice", path="/m/p.md", content=content, append=True)

    assert memory.id == MEMORY_ID
    assert memory.content == "first\nsecond\nthird"
    api.update_memory_entry.assert_not_called()


def test_put_append_does_not_treat_a_partial_line_as_present() -> None:
    store, api = _store_with_conflict(memory_payload(session_id=None, content="seconds later"))

    store.put(actor_id="alice", path="/m/p.md", content="second", append=True)

    api.update_memory_entry.assert_called_once_with(
        STORE_ID, MEMORY_ID, content="seconds later\nsecond", description=None
    )


@pytest.mark.parametrize(("session_id", "expected_id"), [(None, "no-session"), ("s1", "in-s1")])
def test_put_updates_only_the_entry_in_the_requested_session(
    session_id: Optional[str], expected_id: str
) -> None:
    # The path filter matches across sessions, so the SDK must pick the right one itself.
    store, api = _store_with_conflict(
        memory_payload(name=f"{MEMORY_STORE_NAME}/entries/other", session_id="other"),
        memory_payload(name=f"{MEMORY_STORE_NAME}/entries/in-s1", session_id="s1"),
        memory_payload(name=f"{MEMORY_STORE_NAME}/entries/no-session", session_id=None),
    )

    store.put(actor_id="alice", path="/m/p.md", content="new", session_id=session_id)

    api.update_memory_entry.assert_called_once_with(
        STORE_ID, expected_id, content="new", description=None
    )


def test_put_ignores_entries_at_other_paths() -> None:
    # A server that predates the path filter ignores it and returns all of the actor's entries.
    store, api = _store_with_conflict(
        memory_payload(name=f"{MEMORY_STORE_NAME}/entries/other-path", path="/m/other.md"),
        memory_payload(
            name=f"{MEMORY_STORE_NAME}/entries/other-path-no-session",
            path="/m/other.md",
            session_id=None,
        ),
        memory_payload(name=f"{MEMORY_STORE_NAME}/entries/match", path="/m/p.md"),
    )

    store.put(actor_id="alice", path="/m/p.md", content="new", session_id="s1")

    api.update_memory_entry.assert_called_once_with(
        STORE_ID, "match", content="new", description=None
    )


def test_put_does_not_update_a_same_session_entry_at_another_path() -> None:
    store, api = _store_with_conflict(
        memory_payload(name=f"{MEMORY_STORE_NAME}/entries/other-path", path="/m/other.md"),
        create_side_effect=[_already_exists(), _already_exists()],
    )

    with pytest.raises(AgentCliError, match="/m/p.md"):
        store.put(actor_id="alice", path="/m/p.md", content="new", session_id="s1")

    api.update_memory_entry.assert_not_called()


def test_put_lookup_follows_pages() -> None:
    store, api = _store_with_conflict()
    api.list_memory_entries.side_effect = [
        {"managed_memory_entries": [memory_payload(session_id="other")], "next_page_token": "p2"},
        {
            "managed_memory_entries": [
                memory_payload(name=f"{MEMORY_STORE_NAME}/entries/match", session_id=None)
            ]
        },
    ]

    store.put(actor_id="alice", path="/m/p.md", content="new")

    assert api.list_memory_entries.call_args_list[1].kwargs["page_token"] == "p2"
    api.update_memory_entry.assert_called_once_with(
        STORE_ID, "match", content="new", description=None
    )


def test_put_retries_create_once_when_the_conflicting_entry_disappears() -> None:
    store, api = _store_with_conflict(
        create_side_effect=[_already_exists(), memory_payload(content="new")]
    )

    memory = store.put(actor_id="alice", path="/m/p.md", content="new")

    assert memory.content == "new"
    assert api.create_memory_entry.call_count == 2
    api.update_memory_entry.assert_not_called()


def test_put_gives_up_after_one_retry() -> None:
    store, api = _store_with_conflict(create_side_effect=[_already_exists(), _already_exists()])

    with pytest.raises(AgentCliError, match="/m/p.md") as raised:
        store.put(actor_id="alice", path="/m/p.md", content="new")

    assert raised.value.error_code == "ABORTED"
    assert api.create_memory_entry.call_count == 2
    api.update_memory_entry.assert_not_called()


def test_put_propagates_other_create_errors() -> None:
    store, api = _store_with_conflict(
        create_side_effect=AgentCliError("denied", error_code="PERMISSION_DENIED")
    )

    with pytest.raises(AgentCliError, match="denied"):
        store.put(actor_id="alice", path="/m/p.md", content="new")

    api.list_memory_entries.assert_not_called()


@pytest.mark.parametrize(
    "write_mode", ["OVERWRITE", "overwrite", "MANAGED_MEMORY_ENTRY_WRITE_MODE_OVERWRITE"]
)
def test_add_with_overwrite_write_mode_warns_and_puts(write_mode: str) -> None:
    store, api = _store_with_conflict(memory_payload(session_id=None, content="old"))

    with pytest.warns(DeprecationWarning, match="put") as warned:
        store.add(actor_id="alice", path="/m/p.md", content="new", write_mode=write_mode)

    # Attributed to the caller, so the warning points at the code to migrate.
    assert warned[0].filename == __file__
    api.update_memory_entry.assert_called_once_with(
        STORE_ID, MEMORY_ID, content="new", description=None
    )


@pytest.mark.parametrize("write_mode", ["APPEND", "MANAGED_MEMORY_ENTRY_WRITE_MODE_APPEND"])
def test_add_with_append_write_mode_warns_and_appends(write_mode: str) -> None:
    store, api = _store_with_conflict(memory_payload(session_id=None, content="old"))

    with pytest.warns(DeprecationWarning, match="put"):
        store.add(actor_id="alice", path="/m/p.md", content="new", write_mode=write_mode)

    api.update_memory_entry.assert_called_once_with(
        STORE_ID, MEMORY_ID, content="old\nnew", description=None
    )


@pytest.mark.parametrize(
    "write_mode",
    [
        "EXCLUSIVE",
        "MANAGED_MEMORY_ENTRY_WRITE_MODE_EXCLUSIVE",
        "MANAGED_MEMORY_ENTRY_WRITE_MODE_UNSPECIFIED",
    ],
)
def test_add_with_exclusive_write_mode_warns_and_creates(write_mode: str) -> None:
    store, api = _store_with_conflict(memory_payload(session_id=None))

    with pytest.warns(DeprecationWarning, match="put"), pytest.raises(AgentCliError):
        store.add(actor_id="alice", path="/m/p.md", content="new", write_mode=write_mode)

    api.list_memory_entries.assert_not_called()
    api.update_memory_entry.assert_not_called()


def test_add_with_write_mode_creates_with_source_type_and_never_sends_write_mode() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.create_memory_entry.return_value = memory_payload()
    store = client.memory_stores.get(STORE_ID)

    with pytest.warns(DeprecationWarning):
        store.add(
            actor_id="alice",
            path="/m/p.md",
            content="c",
            source_type="MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT",
            write_mode="OVERWRITE",
        )

    api.create_memory_entry.assert_called_once_with(
        STORE_ID,
        "alice",
        "/m/p.md",
        content="c",
        description=None,
        session_id=None,
        source_type="MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT",
    )


def test_add_rejects_unknown_write_mode() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    store = client.memory_stores.get(STORE_ID)

    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="write_mode"):
        store.add(actor_id="alice", path="/m/p.md", content="c", write_mode="UPSERT")

    api.create_memory_entry.assert_not_called()


def test_list_memories_filters_by_exact_path() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.list_memory_entries.return_value = {"managed_memory_entries": [memory_payload()]}
    store = client.memory_stores.get(STORE_ID)

    memories = list(store.list(actor_id="alice", path="/m/p.md"))

    assert [memory.id for memory in memories] == [MEMORY_ID]
    assert api.list_memory_entries.call_args.kwargs["path"] == "/m/p.md"


def test_list_memories_consumes_pages() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.list_memory_entries.side_effect = [
        {"managed_memory_entries": [memory_payload()], "next_page_token": "p2"},
        {"managed_memory_entries": [memory_payload(name=f"{MEMORY_STORE_NAME}/entries/second")]},
    ]
    store = client.memory_stores.get(STORE_ID)

    memories = store.list(actor_id="alice", path_prefix="/m/", page_size=10)

    assert [memory.id for memory in memories] == [MEMORY_ID, "second"]
    assert api.list_memory_entries.call_args_list[1].kwargs["page_token"] == "p2"


def test_get_update_and_delete_memory() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.get_memory_entry.return_value = memory_payload()
    api.update_memory_entry.return_value = memory_payload(content="updated")
    store = client.memory_stores.get(STORE_ID)

    memory = store.get(MEMORY_ID, read_mask="name,content")
    updated = memory.update(content="updated")
    updated.delete()

    assert updated.content == "updated"
    api.get_memory_entry.assert_called_once_with(
        STORE_ID,
        MEMORY_ID,
        read_mask="name,content",
    )
    api.update_memory_entry.assert_called_once_with(
        STORE_ID,
        MEMORY_ID,
        content="updated",
        description=None,
    )
    api.delete_memory_entry.assert_called_once_with(STORE_ID, MEMORY_ID)


def test_search_preserves_scores_and_translates_limit() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.search_memory_entries.return_value = {
        "results": [{"managed_memory_entry": memory_payload(), "score": 0.9}]
    }
    store = client.memory_stores.get(STORE_ID)

    results = store.search(actor_id="alice", query="prefs", limit=10)

    assert results[0].memory.id == MEMORY_ID
    assert results[0].score == 0.9
    api.search_memory_entries.assert_called_once_with(
        STORE_ID,
        "alice",
        "prefs",
        page_size=10,
        path_prefix=None,
        session_id=None,
        read_mask=None,
    )


def test_search_supports_legacy_response_shape() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.search_memory_entries.return_value = {"managed_memory_entries": [memory_payload()]}
    store = client.memory_stores.get(STORE_ID)

    results = store.search(actor_id="alice", query="prefs", limit=10)

    assert results[0].memory.id == MEMORY_ID
    assert results[0].score is None


@pytest.mark.parametrize("limit", [0, 101])
def test_search_rejects_out_of_range_limit(limit: int) -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    store = client.memory_stores.get(STORE_ID)

    with pytest.raises(ValueError, match="limit"):
        store.search(actor_id="alice", query="q", limit=limit)
