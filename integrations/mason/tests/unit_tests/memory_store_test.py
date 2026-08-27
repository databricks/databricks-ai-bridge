"""Tests for typed managed-memory resources."""

from datetime import datetime, timezone

import pytest
from resource_test_fixtures import (
    ENTRY_ID,
    MEM_STORE_NAME,
    STORE_ID,
    entry_payload,
    mem_store_payload,
    resource_client,
)

from databricks_mason import ManagedMemoryStore


def test_create_parses_epoch_millis_timestamps() -> None:
    client, api = resource_client()
    api.create_memory_store.return_value = mem_store_payload()

    store = client.memory_stores.create(display_name="coding_agent_memory", description="d")

    assert isinstance(store, ManagedMemoryStore)
    assert store.store_id == STORE_ID
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

    store = client.memory_stores.get(store_id=STORE_ID)

    assert store.store_id == STORE_ID
    api.get_memory_store.assert_called_once_with(STORE_ID)


def test_get_by_display_name_consumes_pages() -> None:
    client, api = resource_client()
    api.list_memory_stores.side_effect = [
        {
            "managed_memory_stores": [mem_store_payload(display_name="other")],
            "next_page_token": "p2",
        },
        {"managed_memory_stores": [mem_store_payload()]},
    ]

    store = client.memory_stores.get(display_name="coding_agent_memory")

    assert store.store_id == STORE_ID
    assert api.list_memory_stores.call_count == 2


def test_get_or_create_creates_when_absent() -> None:
    client, api = resource_client()
    api.list_memory_stores.return_value = {"managed_memory_stores": []}
    api.create_memory_store.return_value = mem_store_payload(description="new")

    store = client.memory_stores.get_or_create(
        display_name="coding_agent_memory",
        description="new",
    )

    assert store.description == "new"
    api.create_memory_store.assert_called_once_with("coding_agent_memory", "new")


def test_get_validates_selectors() -> None:
    client, _ = resource_client()

    with pytest.raises(ValueError, match="exactly one"):
        client.memory_stores.get()
    with pytest.raises(ValueError, match="exactly one"):
        client.memory_stores.get(store_id=STORE_ID, display_name="x")


def test_get_raises_key_error_when_missing() -> None:
    client, api = resource_client()
    api.list_memory_stores.return_value = {"managed_memory_stores": []}

    with pytest.raises(KeyError, match="missing"):
        client.memory_stores.get(display_name="missing")


def test_bound_store_update_and_delete() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.update_memory_store.return_value = mem_store_payload(description="updated")
    store = client.memory_stores.get(store_id=STORE_ID)

    updated = store.update(description="updated")
    updated.delete()

    assert updated.description == "updated"
    api.update_memory_store.assert_called_once_with(
        STORE_ID,
        display_name=None,
        description="updated",
    )
    api.delete_memory_store.assert_called_once_with(STORE_ID)


def test_bound_store_create_entry() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.create_memory_entry.return_value = entry_payload()
    store = client.memory_stores.get(store_id=STORE_ID)

    entry = store.create_entry(
        actor_id="alice",
        session_id="s1",
        path="/m/p.md",
        content="c",
        description="desc",
    )

    assert entry.entry_id == ENTRY_ID
    api.create_memory_entry.assert_called_once_with(
        STORE_ID,
        "alice",
        "/m/p.md",
        content="c",
        description="desc",
        session_id="s1",
        source_type=None,
    )


def test_list_entries_consumes_pages() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.list_memory_entries.side_effect = [
        {"managed_memory_entries": [entry_payload()], "next_page_token": "p2"},
        {"managed_memory_entries": [entry_payload(name=f"{MEM_STORE_NAME}/entries/second")]},
    ]
    store = client.memory_stores.get(store_id=STORE_ID)

    entries = store.list_entries(actor_id="alice", path_prefix="/m/", page_size=10)

    assert [entry.entry_id for entry in entries] == [ENTRY_ID, "second"]
    assert api.list_memory_entries.call_args_list[1].kwargs["page_token"] == "p2"


def test_get_update_and_delete_entry() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.get_memory_entry.return_value = entry_payload()
    api.update_memory_entry.return_value = entry_payload(content="updated")
    store = client.memory_stores.get(store_id=STORE_ID)

    entry = store.get_entry(entry_id=ENTRY_ID, read_mask="name,content")
    updated = store.update_entry(entry_id=entry.entry_id, content="updated")
    store.delete_entry(entry_id=updated.entry_id)

    assert updated.content == "updated"
    api.get_memory_entry.assert_called_once_with(
        STORE_ID,
        ENTRY_ID,
        read_mask="name,content",
    )
    api.delete_memory_entry.assert_called_once_with(STORE_ID, ENTRY_ID)


def test_search_preserves_scores_and_uses_page_size() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.search_memory_entries.return_value = {
        "results": [{"managed_memory_entry": entry_payload(), "score": 0.9}]
    }
    store = client.memory_stores.get(store_id=STORE_ID)

    results = store.search_entries(actor_id="alice", query="prefs")

    assert results[0].managed_memory_entry.entry_id == ENTRY_ID
    assert results[0].score == 0.9
    api.search_memory_entries.assert_called_once_with(
        STORE_ID,
        "alice",
        "prefs",
        page_size=100,
        path_prefix=None,
        session_id=None,
        read_mask=None,
    )


def test_search_falls_back_to_deprecated_entry_alias() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.search_memory_entries.return_value = {"managed_memory_entries": [entry_payload()]}
    store = client.memory_stores.get(store_id=STORE_ID)

    results = store.search_entries(actor_id="alice", query="prefs", page_size=10)

    assert results[0].managed_memory_entry.entry_id == ENTRY_ID
    assert results[0].score is None


@pytest.mark.parametrize("page_size", [0, 101])
def test_search_rejects_out_of_range_page_size(page_size: int) -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    store = client.memory_stores.get(store_id=STORE_ID)

    with pytest.raises(ValueError, match="page_size"):
        store.search_entries(actor_id="alice", query="q", page_size=page_size)


def test_append_entry_content_creates_when_absent() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.list_memory_entries.return_value = {"managed_memory_entries": []}
    api.create_memory_entry.return_value = entry_payload(content="first")
    store = client.memory_stores.get(store_id=STORE_ID)

    entry = store.append_entry_content(
        actor_id="alice",
        session_id="s1",
        path="/m/p.md",
        content="first",
    )

    assert entry.content == "first"
    api.create_memory_entry.assert_called_once()


def test_append_entry_content_read_modify_writes_existing() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    summary = entry_payload()
    summary.pop("content")
    api.list_memory_entries.return_value = {"managed_memory_entries": [summary]}
    api.get_memory_entry.return_value = entry_payload(content="first", session_id="s1")
    api.update_memory_entry.return_value = entry_payload(content="first\nsecond")
    store = client.memory_stores.get(store_id=STORE_ID)

    entry = store.append_entry_content(
        actor_id="alice",
        session_id="s1",
        path="/m/p.md",
        content="\nsecond",
    )

    assert entry.content == "first\nsecond"
    api.update_memory_entry.assert_called_once_with(
        STORE_ID,
        ENTRY_ID,
        content="first\nsecond",
        description="desc",
    )
