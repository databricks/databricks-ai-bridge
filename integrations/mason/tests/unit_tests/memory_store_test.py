"""Tests for typed managed-memory resources."""

from datetime import datetime, timezone

import pytest
from resource_test_fixtures import (
    MEMORY_ID,
    MEMORY_STORE_NAME,
    STORE_ID,
    mem_store_payload,
    memory_payload,
    resource_client,
)

from databricks_mason import MemoryStore


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


def test_bound_store_create_memory() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.create_memory_entry.return_value = memory_payload()
    store = client.memory_stores.get(STORE_ID)

    memory = store.create_memory(
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


def test_list_memories_consumes_pages() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.list_memory_entries.side_effect = [
        {"managed_memory_entries": [memory_payload()], "next_page_token": "p2"},
        {"managed_memory_entries": [memory_payload(name=f"{MEMORY_STORE_NAME}/entries/second")]},
    ]
    store = client.memory_stores.get(STORE_ID)

    memories = store.list_memories(actor_id="alice", path_prefix="/m/", page_size=10)

    assert [memory.id for memory in memories] == [MEMORY_ID, "second"]
    assert api.list_memory_entries.call_args_list[1].kwargs["page_token"] == "p2"


def test_get_update_and_delete_memory() -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    api.get_memory_entry.return_value = memory_payload()
    api.update_memory_entry.return_value = memory_payload(content="updated")
    store = client.memory_stores.get(STORE_ID)

    memory = store.get_memory(MEMORY_ID, read_mask="name,content")
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

    results = store.search_memories(actor_id="alice", query="prefs", limit=10)

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

    results = store.search_memories(actor_id="alice", query="prefs", limit=10)

    assert results[0].memory.id == MEMORY_ID
    assert results[0].score is None


@pytest.mark.parametrize("limit", [0, 101])
def test_search_rejects_out_of_range_limit(limit: int) -> None:
    client, api = resource_client()
    api.get_memory_store.return_value = mem_store_payload()
    store = client.memory_stores.get(STORE_ID)

    with pytest.raises(ValueError, match="limit"):
        store.search_memories(actor_id="alice", query="q", limit=limit)
