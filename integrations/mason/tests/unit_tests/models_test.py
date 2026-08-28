"""Tests for the dict-backed typed models over real captured response shapes."""

from __future__ import annotations

import json

from databricks_mason import models

# Shapes captured from a live agents/v1 workspace.
STORE = {
    "name": "memory-stores/abc",
    "display_name": "my-store",
    "workspace_id": "123",
    "storage_backend": {"backend_type": "STORAGE_BACKEND_TYPE_LAKEBASE", "backend_id": "b1"},
    "owner_user_id": "u1",
    "description": "d",
    "create_time": "2026-08-28T07:10:59.794Z",
    "update_time": "2026-08-28T07:10:59.794Z",
}
ENTRY = {
    "name": "memory-stores/abc/entries/e1",
    "actor_id": "alice",
    "path": "/notes/1.md",
    "content": "hello",
    "create_time": "t",
    "update_time": "t",
}


def test_typed_and_dict_access_coexist():
    s = models.MemoryStore(STORE)
    assert s.name == "memory-stores/abc"
    assert s.display_name == "my-store"
    assert s["name"] == "memory-stores/abc"  # still a dict
    backend = s.storage_backend
    assert backend is not None
    assert backend.backend_type == "STORAGE_BACKEND_TYPE_LAKEBASE"
    assert backend.backend_id == "b1"


def test_forward_compatible_unknown_fields_preserved():
    s = models.MemoryStore({**STORE, "brand_new_field": 42})
    assert s["brand_new_field"] == 42
    assert s.to_dict()["brand_new_field"] == 42
    assert s.name == "memory-stores/abc"


def test_json_serializable():
    s = models.MemoryStore(STORE)
    assert json.loads(json.dumps(s)) == STORE


def test_dict_methods_not_shadowed():
    # `.items()` must remain dict.items(), not a list accessor.
    s = models.MemoryStore(STORE)
    assert callable(s.items)
    assert ("name", "memory-stores/abc") in list(s.items())


def test_list_models():
    lst = models.MemoryStoreList({"managed_memory_stores": [STORE], "next_page_token": "tok"})
    assert len(lst.stores) == 1
    assert isinstance(lst.stores[0], models.MemoryStore)
    assert lst.stores[0].display_name == "my-store"
    assert lst.next_page_token == "tok"

    empty = models.MemoryStoreList({})  # API omits the key when empty
    assert empty.stores == []
    assert empty.next_page_token is None


def test_search_result():
    r = models.MemorySearchResult(
        {
            "managed_memory_entries": [ENTRY],
            "results": [{"managed_memory_entry": ENTRY, "score": 0.69}],
        }
    )
    assert r.entries[0].path == "/notes/1.md"
    assert r.results[0].score == 0.69
    hit_entry = r.results[0].entry
    assert hit_entry is not None
    assert hit_entry.actor_id == "alice"


def test_session_item_list_and_pop():
    item = {"item_id": "i1", "data": {"role": "user"}, "create_time": "t"}
    lst = models.SessionItemList({"session_items": [item]})
    assert lst.session_items[0].item_id == "i1"
    assert lst.session_items[0].data == {"role": "user"}

    popped = models.PoppedSessionItem({"item": item})
    assert popped.item is not None
    assert popped.item.item_id == "i1"
