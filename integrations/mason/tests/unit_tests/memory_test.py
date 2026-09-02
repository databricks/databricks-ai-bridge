"""Unit tests for `mason memory` store rendering (timestamp field mapping)."""

from __future__ import annotations

from click.testing import CliRunner

from databricks_mason import memory as memory_mod
from databricks_mason.memory import stores


class _Client:
    host = "https://example.databricks.com"

    def __init__(self, store=None, page=None):
        self._store = store
        self._page = page

    def get_memory_store(self, name):
        return self._store

    def list_memory_stores(self, page_size=None, page_token=None):
        return self._page


class _Ctx:
    def __init__(self, client, output="text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


def test_store_created_updated_read_current_and_legacy_fields():
    # Current API field.
    assert (
        memory_mod._store_created({"create_time": "2026-08-15T01:29:00Z"}) == "2026-08-15T01:29:00Z"
    )
    assert (
        memory_mod._store_updated({"update_time": "2026-08-16T01:29:00Z"}) == "2026-08-16T01:29:00Z"
    )
    # Legacy epoch-millis fallback.
    assert memory_mod._store_created({"created_at": 1_755_100_000_000}) == 1_755_100_000_000
    assert memory_mod._store_updated({"updated_at": 1_755_100_000_000}) == 1_755_100_000_000
    # create_time takes precedence over the legacy field.
    assert memory_mod._store_created({"create_time": "new", "created_at": 1}) == "new"
    # Missing -> falsy (renders as em-dash downstream).
    assert not memory_mod._store_created({})


def test_store_get_renders_timestamps_from_create_time():
    store = {
        "name": "memory-stores/abc123",
        "display_name": "demo",
        "create_time": "2026-08-15T01:29:00Z",
        "update_time": "2026-08-16T01:29:00Z",
    }
    result = CliRunner().invoke(stores, ["get", "abc123"], obj=_Ctx(_Client(store=store)))
    assert result.exit_code == 0, result.output
    # Timestamps render (year present) instead of the em-dash placeholder.
    assert "2026" in result.output


def test_store_list_renders_timestamps_from_create_time():
    page = {
        "managed_memory_stores": [
            {
                "name": "memory-stores/abc123",
                "display_name": "demo",
                "create_time": "2026-08-15T01:29:00Z",
                "update_time": "2026-08-15T01:29:00Z",
            }
        ]
    }
    result = CliRunner().invoke(stores, ["list"], obj=_Ctx(_Client(page=page)))
    assert result.exit_code == 0, result.output
    # A humanized relative time appears rather than "—" for the row.
    assert "ago" in result.output or "just now" in result.output
