"""CLI-level unit tests for `mason memory` entries + stores commands.

Backfills coverage for the memory command handlers/render paths in
src/databricks_mason/cli/memory.py using a fake low-level client.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from databricks_mason.cli.memory import entries, stores

STORE = "mem-uuid"
ENTRY = f"memory-stores/{STORE}/entries/e1"


def _entry(**over):
    d = {
        "name": ENTRY,
        "actor_id": "alice",
        "session_id": None,
        "path": "/p/a.md",
        "content": "hello",
        "description": "d",
        "source_type": "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT",
        "create_time": "2026-08-15T01:02:03Z",
        "update_time": "2026-08-16T01:02:03Z",
    }
    d.update(over)
    return d


def _store(**over):
    d = {
        "name": f"memory-stores/{STORE}",
        "display_name": "cli-mem",
        "create_time": "2026-08-15T01:02:03Z",
        "update_time": "2026-08-16T01:02:03Z",
        "description": "d",
    }
    d.update(over)
    return d


class _FakeClient:
    host = "https://ws.example.com"

    def __init__(self):
        self.calls = []

    def create_memory_store(self, display_name, description=None):
        self.calls.append(("create_store", display_name))
        return _store(display_name=display_name)

    def get_memory_store(self, name):
        self.calls.append(("get_store", name))
        return _store()

    def delete_memory_store(self, name):
        self.calls.append(("delete_store", name))

    def create_memory_entry(
        self,
        store,
        actor_id,
        path,
        content=None,
        description=None,
        session_id=None,
        source_type=None,
    ):
        self.calls.append(("create_entry", store, actor_id, path))
        return _entry(path=path, actor_id=actor_id)

    def get_memory_entry(self, store, entry, read_mask=None):
        self.calls.append(("get_entry", store, entry, read_mask))
        return _entry()

    def list_memory_entries(
        self, store, actor_id, path_prefix=None, session_id=None, page_size=None, page_token=None
    ):
        self.calls.append(("list_entries", store, actor_id))
        return {
            "managed_memory_entries": [
                _entry(),
                _entry(name=f"memory-stores/{STORE}/entries/e2", path="/p/b.md"),
            ],
            "next_page_token": None,
        }

    def search_memory_entries(self, store, actor_id, query, page_size=None):
        self.calls.append(("search", store, actor_id, query))
        return {"results": [{"managed_memory_entry": _entry(content="widget note"), "score": 0.9}]}

    def update_memory_entry(self, store, entry, content=None, description=None):
        self.calls.append(("update_entry", store, entry, content))
        return _entry(content=content or "hello")

    def delete_memory_entry(self, store, entry):
        self.calls.append(("delete_entry", store, entry))


class _Ctx:
    def __init__(self, client, output="text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


def _run(cmd, args, output="text"):
    client = _FakeClient()
    result = CliRunner().invoke(cmd, args, obj=_Ctx(client, output=output))
    return result, client


# ---- stores ----
def test_store_create_and_delete_confirmation():
    r, c = _run(stores, ["create", "--display-name", "cli-mem", "--description", "d"])
    assert r.exit_code == 0 and any(x[0] == "create_store" for x in c.calls)
    # delete without --yes aborts
    r2, c2 = _run(stores, ["delete", STORE])
    assert r2.exit_code != 0 and not any(x[0] == "delete_store" for x in c2.calls)
    r3, c3 = _run(stores, ["delete", STORE, "--yes"])
    assert r3.exit_code == 0 and ("delete_store", STORE) in c3.calls


# ---- entries ----
def test_entry_create_text_and_json():
    r, c = _run(
        entries,
        [
            "create",
            "--store",
            STORE,
            "--actor-id",
            "alice",
            "--path",
            "/p/a.md",
            "--content",
            "hello",
        ],
    )
    assert r.exit_code == 0, r.output
    assert any(x[0] == "create_entry" for x in c.calls)
    r2, _ = _run(
        entries,
        ["create", "--store", STORE, "--actor-id", "a", "--path", "/x.md", "--content", "c"],
        output="json",
    )
    assert r2.exit_code == 0 and json.loads(r2.output)["path"] == "/x.md"


def test_entry_get_list_search_update():
    r, _ = _run(entries, ["get", "--store", STORE, ENTRY])
    assert r.exit_code == 0 and "hello" in r.output
    r2, _ = _run(entries, ["list", "--store", STORE, "--actor-id", "alice"])
    assert r2.exit_code == 0 and "/p/a.md" in r2.output
    r3, _ = _run(entries, ["search", "--store", STORE, "--actor-id", "alice", "--query", "widget"])
    assert r3.exit_code == 0 and "widget" in r3.output
    r4, c4 = _run(entries, ["update", "--store", STORE, ENTRY, "--content", "new"])
    assert r4.exit_code == 0 and ("update_entry", STORE, ENTRY, "new") in c4.calls


def test_entry_delete_confirmation():
    r, c = _run(entries, ["delete", "--store", STORE, ENTRY])  # no --yes -> abort
    assert r.exit_code != 0 and not any(x[0] == "delete_entry" for x in c.calls)
    r2, c2 = _run(entries, ["delete", "--store", STORE, ENTRY, "--yes"])
    assert r2.exit_code == 0 and ("delete_entry", STORE, ENTRY) in c2.calls


def test_entry_get_bare_id_without_store_errors():
    # optional --store: bare id (not a full resource name) must require --store
    r, c = _run(entries, ["get", "bare-id"])
    assert r.exit_code != 0
    assert not any(x[0] == "get_entry" for x in c.calls)
