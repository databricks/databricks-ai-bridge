"""CLI-level unit tests for `mason sessions` commands (stores / sessions / items).

Exercises each command's handler + render path with a fake low-level client, covering
both text and JSON output modes. Backfills coverage for src/databricks_mason/sessions.py.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from databricks_mason.sessions import items, sessions, stores

STORE = "cli-store"
SID = "11111111-1111-1111-1111-111111111111"


def _store(**over):
    d = {
        "session_store_name": STORE,
        "session_store_id": "store-uuid",
        "create_time": "2026-08-15T01:02:03Z",
        "update_time": "2026-08-16T01:02:03Z",
        "description": "a store",
        "metadata": {"k": "v"},
    }
    d.update(over)
    return d


def _session(**over):
    d = {
        "session_id": SID,
        "session_store_name": STORE,
        "actor_id": "alice",
        "parent_session_id": None,
        "root_session_id": SID,
        "create_time": "2026-08-15T01:02:03Z",
        "last_activity_time": "2026-08-15T02:02:03Z",
        "metadata": {"stage": "done"},
    }
    d.update(over)
    return d


def _item(i):
    return {
        "item_id": f"item-{i}",
        "create_time": "2026-08-15T01:02:03Z",
        "data": {"role": "user", "content": f"msg-{i}"},
    }


class _FakeClient:
    host = "https://ws.example.com"

    def __init__(self):
        self.calls = []

    def create_session_store(self, name, description=None, metadata=None):
        self.calls.append(("create_store", name))
        return _store(session_store_name=name)

    def get_session_store(self, name):
        self.calls.append(("get_store", name))
        return _store(session_store_name=name)

    def list_session_stores(self, page_size=None, page_token=None):
        self.calls.append(("list_stores", page_size, page_token))
        return {
            "session_stores": [_store(), _store(session_store_name="s2")],
            "next_page_token": None,
        }

    def update_session_store(self, name, description=None, metadata=None):
        self.calls.append(("update_store", name, description))
        return _store(description=description or "a store")

    def delete_session_store(self, name):
        self.calls.append(("delete_store", name))

    def create_session(
        self, store, actor_id, session_id=None, parent_session_id=None, metadata=None
    ):
        self.calls.append(("create_session", store, actor_id))
        return _session(actor_id=actor_id)

    def list_sessions(self, store, filter=None, order_by=None, page_size=None, page_token=None):
        self.calls.append(("list_sessions", store, filter, order_by))
        return {
            "sessions": [_session(), _session(session_id="22222222-2222-2222-2222-222222222222")],
            "next_page_token": None,
        }

    def get_session(self, session_id, store=None):
        self.calls.append(("get_session", session_id, store))
        return _session(session_id=session_id)

    def update_session(self, store, session_id, metadata):
        self.calls.append(("update_session", store, session_id, metadata))
        return _session(metadata=metadata)

    def delete_session(self, store, session_id, force=False):
        self.calls.append(("delete_session", store, session_id, force))

    def fork_session(self, store, source, actor_id, up_to=None, session_id=None, metadata=None):
        self.calls.append(("fork", store, source, actor_id))
        return {"session": _session(session_id="forked-1", root_session_id="forked-1")}

    def list_session_items(self, store, session_id, order_by=None, page_size=None, page_token=None):
        self.calls.append(("list_items", store, session_id, order_by))
        return {"session_items": [_item(0), _item(1)], "next_page_token": None}

    def append_session_items(self, store, session_id, payload):
        self.calls.append(("append", store, session_id, payload))
        return {"session_items": [_item(i) for i in range(len(payload))]}

    def pop_session_item(self, store, session_id):
        self.calls.append(("pop", store, session_id))
        return {"item": _item(9)}

    def clear_session_items(self, store, session_id):
        self.calls.append(("clear", store, session_id))


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


# ---- session stores ----
def test_stores_create_text_and_json():
    r, c = _run(
        stores, ["create", "--name", STORE, "--description", "d", "--metadata", '{"k":"v"}']
    )
    assert r.exit_code == 0, r.output
    assert STORE in r.output
    r2, _ = _run(stores, ["create", "--name", STORE], output="json")
    assert r2.exit_code == 0 and json.loads(r2.output)["session_store_name"] == STORE


def test_stores_list_get_update():
    r, c = _run(stores, ["list", "--page-size", "5"])
    assert r.exit_code == 0 and STORE in r.output
    r2, _ = _run(stores, ["get", STORE])
    assert r2.exit_code == 0 and STORE in r2.output
    r3, c3 = _run(stores, ["update", STORE, "--description", "new"])
    assert r3.exit_code == 0 and ("update_store", STORE, "new") in c3.calls


def test_stores_delete_requires_confirmation():
    r, c = _run(stores, ["delete", STORE], output="text")  # no --yes, no input -> abort
    assert r.exit_code != 0
    assert not any(x[0] == "delete_store" for x in c.calls)
    r2, c2 = _run(stores, ["delete", STORE, "--yes"])
    assert r2.exit_code == 0 and ("delete_store", STORE) in c2.calls


# ---- sessions ----
def test_session_create_list_get_update():
    r, c = _run(
        sessions, ["create", "--store", STORE, "--actor-id", "alice", "--metadata", '{"k":"v"}']
    )
    assert r.exit_code == 0, r.output
    r2, _ = _run(sessions, ["list", "--store", STORE, "--filter", 'actor_id = "alice"'])
    assert r2.exit_code == 0 and "alice" in r2.output
    r3, _ = _run(sessions, ["get", SID, "--store", STORE])
    assert r3.exit_code == 0 and SID in r3.output
    r4, c4 = _run(sessions, ["update", SID, "--store", STORE, "--metadata", '{"stage":"done"}'])
    assert r4.exit_code == 0 and any(x[0] == "update_session" for x in c4.calls)


def test_session_get_without_store_errors():
    r, _ = _run(sessions, ["get", SID])  # store required in this preview
    assert r.exit_code != 0
    assert "store" in r.output.lower()


def test_session_delete_and_force():
    r, c = _run(sessions, ["delete", SID, "--store", STORE, "--yes"])
    assert r.exit_code == 0 and ("delete_session", STORE, SID, False) in c.calls
    r2, c2 = _run(sessions, ["delete", SID, "--store", STORE, "--force", "--yes"])
    assert r2.exit_code == 0 and ("delete_session", STORE, SID, True) in c2.calls


def test_session_fork():
    r, c = _run(
        sessions, ["fork", "--store", STORE, "--source-session-id", SID, "--actor-id", "alice"]
    )
    assert r.exit_code == 0, r.output
    assert any(x[0] == "fork" for x in c.calls)


# ---- items ----
def test_items_append_list_pop_clear():
    r, c = _run(
        items,
        [
            "append",
            "--store",
            STORE,
            "--session-id",
            SID,
            "--data",
            '{"role":"user","content":"hi"}',
            "--data",
            '{"role":"assistant","content":"yo"}',
        ],
    )
    assert r.exit_code == 0 and any(x[0] == "append" for x in c.calls)
    r2, _ = _run(items, ["list", "--store", STORE, "--session-id", SID])
    assert r2.exit_code == 0 and "msg-0" in r2.output
    r3, _ = _run(items, ["pop", "--store", STORE, "--session-id", SID])
    assert r3.exit_code == 0 and "item-9" in r3.output  # popped item surfaced
    r4, c4 = _run(items, ["clear", "--store", STORE, "--session-id", SID])
    assert r4.exit_code == 0 and ("clear", STORE, SID) in c4.calls


def test_items_append_invalid_json_errors():
    r, c = _run(items, ["append", "--store", STORE, "--session-id", SID, "--data", "not-json{"])
    assert r.exit_code != 0
    assert not any(x[0] == "append" for x in c.calls)


def test_items_append_requires_data_or_file():
    r, _ = _run(items, ["append", "--store", STORE, "--session-id", SID])
    assert r.exit_code != 0
