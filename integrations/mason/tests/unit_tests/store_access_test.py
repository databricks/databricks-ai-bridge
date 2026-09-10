"""Unit tests for the store-access resource plumbing (postgres app resource)."""

from __future__ import annotations

import json
import types

from databricks_mason import lakebase_durability_store, memory_store_access, session_store_access
from databricks_mason import store_access as sa


def test_session_backend_targets_sessions_tables():
    b = session_store_access.backend("my-store")
    assert b.database == "my-store"
    assert b.schema == "public"
    assert b.tables == ("sessions", "session_items")
    assert b.resource_name == "postgres"
    assert b.database_path == (
        "projects/databricks-internal-agent-session-store/branches/production/databases/my-store"
    )


def test_memory_backend_targets_memory_entries():
    db = memory_store_access.database_from_backend_id(
        "projects/databricks-internal-agent-memory-store/branches/production/databases/memory-abc"
    )
    assert db == "memory-abc"
    b = memory_store_access.backend(db)
    assert b.schema == "memory"
    assert b.tables == ("memory_entries",)
    assert b.resource_name == "postgres-memory"  # distinct name so both stores coexist on one app


def test_apply_postgres_resources_sends_all_backends_in_one_update(monkeypatch):
    captured = {}

    def fake_db(args, profile, **kw):
        captured["args"] = args
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    backends = [session_store_access.backend("s"), memory_store_access.backend("memory-x")]
    assert sa.apply_postgres_resources("app", backends, "prof") is None
    payload = json.loads(captured["args"][captured["args"].index("--json") + 1])
    names = {r["name"] for r in payload["resources"]}
    assert names == {"postgres", "postgres-memory"}  # one update carries both


def test_session_store_update_preserves_dedicated_durability_resource(monkeypatch):
    resources = []

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        payload = json.loads(args[args.index("--json") + 1])
        resources[:] = payload["resources"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    durability = lakebase_durability_store.backend("mason-app")
    session = session_store_access.backend("sessions")

    assert sa.apply_postgres_resources("app", [durability], "prof") is None
    assert sa.apply_postgres_resources("app", [session], "prof") is None

    assert {resource["name"] for resource in resources} == {
        "postgres-durability",
        "postgres",
    }
