"""Unit tests for the store-access grant plumbing (postgres resource + owner-issued GRANT)."""

from __future__ import annotations

import json
import types

import psycopg

from databricks_mason import memory_store_access, session_store_access
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


class _FakeConn:
    """Stand-in for a psycopg connection: records the connect kwargs and executed SQL."""

    def __init__(self, captured):
        self._captured = captured

    def execute(self, sql):
        # store_access encodes the composed GRANT to bytes; decode so assertions read as text.
        self._captured["sql"] = sql.decode() if isinstance(sql, bytes) else sql

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_grant_tables_runs_scoped_grant_over_psycopg(monkeypatch):
    monkeypatch.setattr(sa, "_resolve_pg_host", lambda b, p: "ep-x.databricks.com")
    monkeypatch.setattr(sa, "_mint_token", lambda b, p: "tok")
    captured = {}

    def fake_connect(**kwargs):
        captured["connect"] = kwargs
        return _FakeConn(captured)

    monkeypatch.setattr(sa.psycopg, "connect", fake_connect)

    err = sa.grant_tables(memory_store_access.backend("memory-x"), "sp-1", "me@x.com", "prof")

    assert err is None
    assert captured["connect"]["dbname"] == "memory-x"  # connects to the per-store database
    assert captured["connect"]["password"] == "tok"
    assert captured["connect"]["user"] == "me@x.com"
    assert captured["connect"]["autocommit"] is True  # DDL applies without an explicit commit
    sql = captured["sql"]
    # tables are schema-qualified so the SP's search_path doesn't matter.
    assert 'ON memory.memory_entries TO "sp-1"' in sql
    assert "USAGE ON SCHEMA memory" in sql


def test_grant_tables_reports_connection_error(monkeypatch):
    monkeypatch.setattr(sa, "_resolve_pg_host", lambda b, p: "ep-x.databricks.com")
    monkeypatch.setattr(sa, "_mint_token", lambda b, p: "tok")

    def fake_connect(**kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(sa.psycopg, "connect", fake_connect)
    err = sa.grant_tables(session_store_access.backend("s"), "sp", "me@x.com", "prof")
    assert err is not None and "connection refused" in err


def test_resolve_pg_host_reads_endpoint(monkeypatch):
    endpoint_json = json.dumps({"status": {"hosts": {"host": "ep-plain.databricks.com"}}})
    monkeypatch.setattr(
        sa,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(
            returncode=0, stdout=endpoint_json, stderr=""
        ),
    )
    assert (
        sa._resolve_pg_host(session_store_access.backend("s"), "prof") == "ep-plain.databricks.com"
    )
