"""Unit tests for the app-resource plumbing (postgres + experiment app resources)."""

from __future__ import annotations

import json
import types

from databricks_mason import app_resources as sa
from databricks_mason import lakebase_durability_store


def _backend(database: str, resource_name: str) -> sa.LakebaseBackend:
    """Build a LakebaseBackend for resource-attach tests (durability supplies real ones)."""
    return sa.LakebaseBackend(
        project="proj",
        branch="production",
        endpoint_id="primary",
        database=database,
        schema="public",
        tables=(),
        resource_name=resource_name,
    )


def test_apply_postgres_resources_sends_all_backends_in_one_update(monkeypatch):
    captured = {}

    def fake_db(args, profile, **kw):
        captured["args"] = args
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    backends = [_backend("s", "postgres"), _backend("memory-x", "postgres-memory")]
    assert sa.apply_postgres_resources("app", backends, "prof") is None
    payload = json.loads(captured["args"][captured["args"].index("--json") + 1])
    names = {r["name"] for r in payload["resources"]}
    assert names == {"postgres", "postgres-memory"}  # one update carries both


def test_apply_postgres_resources_preserves_existing_and_updates_ours(monkeypatch):
    resources = [{"name": "user-owned", "secret": {}}, {"name": "postgres-durability", "old": True}]

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        payload = json.loads(args[args.index("--json") + 1])
        resources[:] = payload["resources"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    backend = _backend("db-new", "postgres-durability")
    assert sa.apply_postgres_resources("myapp", [backend], "prof") is None
    # The user-owned resource is preserved; our managed resource is replaced (not duplicated).
    assert [r["name"] for r in resources] == ["user-owned", "postgres-durability"]
    ours = next(r for r in resources if r["name"] == "postgres-durability")
    assert "old" not in ours and ours["postgres"]["permission"] == "CAN_CONNECT_AND_CREATE"


def test_apply_postgres_resources_reports_failure(monkeypatch):
    monkeypatch.setattr(
        sa,
        "_databricks",
        lambda args, profile, **kw: (
            types.SimpleNamespace(returncode=1, stdout="", stderr="denied: needs MANAGE")
            if args[:2] == ["apps", "update"]
            else types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": []}), stderr=""
            )
        ),
    )
    err = sa.apply_postgres_resources("app", [_backend("db", "postgres-durability")], "prof")
    assert err == "denied: needs MANAGE"


def test_durability_resource_coexists_with_a_second_managed_resource(monkeypatch):
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
    other = _backend("other", "postgres-other")

    assert sa.apply_postgres_resources("app", [durability], "prof") is None
    assert sa.apply_postgres_resources("app", [other], "prof") is None

    assert {resource["name"] for resource in resources} == {
        "postgres-durability",
        "postgres-other",
    }
