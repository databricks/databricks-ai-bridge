"""Unit tests for the app-resource plumbing (postgres + experiment app resources)."""

from __future__ import annotations

import json
import types
from typing import Any

from databricks_mason import app_resources as sa
from databricks_mason.cli.tracing import TraceTable


def _backend(database: str, resource_name: str) -> sa.LakebaseBackend:
    """Build a LakebaseBackend for resource-attach tests."""
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
    names = {r["name"] for r in payload["app"]["resources"]}
    assert names == {"postgres", "postgres-memory"}  # one update carries both


def test_apply_postgres_resources_preserves_existing_and_updates_ours(monkeypatch):
    resources: list[dict[str, Any]] = [
        {"name": "user-owned", "secret": {}},
        {"name": "postgres-runtime-store", "old": True},
    ]

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        payload = json.loads(args[args.index("--json") + 1])
        resources[:] = payload["app"]["resources"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    backend = _backend("db-new", "postgres-runtime-store")
    assert sa.apply_postgres_resources("myapp", [backend], "prof") is None
    # The user-owned resource is preserved; our managed resource is replaced (not duplicated).
    assert [r["name"] for r in resources] == ["user-owned", "postgres-runtime-store"]
    ours = next(r for r in resources if r["name"] == "postgres-runtime-store")
    assert "old" not in ours and ours["postgres"]["permission"] == "CAN_CONNECT_AND_CREATE"


def test_resource_update_is_masked_to_resources(monkeypatch):
    # Regression (ML-69759): the resource grant must scope its write to `resources` via update_mask
    # and touch no other app field — a bare `apps update` reset user_api_scopes and broke OBO on
    # every deploy.
    captured = {}

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": []}), stderr=""
            )
        captured["args"] = args
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    assert (
        sa.apply_postgres_resources("app", [_backend("db", "postgres-runtime-store")], "prof")
        is None
    )
    assert captured["args"][:2] == [
        "apps",
        "create-update",
    ]  # masked upsert, not a full-spec update
    payload = json.loads(captured["args"][captured["args"].index("--json") + 1])
    assert payload["update_mask"] == "resources"
    assert set(payload["app"]) == {"resources"}  # only resources written; user_api_scopes untouched


def test_apply_postgres_resources_reports_failure(monkeypatch):
    monkeypatch.setattr(
        sa,
        "_databricks",
        lambda args, profile, **kw: (
            types.SimpleNamespace(returncode=1, stdout="", stderr="denied: needs MANAGE")
            if args[:2] == ["apps", "create-update"]
            else types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": []}), stderr=""
            )
        ),
    )
    err = sa.apply_postgres_resources("app", [_backend("db", "postgres-runtime-store")], "prof")
    assert err == "denied: needs MANAGE"


def test_runtime_store_resource_coexists_with_a_second_managed_resource(monkeypatch):
    resources: list[dict[str, Any]] = []

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        payload = json.loads(args[args.index("--json") + 1])
        resources[:] = payload["app"]["resources"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    runtime_store = _backend("runtime-db", "postgres-runtime-store")
    other = _backend("other", "postgres-other")

    assert sa.apply_postgres_resources("app", [runtime_store], "prof") is None
    assert sa.apply_postgres_resources("app", [other], "prof") is None

    assert {resource["name"] for resource in resources} == {
        "postgres-runtime-store",
        "postgres-other",
    }


# --- trace resources (experiment + UC OTEL tables) ---------------------------


def test_apply_trace_resources_managed_experiment_writes_only_the_experiment(monkeypatch):
    resources: list[dict[str, Any]] = [{"name": "user-owned", "secret": {}}]
    captured = {}

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        captured["args"] = args
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    assert sa.apply_trace_resources("app", "exp-1", [], "prof") is None
    payload = json.loads(captured["args"][captured["args"].index("--json") + 1])
    assert payload["update_mask"] == "resources"  # masked upsert, like the other resources
    written = payload["app"]["resources"]
    # the unrelated user resource is preserved; ours is exactly the experiment resource
    assert [r["name"] for r in written] == ["user-owned", "mason-trace-experiment"]
    ours = next(r for r in written if r["name"] == "mason-trace-experiment")
    assert ours["experiment"] == {"experiment_id": "exp-1", "permission": "CAN_EDIT"}


def test_apply_trace_resources_uc_experiment_adds_one_table_resource_per_table(monkeypatch):
    captured = {}

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": []}), stderr=""
            )
        captured["args"] = args
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    tables = [
        TraceTable("spans", "cat.schema.otel_spans"),
        TraceTable("logs", "cat.schema.otel_logs"),
        TraceTable("metrics", "cat.schema.otel_metrics"),
    ]
    assert sa.apply_trace_resources("app", "exp-uc", tables, "prof") is None
    payload = json.loads(captured["args"][captured["args"].index("--json") + 1])
    written = payload["app"]["resources"]
    assert [r["name"] for r in written] == [
        "mason-trace-experiment",
        "mason-trace-table-spans",
        "mason-trace-table-logs",
        "mason-trace-table-metrics",
    ]
    assert written[0]["experiment"] == {"experiment_id": "exp-uc", "permission": "CAN_EDIT"}
    assert [r["uc_securable"] for r in written[1:]] == [
        {
            "securable_full_name": "cat.schema.otel_spans",
            "securable_type": "TABLE",
            "permission": "MODIFY",
        },
        {
            "securable_full_name": "cat.schema.otel_logs",
            "securable_type": "TABLE",
            "permission": "MODIFY",
        },
        {
            "securable_full_name": "cat.schema.otel_metrics",
            "securable_type": "TABLE",
            "permission": "MODIFY",
        },
    ]


def test_apply_trace_resources_converges_when_rebinding_uc_to_managed(monkeypatch):
    # Leak fix: a prior UC deploy left mason-trace-table-* resources behind; redeploying with a
    # managed experiment (tables == ()) must drop them in the same write, not leave the SP with
    # MODIFY on the old UC tables. The stale resources use the previous index-based naming
    # (mason-trace-table-<i>) on purpose: pruning matches by prefix, so those are dropped too.
    resources: list[dict[str, Any]] = [
        {"name": "user-owned", "secret": {}},
        {"name": "mason-trace-experiment", "experiment": {"experiment_id": "old-uc"}},
        {"name": "mason-trace-table-0", "uc_securable": {"securable_full_name": "old.cat.spans"}},
        {"name": "mason-trace-table-1", "uc_securable": {"securable_full_name": "old.cat.logs"}},
    ]

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        payload = json.loads(args[args.index("--json") + 1])
        resources[:] = payload["app"]["resources"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    assert sa.apply_trace_resources("app", "exp-managed", [], "prof") is None
    # stale table resources are gone, the user resource is preserved, the experiment resource is
    # re-written to point at the managed experiment
    assert [r["name"] for r in resources] == ["user-owned", "mason-trace-experiment"]
    ours = next(r for r in resources if r["name"] == "mason-trace-experiment")
    assert ours["experiment"] == {"experiment_id": "exp-managed", "permission": "CAN_EDIT"}


def test_apply_trace_resources_reports_failure(monkeypatch):
    monkeypatch.setattr(
        sa,
        "_databricks",
        lambda args, profile, **kw: (
            types.SimpleNamespace(returncode=1, stdout="", stderr="denied: needs MANAGE")
            if args[:2] == ["apps", "create-update"]
            else types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": []}), stderr=""
            )
        ),
    )
    err = sa.apply_trace_resources(
        "app", "exp-1", [TraceTable("spans", "cat.schema.otel_spans")], "prof"
    )
    assert err == "denied: needs MANAGE"


def test_apply_trace_resources_prunes_everything_when_unbound(monkeypatch):
    # The unbind case: experiment_id None -> the desired set is empty, so the write drops EVERY
    # mason-trace-* resource (experiment + UC tables) while preserving unrelated user resources.
    resources: list[dict[str, Any]] = [
        {"name": "user-owned", "secret": {}},
        {"name": "mason-trace-experiment", "experiment": {"experiment_id": "exp-1"}},
        {"name": "mason-trace-table-0", "uc_securable": {"securable_full_name": "c.s.spans"}},
        {"name": "mason-trace-table-1", "uc_securable": {"securable_full_name": "c.s.logs"}},
    ]

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        payload = json.loads(args[args.index("--json") + 1])
        resources[:] = payload["app"]["resources"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    assert sa.apply_trace_resources("app", None, [], "prof") is None
    assert resources == [{"name": "user-owned", "secret": {}}]
