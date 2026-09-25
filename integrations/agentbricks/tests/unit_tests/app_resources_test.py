"""Unit tests for the app-resource plumbing (postgres + experiment app resources)."""

from __future__ import annotations

import json
import types
from typing import Any

from databricks_agentbricks import app_resources as sa
from databricks_agentbricks.trace_tables import TraceTable, TraceTableKind


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


def test_apply_tool_resources_replaces_complete_owned_subset_and_preserves_unrelated(monkeypatch):
    resources: list[dict[str, Any]] = [
        {"name": "user-owned", "secret": {"scope": "keep"}},
        {"name": "agentbricks-tool-stale", "uc_securable": {"permission": "SELECT"}},
        {"name": "agentbricks-tool-replaced", "uc_securable": {"permission": "SELECT"}},
    ]
    desired = [
        {
            "name": "agentbricks-tool-replaced",
            "uc_securable": {
                "securable_full_name": "main.data.rows",
                "securable_type": "TABLE",
                "permission": "MODIFY",
            },
        },
        {
            "name": "agentbricks-tool-new",
            "genie_space": {"name": "genie", "space_id": "0" * 32, "permission": "CAN_RUN"},
        },
    ]
    payloads = []
    reads = 0

    def fake_db(args, profile, **kw):
        nonlocal reads
        if args[:2] == ["apps", "get"]:
            reads += 1
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        payload = json.loads(args[args.index("--json") + 1])
        payloads.append(payload)
        resources[:] = payload["app"]["resources"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)

    assert sa.apply_tool_resources("app", desired, "prof") is None

    expected_resources = [
        {"name": "user-owned", "secret": {"scope": "keep"}},
        *sorted(desired, key=lambda resource: resource["name"]),
    ]
    assert resources == expected_resources
    assert payloads == [
        {
            "app": {"resources": expected_resources},
            "update_mask": "resources",
        }
    ]
    assert reads == 2  # initial state plus authoritative post-update readback


def test_apply_tool_resources_fails_when_readback_is_missing_desired_resource(monkeypatch):
    desired = [{"name": "agentbricks-tool-new", "uc_securable": {"permission": "SELECT"}}]
    reads = 0

    def fake_db(args, profile, **kw):
        nonlocal reads
        if args[:2] == ["apps", "get"]:
            reads += 1
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"resources": []}),
                stderr="",
            )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)

    error = sa.apply_tool_resources("app", desired, "prof")

    assert error == "Could not verify App tool resources: Agent Bricks-owned resources do not match"
    assert reads == 2


def test_apply_tool_resources_accepts_server_enriched_readback(monkeypatch):
    desired = [
        {
            "name": "agentbricks-tool-table",
            "uc_securable": {
                "securable_full_name": "main.data.rows",
                "securable_type": "TABLE",
                "permission": "SELECT",
            },
        }
    ]
    reads = 0

    def fake_db(args, profile, **kw):
        nonlocal reads
        if args[:2] == ["apps", "get"]:
            reads += 1
            resources = (
                []
                if reads == 1
                else [
                    {
                        **desired[0],
                        "uc_securable": {
                            **desired[0]["uc_securable"],
                            "securable_kind": "TABLE_DELTA",
                        },
                    }
                ]
            )
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"resources": resources}),
                stderr="",
            )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)

    assert sa.apply_tool_resources("app", desired, "prof") is None


def test_apply_tool_resources_fails_closed_when_readback_fails(monkeypatch):
    reads = 0

    def fake_db(args, profile, **kw):
        nonlocal reads
        if args[:2] == ["apps", "get"]:
            reads += 1
            if reads == 1:
                return types.SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"resources": []}),
                    stderr="",
                )
            return types.SimpleNamespace(returncode=1, stdout="", stderr="readback denied")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)

    error = sa.apply_tool_resources(
        "app", [{"name": "agentbricks-tool-new", "uc_securable": {}}], "prof"
    )

    assert error == "Could not verify App tool resources: readback denied"
    assert reads == 2


def test_apply_tool_resources_unchanged_state_reads_once_without_update(monkeypatch):
    desired = [{"name": "agentbricks-tool-current", "uc_securable": {"permission": "EXECUTE"}}]
    calls = []

    def fake_db(args, profile, **kw):
        calls.append(args)
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"resources": desired}),
            stderr="",
        )

    monkeypatch.setattr(sa, "_databricks", fake_db)

    assert sa.apply_tool_resources("app", desired, "prof") is None
    assert calls == [["apps", "get", "app", "-o", "json"]]


def test_apply_tool_resources_prunes_owned_resources_and_skips_unchanged_update(monkeypatch):
    resources: list[dict[str, Any]] = [
        {"name": "user-owned", "secret": {}},
        {"name": "agentbricks-tool-stale", "uc_securable": {}},
    ]
    updates = []

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        updates.append(args)
        payload = json.loads(args[args.index("--json") + 1])
        resources[:] = payload["app"]["resources"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)

    assert sa.apply_tool_resources("app", [], "prof") is None
    assert resources == [{"name": "user-owned", "secret": {}}]
    assert sa.apply_tool_resources("app", [], "prof") is None
    assert len(updates) == 1


def test_apply_tool_resources_reports_failure(monkeypatch):
    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(returncode=0, stdout='{"resources": []}', stderr="")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="denied: needs MANAGE")

    monkeypatch.setattr(sa, "_databricks", fake_db)

    error = sa.apply_tool_resources(
        "app", [{"name": "agentbricks-tool-new", "uc_securable": {}}], "prof"
    )

    assert error == "denied: needs MANAGE"


def test_apply_tool_resources_read_failure_is_fail_closed(monkeypatch):
    calls = []

    def fake_db(args, profile, **kw):
        calls.append(args)
        return types.SimpleNamespace(returncode=1, stdout="", stderr="cannot read app")

    monkeypatch.setattr(sa, "_databricks", fake_db)

    error = sa.apply_tool_resources(
        "app", [{"name": "agentbricks-tool-new", "uc_securable": {}}], "prof"
    )

    assert error == "Could not read existing App resources: cannot read app"
    assert calls == [["apps", "get", "app", "-o", "json"]]


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
    assert [r["name"] for r in written] == ["user-owned", "agentbricks-trace-experiment"]
    ours = next(r for r in written if r["name"] == "agentbricks-trace-experiment")
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
        TraceTable(TraceTableKind.SPANS, "cat.schema.otel_spans"),
        TraceTable(TraceTableKind.LOGS, "cat.schema.otel_logs"),
        TraceTable(TraceTableKind.METRICS, "cat.schema.otel_metrics"),
    ]
    assert sa.apply_trace_resources("app", "exp-uc", tables, "prof") is None
    payload = json.loads(captured["args"][captured["args"].index("--json") + 1])
    written = payload["app"]["resources"]
    assert [r["name"] for r in written] == [
        "agentbricks-trace-experiment",
        "agentbricks-trace-spans",
        "agentbricks-trace-logs",
        "agentbricks-trace-metrics",
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


def test_apply_trace_resources_names_stay_within_databricks_apps_30_char_limit(monkeypatch):
    # Databricks Apps requires each resource name to be 2-30 chars and rejects the ENTIRE resource
    # array if any name is too long - which silently drops every trace grant. Guard the experiment +
    # all four OTEL-table resource names (the longest kind, "annotations", is the tight one).
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
        TraceTable(TraceTableKind.SPANS, "c.s.otel_spans"),
        TraceTable(TraceTableKind.LOGS, "c.s.otel_logs"),
        TraceTable(TraceTableKind.ANNOTATIONS, "c.s.otel_annotations"),
        TraceTable(TraceTableKind.METRICS, "c.s.otel_metrics"),
    ]
    assert sa.apply_trace_resources("app", "exp-uc", tables, "prof") is None
    payload = json.loads(captured["args"][captured["args"].index("--json") + 1])
    names = [r["name"] for r in payload["app"]["resources"]]
    assert any(n.endswith("annotations") for n in names)  # the longest name is exercised
    too_long = [n for n in names if not (2 <= len(n) <= 30)]
    assert not too_long, f"resource names must be 2-30 chars for Databricks Apps; got {too_long}"


def test_apply_trace_resources_converges_when_rebinding_uc_to_managed(monkeypatch):
    # Leak fix: a prior UC deploy left agentbricks-trace-table-* resources behind; redeploying with a
    # managed experiment (tables == ()) must drop them in the same write, not leave the SP with
    # MODIFY on the old UC tables. The stale resources use the previous index-based naming
    # (agentbricks-trace-table-<i>) on purpose: pruning matches by prefix, so those are dropped too.
    resources: list[dict[str, Any]] = [
        {"name": "user-owned", "secret": {}},
        {"name": "agentbricks-trace-experiment", "experiment": {"experiment_id": "old-uc"}},
        {
            "name": "agentbricks-trace-table-0",
            "uc_securable": {"securable_full_name": "old.cat.spans"},
        },
        {
            "name": "agentbricks-trace-table-1",
            "uc_securable": {"securable_full_name": "old.cat.logs"},
        },
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
    assert [r["name"] for r in resources] == ["user-owned", "agentbricks-trace-experiment"]
    ours = next(r for r in resources if r["name"] == "agentbricks-trace-experiment")
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
        "app", "exp-1", [TraceTable(TraceTableKind.SPANS, "cat.schema.otel_spans")], "prof"
    )
    assert err == "denied: needs MANAGE"


def _fake_db_get_fails(calls):
    """A `_databricks` stub whose `apps get` fails; records every command's first two args."""

    def fake_db(args, profile, **kw):
        calls.append(args[:2])
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=1, stdout="", stderr="transient: apps get failed"
            )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    return fake_db


def test_apply_trace_resources_skips_write_when_current_resources_unreadable(monkeypatch):
    # A failed `apps get` must NOT be treated as "no resources": the write is a full-array replace, so
    # proceeding with an empty `preserved` would drop every OTHER resource the app has. Bail with a
    # reason and perform NO write, leaving the app's resources (and existing trace grants) intact.
    calls: list[list[str]] = []
    monkeypatch.setattr(sa, "_databricks", _fake_db_get_fails(calls))
    err = sa.apply_trace_resources(
        "app", "exp-1", [TraceTable(TraceTableKind.SPANS, "cat.schema.otel_spans")], "prof"
    )
    assert err and "apps get failed" in err
    assert ["apps", "create-update"] not in calls  # never wrote


def test_apply_postgres_resources_skips_write_when_current_resources_unreadable(monkeypatch):
    # Same guard for the postgres reconcile - a failed read bails before the full-array replace.
    calls: list[list[str]] = []
    monkeypatch.setattr(sa, "_databricks", _fake_db_get_fails(calls))
    err = sa.apply_postgres_resources("app", [_backend("db", "postgres-runtime-store")], "prof")
    assert err and "apps get failed" in err
    assert ["apps", "create-update"] not in calls  # never wrote


def test_apply_trace_resources_prunes_everything_when_unbound(monkeypatch):
    # The unbind case: experiment_id None -> the desired set is empty, so the write drops EVERY
    # agentbricks-trace-* resource (experiment + UC tables) while preserving unrelated user resources.
    resources: list[dict[str, Any]] = [
        {"name": "user-owned", "secret": {}},
        {"name": "agentbricks-trace-experiment", "experiment": {"experiment_id": "exp-1"}},
        {"name": "agentbricks-trace-table-0", "uc_securable": {"securable_full_name": "c.s.spans"}},
        {"name": "agentbricks-trace-table-1", "uc_securable": {"securable_full_name": "c.s.logs"}},
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
