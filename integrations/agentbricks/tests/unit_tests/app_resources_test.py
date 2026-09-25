"""Unit tests for the app-resource plumbing (postgres + experiment app resources)."""

from __future__ import annotations

import json
import types
from typing import Any

from databricks_agentbricks import app_resources as sa


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
