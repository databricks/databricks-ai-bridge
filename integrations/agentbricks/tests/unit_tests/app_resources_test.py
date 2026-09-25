"""Unit tests for the app-resource plumbing (postgres + experiment app resources)."""

from __future__ import annotations

import json
import types
from typing import Any

from databricks_agentbricks import app_resources as sa
from databricks_agentbricks.agent_project import ConnectionSpec


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


def test_connection_resources_preserve_unrelated_cleanup_legacy_and_grant_only_app_principal(
    monkeypatch,
):
    resources: list[dict[str, Any]] = [
        {"name": "owner-resource", "secret": {"scope": "external"}},
        {
            "name": "agentbricks-cx-stale",
            "uc_securable": {"securable_full_name": "main.connections.old"},
        },
    ]
    grants: list[tuple[str, dict[str, Any]]] = []

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "resources": resources,
                        "service_principal_client_id": "app-sp-client-id",
                    }
                ),
                stderr="",
            )
        payload = json.loads(args[args.index("--json") + 1])
        if args[:2] == ["grants", "update"]:
            grants.append((args[3], payload))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        resources[:] = payload["app"]["resources"]
        assert payload["update_mask"] == "resources"
        assert set(payload["app"]) == {"resources"}
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    bindings = [
        ConnectionSpec("Salesforce_API", "main.connections.salesforce", "http", "app"),
        ConnectionSpec("github", "main.connections.github", "mcp", "user"),
    ]

    assert sa.apply_connection_resources("agent", bindings, "profile") is None

    assert resources == [{"name": "owner-resource", "secret": {"scope": "external"}}]
    assert grants == [
        (
            full_name,
            {"changes": [{"principal": "app-sp-client-id", "add": [privilege]}]},
        )
        for full_name, privilege in (
            ("main", "USE_CATALOG"),
            ("main.connections", "USE_SCHEMA"),
            ("main.connections.salesforce", "USE_CONNECTION"),
        )
    ]


def test_connection_resources_grant_three_part_fqn_to_app_service_principal(monkeypatch):
    calls: list[list[str]] = []

    def fake_db(args, profile, **kw):
        calls.append(args)
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "resources": [],
                        "service_principal_client_id": "app-sp-client-id",
                    }
                ),
                stderr="",
            )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)

    assert (
        sa.apply_connection_resources(
            "agent",
            [
                ConnectionSpec("github", "main.agent_connections.github_pat", "http", "app"),
                ConnectionSpec("linear", "main.agent_connections.linear_oauth", "mcp", "user"),
            ],
            "profile",
        )
        is None
    )

    grant_calls = [args for args in calls if args[:2] == ["grants", "update"]]
    assert [(args[2], args[3]) for args in grant_calls] == [
        ("catalog", "main"),
        ("schema", "main.agent_connections"),
        ("connection", "main.agent_connections.github_pat"),
    ]
    assert [json.loads(args[args.index("--json") + 1]) for args in grant_calls] == [
        {"changes": [{"principal": "app-sp-client-id", "add": [privilege]}]}
        for privilege in ("USE_CATALOG", "USE_SCHEMA", "USE_CONNECTION")
    ]


def test_connection_resources_remove_stale_binding_when_alias_becomes_user(monkeypatch):
    resources: list[dict[str, Any]] = [
        {"name": "owner-resource", "secret": {}},
        {"name": "agentbricks-cx-provider", "uc_securable": {}},
    ]

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": resources}), stderr=""
            )
        resources[:] = json.loads(args[args.index("--json") + 1])["app"]["resources"]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)

    assert (
        sa.apply_connection_resources(
            "agent",
            [ConnectionSpec("provider", "main.connections.provider", "http", "user")],
            "profile",
        )
        is None
    )
    assert resources == [{"name": "owner-resource", "secret": {}}]


def test_connection_grant_missing_app_service_principal_fails_without_update(monkeypatch):
    updates = []

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(returncode=0, stdout='{"resources": []}', stderr="")
        updates.append(args)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    error = sa.apply_connection_resources(
        "agent",
        [ConnectionSpec("provider", "main.connections.provider", "http", "app")],
        "profile",
    )

    assert error == "Could not resolve the Databricks App service principal."
    assert updates == []


def test_connection_resource_failure_is_redacted(monkeypatch):
    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=('{"resources": [], "service_principal_client_id": "app-sp-client-id"}'),
                stderr="",
            )
        return types.SimpleNamespace(
            returncode=1,
            stdout="SENTINEL-RESPONSE-BODY",
            stderr="denied with SENTINEL-CLIENT-SECRET",
        )

    monkeypatch.setattr(sa, "_databricks", fake_db)
    error = sa.apply_connection_resources(
        "agent",
        [ConnectionSpec("provider", "main.connections.provider", "http", "app")],
        "profile",
    )

    assert error == "Databricks rejected the UC Connection hierarchy grant."
    assert "SENTINEL" not in error


def test_connection_resource_read_failure_never_overwrites_existing_resources(monkeypatch):
    updates = []

    def fake_db(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=1,
                stdout="SENTINEL-EXISTING-RESOURCE",
                stderr="denied with SENTINEL-TOKEN",
            )
        updates.append(args)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sa, "_databricks", fake_db)
    error = sa.apply_connection_resources(
        "agent",
        [ConnectionSpec("provider", "main.connections.provider", "http", "app")],
        "profile",
    )

    assert error == "Could not safely read existing Databricks App resources."
    assert updates == []
    assert "SENTINEL" not in error
