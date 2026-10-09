"""Unit tests for the deploy bug fixes (ML-69245/69246/69248/69259)."""

from __future__ import annotations

import json
import types

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.legacy_runtime_store import LakebaseBackend
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.deployment.names import DeploymentName, _prefixed_name


class _Ctx:
    profile = "prof"
    output = "text"
    api_client_provider = types.SimpleNamespace(get=lambda: types.SimpleNamespace())


# --- ML-69259 / 69247: deployment-name validation ---------------------------


@pytest.mark.parametrize("bad", ["", "   ", "../../x", "a/b", "a b", "..", "with\ttab"])
def test_validate_deployment_name_rejects_unsafe(bad):
    with pytest.raises(AgentCliError):
        DeploymentName(bad)


def test_validate_deployment_name_accepts_good():
    assert DeploymentName("agent-bricks-agent-1") == "agent-bricks-agent-1"


def test_validate_deployment_name_rejects_too_long():
    with pytest.raises(AgentCliError, match="too long"):
        DeploymentName("agent-bricks-" + "a" * 18)  # 31 chars


def test_validate_deployment_name_accepts_platform_limit():
    name = "a" * 30
    assert DeploymentName(name) == name


# --- deployment prefixes + list filtering -----------------------------------


def test_prefixed_name_adds_prefix_when_absent():
    assert _prefixed_name("foo") == "agent-bricks-foo"


def test_prefixed_name_is_idempotent():
    assert _prefixed_name("agent-bricks-foo") == "agent-bricks-foo"


def test_deployments_list_shows_agent_bricks_apps(monkeypatch):
    apps = {
        "apps": [
            {"name": "agent-bricks-new"},
            {"name": "agent-bricks-foo"},
            {"name": "someone-else-app"},
            {"name": "agent-bricks-bar"},
        ]
    }
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=json.dumps(apps), stderr=""),
    )

    class _JsonCtx:
        profile = "prof"
        output = "json"
        api_client_provider = types.SimpleNamespace(get=lambda: types.SimpleNamespace())

    result = CliRunner().invoke(deploy_mod.deployments_list, [], obj=_JsonCtx())
    assert result.exit_code == 0, result.output
    names = {a["name"] for a in json.loads(result.output)}
    assert names == {"agent-bricks-new", "agent-bricks-foo", "agent-bricks-bar"}


def test_deployments_get_rejects_empty_name_without_calling_cli(monkeypatch):
    called = []
    monkeypatch.setattr(deploy_mod, "_databricks", lambda *a, **k: called.append(a))
    result = CliRunner().invoke(deploy_mod.deployments_get, [""], obj=_Ctx())
    assert result.exit_code != 0
    assert "Invalid deployment name" in result.output
    assert called == []  # never shelled out to `databricks apps get`


# --- ML-69246: confirmation on destructive deployment ops --------------------


def test_delete_aborts_without_confirmation(monkeypatch):
    called = []
    monkeypatch.setattr(deploy_mod, "_databricks", lambda *a, **k: called.append(a))
    result = CliRunner().invoke(deploy_mod.deployments_delete, ["myapp"], obj=_Ctx(), input="n\n")
    assert result.exit_code != 0  # aborted
    assert called == []


def test_delete_proceeds_with_yes(monkeypatch):
    called = []
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **k: (
            called.append(args) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    result = CliRunner().invoke(deploy_mod.deployments_delete, ["myapp", "--yes"], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert called and called[0][:3] == ["apps", "delete", "myapp"]


# --- ML-69245: postgres resources are MERGED, not replaced -------------------


def test_apply_postgres_resources_preserves_existing_and_updates_ours(monkeypatch):
    # A typed backend whose postgres_resource() is named "postgres".
    backend = LakebaseBackend(
        project="p",
        branch="production",
        endpoint_id="primary",
        database="db-new",
        schema="public",
        tables=(),
        resource_name="postgres",
    )
    existing = {
        "resources": [
            {"name": "sql-warehouse", "sql_warehouse": {"id": "w1"}},  # user-owned, must survive
            {"name": "postgres", "postgres": {"database": "db-old"}},  # ours, must be replaced
        ]
    }
    calls = {}

    def fake_databricks(args, profile, **kw):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(existing), stderr="")
        # the update call
        calls["update"] = args
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    client = AppsClient("prof", runner=fake_databricks)
    assert client.attach_postgres_backends("myapp", [backend]) is None

    payload = json.loads(calls["update"][calls["update"].index("--json") + 1])
    names = [r["name"] for r in payload["app"]["resources"]]
    assert "sql-warehouse" in names  # preserved
    assert names.count("postgres") == 1  # not duplicated
    pg = next(r for r in payload["app"]["resources"] if r["name"] == "postgres")
    assert "db-new" in pg["postgres"]["database"]  # updated to ours
