"""Unit tests for the deploy bug fixes (ML-69245/69246/69248/69259)."""

from __future__ import annotations

import json
import types

import pytest
from click.testing import CliRunner

from databricks_mason import deploy as deploy_mod
from databricks_mason.errors import AgentCliError


class _Ctx:
    profile = "prof"
    output = "text"


# --- ML-69259 / 69247: deployment-name validation ---------------------------


@pytest.mark.parametrize("bad", ["", "   ", "../../x", "a/b", "a b", "..", "with\ttab"])
def test_validate_deployment_name_rejects_unsafe(bad):
    with pytest.raises(AgentCliError):
        deploy_mod._validate_deployment_name(bad)


def test_validate_deployment_name_accepts_good():
    assert deploy_mod._validate_deployment_name("mason-agent-1") == "mason-agent-1"


def test_validate_deployment_name_rejects_too_long():
    with pytest.raises(AgentCliError, match="too long"):
        deploy_mod._validate_deployment_name("mason-" + "a" * 25)  # 31 chars


# --- `mason-` deployment prefix + list filtering -----------------------------


def test_prefixed_name_adds_prefix_when_absent():
    assert deploy_mod._prefixed_name("foo") == "mason-foo"


def test_prefixed_name_is_idempotent():
    assert deploy_mod._prefixed_name("mason-foo") == "mason-foo"


def test_deployments_list_shows_only_mason_apps(monkeypatch):
    apps = {"apps": [{"name": "mason-foo"}, {"name": "someone-else-app"}, {"name": "mason-bar"}]}
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=json.dumps(apps), stderr=""),
    )

    class _JsonCtx:
        profile = "prof"
        output = "json"

    result = CliRunner().invoke(deploy_mod.deployments_list, [], obj=_JsonCtx())
    assert result.exit_code == 0, result.output
    names = {a["name"] for a in json.loads(result.output)}
    assert names == {"mason-foo", "mason-bar"}


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
