"""Unit tests for the deploy bug fixes (ML-69245/69246/69248/69259)."""

from __future__ import annotations

import json
import types
from unittest import mock

import pytest
from click.testing import CliRunner

from databricks_mason import app_resources as sa
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


# --- ML-69248: session store pre-validation ----------------------------------


def test_validate_stores_raises_when_session_store_missing():
    client = mock.Mock()
    client.get_session_store.side_effect = AgentCliError(
        "session store not found", error_code="NOT_FOUND"
    )
    with pytest.raises(AgentCliError) as exc:
        deploy_mod.validate_stores(client, memory_store=None, session_store="ghost")
    assert "does not exist" in str(exc.value)
    client.get_session_store.assert_called_once_with("ghost")


# --- ML-69245: postgres resources are MERGED, not replaced -------------------


def test_apply_postgres_resources_preserves_existing_and_updates_ours(monkeypatch):
    # A typed backend whose postgres_resource() is named "postgres".
    backend = sa.LakebaseBackend(
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

    monkeypatch.setattr(sa, "_databricks", fake_databricks)
    assert sa.apply_postgres_resources("myapp", [backend], "prof") is None

    payload = json.loads(calls["update"][calls["update"].index("--json") + 1])
    names = [r["name"] for r in payload["resources"]]
    assert "sql-warehouse" in names  # preserved
    assert names.count("postgres") == 1  # not duplicated
    pg = next(r for r in payload["resources"] if r["name"] == "postgres")
    assert "db-new" in pg["postgres"]["database"]  # updated to ours


# --- ML-69653: fail fast when databricks-mason is pinned to a local (file://) path ------------

_FILE_SRC = (
    '[project]\nname = "x"\ndependencies = ["databricks-mason[runtime]>=0.1.5"]\n'
    "[tool.uv.sources]\n"
    'databricks-mason = {git = "file:///home/dev/checkout", rev = "abc", '
    'subdirectory = "integrations/mason"}\n'
)
_PATH_SRC = (
    '[project]\nname = "x"\ndependencies = ["databricks-mason[runtime]>=0.1.5"]\n'
    "[tool.uv.sources]\n"
    'databricks-mason = {path = "../checkout/integrations/mason"}\n'
)
_GIT_SRC = (
    '[project]\nname = "x"\ndependencies = ["databricks-mason[runtime]>=0.1.5"]\n'
    "[tool.uv.sources]\n"
    'databricks-mason = {git = "https://github.com/databricks/databricks-ai-bridge.git", '
    'rev = "abc", subdirectory = "integrations/mason"}\n'
)
_REGISTRY_SRC = '[project]\nname = "x"\ndependencies = ["databricks-mason[runtime]>=0.1.5"]\n'


def _proj(tmp_path, pyproject_text=None):
    if pyproject_text is not None:
        (tmp_path / "pyproject.toml").write_text(pyproject_text)
    return tmp_path


def test_local_mason_source_detects_file_uri(tmp_path):
    assert deploy_mod._local_mason_source(_proj(tmp_path, _FILE_SRC)) == "file:///home/dev/checkout"


def test_local_mason_source_detects_path(tmp_path):
    assert (
        deploy_mod._local_mason_source(_proj(tmp_path, _PATH_SRC))
        == "../checkout/integrations/mason"
    )


def test_local_mason_source_none_for_remote_git(tmp_path):
    assert deploy_mod._local_mason_source(_proj(tmp_path, _GIT_SRC)) is None


def test_local_mason_source_none_for_registry(tmp_path):
    assert deploy_mod._local_mason_source(_proj(tmp_path, _REGISTRY_SRC)) is None


def test_local_mason_source_none_when_no_pyproject(tmp_path):
    assert deploy_mod._local_mason_source(tmp_path) is None


def test_check_deployable_raises_for_local_source(tmp_path):
    with pytest.raises(AgentCliError, match="local path"):
        deploy_mod._check_deployable_mason_source(_proj(tmp_path, _FILE_SRC))


def test_check_deployable_passes_for_remote_git(tmp_path):
    deploy_mod._check_deployable_mason_source(_proj(tmp_path, _GIT_SRC))  # no raise


def test_deploy_fails_fast_on_local_source_without_touching_client(tmp_path, monkeypatch):
    # A file://-pinned project must be rejected before any client/sync work happens.
    (tmp_path / "agent.toml").write_text('schema_version = 1\n[agent]\nframework = "langgraph"\n')
    (tmp_path / "pyproject.toml").write_text(_FILE_SRC)
    (tmp_path / "app.yaml").write_text('command: ["uv", "run", "start-server"]\n')
    called = {"client": False, "databricks": False}

    class _C(_Ctx):
        def client(self):
            called["client"] = True
            raise AssertionError("client() must not be reached")

    monkeypatch.setattr(
        deploy_mod, "_databricks", lambda *a, **k: called.__setitem__("databricks", True)
    )
    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(tmp_path)], obj=_C())
    assert result.exit_code != 0
    assert "local path" in result.output
    assert called == {"client": False, "databricks": False}
