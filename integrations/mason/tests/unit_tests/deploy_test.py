"""Unit tests for the deploy wrapper: app.yaml env injection, store reuse, deploy argv."""

from __future__ import annotations

import json
import pathlib
import types
from unittest import mock

import yaml
from click.testing import CliRunner

from databricks_mason import deploy as deploy_mod
from databricks_mason.errors import AgentCliError


def test_upsert_manifest_env_scaffolds_when_missing(tmp_path: pathlib.Path):
    scaffolded = deploy_mod._upsert_manifest_env(
        tmp_path, {"AGENT_MEMORY_STORE": "memory-stores/x"}
    )
    assert scaffolded is True
    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert {"name": "AGENT_MEMORY_STORE", "value": "memory-stores/x"} in doc["env"]
    assert "command" in doc  # placeholder written


def test_upsert_manifest_env_updates_existing(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["uvicorn", "app:app"],
                "env": [{"name": "AGENT_MEMORY_STORE", "value": "old"}],
            }
        )
    )
    scaffolded = deploy_mod._upsert_manifest_env(
        tmp_path, {"AGENT_MEMORY_STORE": "new", "AGENT_SESSION_STORE": "s"}
    )
    assert scaffolded is False
    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert doc["command"] == ["uvicorn", "app:app"]  # preserved
    by_name = {e["name"]: e["value"] for e in doc["env"]}
    assert by_name == {"AGENT_MEMORY_STORE": "new", "AGENT_SESSION_STORE": "s"}


def test_ensure_session_store_reuses_on_already_exists():
    client = mock.Mock()
    client.create_session_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.get_session_store.return_value = {"session_store_name": "s"}
    assert deploy_mod._ensure_session_store(client, "s") == {"session_store_name": "s"}


class _FakeClient:
    host = "https://ws"
    current_user = "me@example.com"

    def get_memory_store(self, name):
        return {"name": f"memory-stores/{name}"}


class _FakeCtx:
    profile = "prof"
    output = "text"

    def client(self):
        return _FakeClient()


def test_deploy_drives_sync_and_apps_deploy(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--with-memory-store", "mem"],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    ws = "/Workspace/Users/me@example.com/mason_deployments/myapp"
    # uv.lock is excluded so the build resolves fresh against its own index (not the dev machine's).
    assert ["sync", str(src), ws, "--exclude", "uv.lock"] in calls
    assert ["apps", "deploy", "myapp", "--source-code-path", ws] in calls
    assert "AGENT_MEMORY_STORE" in (src / "app.yaml").read_text()


def test_deploy_with_traces_injects_tracing_env(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        [
            "myapp",
            "--source",
            str(src),
            "--with-traces",
            "cat.schema",
            "--traces-experiment",
            "/Shared/x",
        ],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    doc = yaml.safe_load((src / "app.yaml").read_text())
    env = {e["name"]: e["value"] for e in doc["env"]}
    assert env["MLFLOW_TRACING_DESTINATION"] == "cat.schema"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "/Shared/x"


class _JsonCtx:
    profile = "prof"
    output = "json"


def test_lifecycle_commands_honor_json_output(monkeypatch):
    # start/stop/delete must emit JSON (not the Rich success panel) under --output json.
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    for command, key in (
        (deploy_mod.deployments_start, "started"),
        (deploy_mod.deployments_stop, "stopped"),
        (deploy_mod.deployments_delete, "deleted"),
    ):
        result = CliRunner().invoke(command, ["myapp"], obj=_JsonCtx())
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {key: "myapp"}
