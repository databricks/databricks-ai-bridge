"""Unit tests for `mason tracing`: configure/disable binding, experiment provisioning, list/get.

Tracing is managed MLflow tracing, `experiment_id`-centric. The pure surface (default_experiment_name,
experiment_url) is tested directly; the mlflow-backed paths are exercised with a mocked
`_mlflow`/`_set_tracking_uri` (the hermetic env shouldn't touch a real workspace).
"""

from __future__ import annotations

import json
import pathlib
from unittest import mock

import pytest
from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject
from databricks_mason.cli import tracing as tracing_mod
from databricks_mason.errors import AgentCliError

_AGENT_TOML = 'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "mason"\n'


class _Ctx:
    """Stand-in for CliContext: tracing reads .profile / .output, and .client() for the list default."""

    def __init__(self, output: str = "text", profile=None, user="me@example.com"):
        self.output = output
        self.profile = profile
        self._user = user

    def client(self):
        return mock.Mock(current_user=self._user, host="https://ws")


def _project(tmp_path: pathlib.Path, *, experiment_id: str | None = None, disabled: bool = False):
    body = _AGENT_TOML
    if experiment_id or disabled:
        body += "\n[tracing]\n"
        if experiment_id:
            body += f'experiment_id = "{experiment_id}"\n'
        if disabled:
            body += "disabled = true\n"
    (tmp_path / "agent.toml").write_text(body)
    return tmp_path


# --- pure surface -----------------------------------------------------------


def test_default_experiment_name_is_per_project_under_user_home():
    assert (
        tracing_mod.default_experiment_name("me@x.com", "my-agent")
        == "/Users/me@x.com/mason-traces/my-agent"
    )


def test_default_experiment_name_requires_project():
    with pytest.raises(AgentCliError):
        tracing_mod.default_experiment_name("me@x.com", None)


def test_experiment_url_builds_traces_tab_link():
    assert (
        tracing_mod.experiment_url("https://ws.databricks.com/", "123")
        == "https://ws.databricks.com/ml/experiments/123?compareRunsMode=TRACES"
    )
    assert tracing_mod.experiment_url(None, "123") is None
    assert tracing_mod.experiment_url("unknown", "123") is None


# --- create_experiment_idempotent ------------------------------------------------------


def test_create_experiment_idempotent_creates_parent_dir_for_nested_path():
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = None  # doesn't exist yet
    mlflow.create_experiment.return_value = "eid-1"
    client = mock.Mock()
    with mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow):
        eid = tracing_mod.create_experiment_idempotent(
            None, client, "/Users/me@x.com/mason-traces/demo"
        )
    assert eid == "eid-1"
    # the intermediate workspace folder is created before the experiment (mlflow won't make it)
    client.ensure_workspace_dir.assert_called_once_with("/Users/me@x.com/mason-traces")


def test_create_experiment_idempotent_reuses_existing_without_mkdir():
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="eid-2")
    client = mock.Mock()
    with mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow):
        assert tracing_mod.create_experiment_idempotent(None, client, "/Shared/x") == "eid-2"
    client.ensure_workspace_dir.assert_not_called()  # existing experiment -> no dir work
    mlflow.create_experiment.assert_not_called()


# --- configure / disable ----------------------------------------------------


def test_configure_pins_experiment_id(tmp_path: pathlib.Path):
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment.return_value = mock.Mock(tags={})  # exists, managed (no UC tag)
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_configure,
            ["--experiment", "123", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"experiment_id": "123", "disabled": False}
    project = AgentProject.load(tmp_path)
    assert project.trace_experiment_id == "123"
    assert project.trace_workspace_host == "https://ws"


def test_configure_rejects_unknown_experiment_id(tmp_path: pathlib.Path):
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment.return_value = None  # no such experiment
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_configure,
            ["--experiment", "nope", "--source", str(tmp_path)],
            obj=_Ctx(),
        )
    assert result.exit_code != 0
    assert "No MLflow experiment" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_id is None  # nothing persisted


def test_configure_rejects_uc_backed_experiment(tmp_path: pathlib.Path):
    # mason supports managed tracing only; a UC-backed experiment (carries the UC destination tag)
    # is rejected up front rather than silently wiring a config that fails at read/deploy.
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment.return_value = mock.Mock(
        tags={"mlflow.experiment.databricksTraceDestinationPath": "cat.schema"}
    )
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_configure,
            ["--experiment", "uc-1", "--source", str(tmp_path)],
            obj=_Ctx(),
        )
    assert result.exit_code != 0
    assert "UC-backed MLflow tracing is not supported" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_id is None  # nothing persisted


def test_configure_default_enables_per_project_offline(tmp_path: pathlib.Path):
    # No --experiment: enables the per-project default. Pure agent.toml write, no mlflow call.
    _project(tmp_path, disabled=True)
    result = CliRunner().invoke(
        tracing_mod.tracing_configure, ["--source", str(tmp_path)], obj=_Ctx()
    )
    assert result.exit_code == 0, result.output
    project = AgentProject.load(tmp_path)
    assert project.trace_experiment_id is None
    assert project.trace_disabled is False  # re-enabled


def test_disable_writes_disabled(tmp_path: pathlib.Path):
    _project(tmp_path, experiment_id="123")
    result = CliRunner().invoke(
        tracing_mod.tracing_disable, ["--source", str(tmp_path)], obj=_Ctx(output="json")
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"disabled": True}
    assert AgentProject.load(tmp_path).trace_disabled is True


# --- list / get -------------------------------------------------------------


def _trace(trace_id):
    import types

    return types.SimpleNamespace(
        info=types.SimpleNamespace(
            trace_id=trace_id, status="OK", execution_time_ms=5, timestamp_ms=1
        )
    )


def test_list_searches_by_explicit_experiment_id(tmp_path: pathlib.Path):
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.search_traces.return_value = [_trace("tr-1")]
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--experiment", "eid-9", "--limit", "7", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    kwargs = mlflow.search_traces.call_args.kwargs
    assert kwargs["locations"] == ["eid-9"]
    assert kwargs["max_results"] == 7
    assert json.loads(result.output)[0]["trace_id"] == "tr-1"


def test_list_defaults_to_projects_pinned_experiment(tmp_path: pathlib.Path):
    _project(tmp_path, experiment_id="p1")
    mlflow = mock.Mock()
    mlflow.search_traces.return_value = []
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    assert mlflow.search_traces.call_args.kwargs["locations"] == ["p1"]


def test_list_ignores_project_pin_from_another_workspace(tmp_path: pathlib.Path):
    manifest = _project(tmp_path, experiment_id="dogfood-id") / "agent.toml"
    manifest.write_text(
        manifest.read_text() + 'workspace_host = "https://dogfood.databricks.com"\n'
    )
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="tilefood-id")
    mlflow.search_traces.return_value = []
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )

    assert result.exit_code == 0, result.output
    assert mlflow.search_traces.call_args.kwargs["locations"] == ["tilefood-id"]


def test_list_empty_when_no_experiment_exists(tmp_path: pathlib.Path):
    # No pinned id and the per-project experiment isn't created yet -> nothing traced, list is empty.
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = None
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []
    mlflow.search_traces.assert_not_called()


def test_get_reports_missing_trace(tmp_path: pathlib.Path):
    mlflow = mock.Mock()
    mlflow.get_trace.return_value = None
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(tracing_mod.tracing_get, ["tr-x"], obj=_Ctx())
    assert result.exit_code != 0
    assert "No trace found" in result.output


def test_status_str_handles_enum_like_and_none():
    class _EnumLike:
        name = "OK"

    assert tracing_mod._status_str(_EnumLike()) == "OK"
    assert tracing_mod._status_str(None) is None


# --- local dev tracing server (`mason dev`) ---------------------------------


def test_start_local_tracing_server_launches_sqlite_server(tmp_path: pathlib.Path, monkeypatch):
    # Launch `uvx mlflow server` backed by sqlite under .mason/, returning the MLFLOW_* env pointing
    # at it (bare experiment name = project dir, no workspace path / username needed).
    monkeypatch.setattr(tracing_mod, "_free_port", lambda: 5599)
    captured: dict = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return mock.Mock()

    monkeypatch.setattr(tracing_mod.subprocess, "Popen", _fake_popen)
    server, env = tracing_mod.start_local_tracing_server(tmp_path)
    assert server is not None
    assert env["MLFLOW_TRACKING_URI"] == "http://127.0.0.1:5599"
    assert env["MLFLOW_EXPERIMENT_NAME"] == tmp_path.resolve().name
    assert (tmp_path / ".mason").is_dir()
    joined = " ".join(captured["cmd"])
    assert captured["cmd"][0] == "uvx" and "server" in captured["cmd"]
    assert "sqlite:///" in joined and ".mason/mlflow.db" in joined


def test_start_local_tracing_server_degrades_when_launch_fails(tmp_path: pathlib.Path, monkeypatch):
    # If the server process can't be spawned (e.g. uv missing), degrade to (None, {}) so `mason dev`
    # runs without traces rather than aborting.
    monkeypatch.setattr(tracing_mod, "_free_port", lambda: 5599)

    def _boom(cmd, **kwargs):
        raise OSError("uvx not found")

    monkeypatch.setattr(tracing_mod.subprocess, "Popen", _boom)
    server, env = tracing_mod.start_local_tracing_server(tmp_path)
    assert server is None and env == {}
