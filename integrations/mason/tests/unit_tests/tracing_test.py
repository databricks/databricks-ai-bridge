"""Unit tests for `mason tracing`: configure/disable binding, experiment provisioning, list/get.

Tracing is managed MLflow tracing, bound by experiment **name** (portable across workspaces). The pure
surface (default_experiment_name, experiment_url) is tested directly; the mlflow-backed paths are
exercised with a mocked `_mlflow`/`_set_tracking_uri` (the hermetic env shouldn't touch a real
workspace).
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


def _project(tmp_path: pathlib.Path, *, experiment_name: str | None = None, disabled: bool = False):
    body = _AGENT_TOML
    if experiment_name or disabled:
        body += "\n[tracing]\n"
        if experiment_name:
            body += f'experiment_name = "{experiment_name}"\n'
        if disabled:
            body += "disabled = true\n"
    (tmp_path / "agent.toml").write_text(body)
    return tmp_path


# --- pure surface -----------------------------------------------------------


def test_default_experiment_name_is_per_project_under_shared():
    # Under /Shared (username-free, workspace-independent), not the user's home.
    assert tracing_mod.default_experiment_name("my-agent") == "/Shared/mason_traces/my-agent"
    # an optional token (shared with the store names) disambiguates like-named projects
    assert (
        tracing_mod.default_experiment_name("My Agent!", "abc123")
        == "/Shared/mason_traces/my-agent-abc123"
    )


def test_default_experiment_name_requires_project():
    with pytest.raises(AgentCliError):
        tracing_mod.default_experiment_name(None)


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
        eid = tracing_mod.create_experiment_idempotent(None, client, "/Shared/mason_traces/demo")
    assert eid == "eid-1"
    # the intermediate workspace folder is created before the experiment (mlflow won't make it)
    client.ensure_workspace_dir.assert_called_once_with("/Shared/mason_traces")


def test_create_experiment_idempotent_reuses_existing_without_mkdir():
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="eid-2")
    client = mock.Mock()
    with mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow):
        assert tracing_mod.create_experiment_idempotent(None, client, "/Shared/x") == "eid-2"
    client.ensure_workspace_dir.assert_not_called()  # existing experiment -> no dir work
    mlflow.create_experiment.assert_not_called()


# --- configure / disable ----------------------------------------------------


def test_configure_sets_experiment_name(tmp_path: pathlib.Path):
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = (
        None  # not created yet — allowed (deploy creates it)
    )
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_configure,
            ["--experiment-name", "/Shared/mason_traces/mine", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "experiment_name": "/Shared/mason_traces/mine",
        "disabled": False,
    }
    assert AgentProject.load(tmp_path).trace_experiment_name == "/Shared/mason_traces/mine"


def test_configure_rejects_non_absolute_name(tmp_path: pathlib.Path):
    # An experiment name must be an absolute workspace path; a bare name is rejected up front.
    _project(tmp_path)
    result = CliRunner().invoke(
        tracing_mod.tracing_configure,
        ["--experiment-name", "not-a-path", "--source", str(tmp_path)],
        obj=_Ctx(),
    )
    assert result.exit_code != 0
    assert "absolute workspace path" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_name is None  # nothing persisted


def test_configure_rejects_uc_backed_experiment(tmp_path: pathlib.Path):
    # mason supports managed tracing only; if the name already resolves to a UC-backed experiment
    # (carries the UC destination tag) it's rejected rather than wiring a config that fails later.
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = mock.Mock(
        tags={"mlflow.experiment.databricksTraceDestinationPath": "cat.schema"}
    )
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_configure,
            ["--experiment-name", "/Shared/uc", "--source", str(tmp_path)],
            obj=_Ctx(),
        )
    assert result.exit_code != 0
    assert "UC-backed MLflow tracing is not supported" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_name is None  # nothing persisted


def test_configure_default_enables_per_project_offline(tmp_path: pathlib.Path):
    # No --experiment-name: clears any explicit name and re-enables the default. Pure agent.toml
    # write, no mlflow call.
    _project(tmp_path, disabled=True)
    result = CliRunner().invoke(
        tracing_mod.tracing_configure, ["--source", str(tmp_path)], obj=_Ctx()
    )
    assert result.exit_code == 0, result.output
    project = AgentProject.load(tmp_path)
    assert project.trace_experiment_name is None
    assert project.trace_disabled is False  # re-enabled


def test_configure_by_experiment_id_stores_resolved_name(tmp_path: pathlib.Path):
    # --experiment-id is a convenience: resolve the id to the experiment's name and store the NAME.
    _project(tmp_path)
    mlflow = mock.Mock()
    experiment = mock.Mock(tags={})
    experiment.name = (
        "/Shared/mason_traces/from-id"  # set explicitly (Mock(name=) is special-cased)
    )
    mlflow.get_experiment.return_value = experiment
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_configure,
            ["--experiment-id", "123", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "experiment_name": "/Shared/mason_traces/from-id",
        "disabled": False,
    }
    # stored as the resolved NAME, never the id
    assert AgentProject.load(tmp_path).trace_experiment_name == "/Shared/mason_traces/from-id"


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
            ["--experiment-id", "nope", "--source", str(tmp_path)],
            obj=_Ctx(),
        )
    assert result.exit_code != 0
    assert "No MLflow experiment" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_name is None


def test_configure_rejects_both_name_and_id(tmp_path: pathlib.Path):
    _project(tmp_path)
    result = CliRunner().invoke(
        tracing_mod.tracing_configure,
        ["--experiment-name", "/Shared/x", "--experiment-id", "1", "--source", str(tmp_path)],
        obj=_Ctx(),
    )
    assert result.exit_code != 0
    assert "not both" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_name is None


def test_disable_writes_disabled(tmp_path: pathlib.Path):
    _project(tmp_path, experiment_name="/Shared/mason_traces/x")
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


def test_list_by_explicit_experiment_id(tmp_path: pathlib.Path):
    # --experiment-id targets that workspace experiment directly (no project resolution).
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.search_traces.return_value = [_trace("tr-1")]
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--experiment-id", "eid-9", "--limit", "7", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    kwargs = mlflow.search_traces.call_args.kwargs
    assert kwargs["locations"] == ["eid-9"]
    assert kwargs["max_results"] == 7
    assert json.loads(result.output)[0]["trace_id"] == "tr-1"


def test_list_by_explicit_experiment_name(tmp_path: pathlib.Path):
    # --experiment-name is resolved to its id in the current workspace, then read (the --store analog).
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="by-name-1", tags={})
    mlflow.search_traces.return_value = [_trace("tr-2")]
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--experiment-name", "/Shared/mason_traces/mine", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    mlflow.get_experiment_by_name.assert_called_once_with("/Shared/mason_traces/mine")
    assert mlflow.search_traces.call_args.kwargs["locations"] == ["by-name-1"]
    assert json.loads(result.output)[0]["trace_id"] == "tr-2"


def test_list_rejects_both_name_and_id(tmp_path: pathlib.Path):
    _project(tmp_path)
    result = CliRunner().invoke(
        tracing_mod.tracing_list,
        ["--experiment-name", "/Shared/x", "--experiment-id", "1", "--source", str(tmp_path)],
        obj=_Ctx(),
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_list_defaults_to_projects_bound_experiment(tmp_path: pathlib.Path):
    # No explicit --experiment: resolve the project's bound name to an id in the current workspace.
    _project(tmp_path, experiment_name="/Shared/mason_traces/demo")
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="p1")
    mlflow.search_traces.return_value = []
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    mlflow.get_experiment_by_name.assert_called_once_with("/Shared/mason_traces/demo")
    assert mlflow.search_traces.call_args.kwargs["locations"] == ["p1"]


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


def test_get_by_explicit_experiment_id(tmp_path: pathlib.Path):
    # --experiment-id points get at that workspace store (no project resolution, no local fallback).
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_trace.return_value = _trace("tr-9")
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_get,
            ["tr-9", "--experiment-id", "eid-9", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["trace_id"] == "tr-9"
    assert not any(
        str(c.args[0]).startswith("sqlite:///") for c in mlflow.set_tracking_uri.call_args_list
    )


def test_get_rejects_both_name_and_id(tmp_path: pathlib.Path):
    result = CliRunner().invoke(
        tracing_mod.tracing_get,
        ["tr-9", "--experiment-name", "/Shared/x", "--experiment-id", "1"],
        obj=_Ctx(),
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_list_errors_when_explicit_name_missing(tmp_path: pathlib.Path):
    # A typed --experiment-name that doesn't exist errors (not silently empty), so a typo isn't
    # mistaken for an empty experiment. Only the project default is allowed to be absent.
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = None
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--experiment-name", "/Shared/nope", "--source", str(tmp_path)],
            obj=_Ctx(),
        )
    assert result.exit_code != 0
    assert "No MLflow experiment named" in result.output
    mlflow.search_traces.assert_not_called()


def test_list_errors_when_explicit_id_missing(tmp_path: pathlib.Path):
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment.return_value = None
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--experiment-id", "nope", "--source", str(tmp_path)],
            obj=_Ctx(),
        )
    assert result.exit_code != 0
    assert "No MLflow experiment found with id" in result.output
    mlflow.search_traces.assert_not_called()


def test_get_errors_when_explicit_name_missing():
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = None
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_get, ["tr-x", "--experiment-name", "/Shared/nope"], obj=_Ctx()
        )
    assert result.exit_code != 0
    assert "No MLflow experiment named" in result.output
    mlflow.get_trace.assert_not_called()


def test_list_reads_local_dev_store_when_not_provisioned(tmp_path: pathlib.Path):
    # No workspace experiment yet, but a local `mason dev` store exists -> list reads the local
    # traces, consistent with where `mason dev` traced pre-deploy.
    _project(tmp_path, experiment_name="/Shared/mason_traces/demo")
    (tmp_path / ".mason").mkdir()
    (tmp_path / ".mason" / "mlflow.db").write_text("")  # only needs to exist
    mlflow = mock.Mock()
    # workspace miss, then local hit (bare project-name experiment in the sqlite store)
    mlflow.get_experiment_by_name.side_effect = [None, mock.Mock(experiment_id="local-1", tags={})]
    mlflow.search_traces.return_value = [_trace("tr-local")]
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["trace_id"] == "tr-local"
    # read from the local sqlite store, not the workspace
    assert any(
        str(c.args[0]).startswith("sqlite:///") for c in mlflow.set_tracking_uri.call_args_list
    )


def test_get_reads_local_dev_store_when_not_provisioned(tmp_path: pathlib.Path):
    # get resolves its store the same way as list: the local dev store when the workspace experiment
    # isn't provisioned yet.
    _project(tmp_path, experiment_name="/Shared/mason_traces/demo")
    (tmp_path / ".mason").mkdir()
    (tmp_path / ".mason" / "mlflow.db").write_text("")
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.side_effect = [None, mock.Mock(experiment_id="local-1", tags={})]
    mlflow.get_trace.return_value = _trace("tr-local")
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_get,
            ["tr-local", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["trace_id"] == "tr-local"
    assert any(
        str(c.args[0]).startswith("sqlite:///") for c in mlflow.set_tracking_uri.call_args_list
    )


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
