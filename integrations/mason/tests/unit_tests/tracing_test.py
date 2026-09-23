"""Unit tests for `mason tracing`: configure/disable binding, experiment provisioning, list/get.

Tracing is managed MLflow tracing, bound by experiment **name** (portable across workspaces). The pure
surface (default_experiment_name, experiment_url) is tested directly; the mlflow-backed paths are
exercised with a mocked `_mlflow`/`_set_tracking_uri` (the hermetic env shouldn't touch a real
workspace).
"""

from __future__ import annotations

import json
import os
import pathlib
import types
from unittest import mock

import pytest
from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject
from databricks_mason.cli import tracing as tracing_mod
from databricks_mason.errors import AgentCliError

_AGENT_TOML = 'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "mason"\n'


def _not_found_exc():
    """The MlflowException MLflow's id lookup raises for a missing experiment (not a None return)."""
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    return MlflowException("Experiment does not exist.", error_code=RESOURCE_DOES_NOT_EXIST)


class _Ctx:
    """Stand-in for CliContext: tracing reads .profile / .output, and .client() for the list default."""

    def __init__(self, output: str = "text", profile=None, user="me@example.com"):
        self.output = output
        self.profile = profile
        self._user = user

    def client(self):
        return mock.Mock(current_user=self._user, host="https://ws")


def _project(tmp_path: pathlib.Path, *, experiment_name: str | None = None):
    body = _AGENT_TOML
    if experiment_name:
        body += f'\n[tracing]\nexperiment_name = "{experiment_name}"\n'
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
        result = tracing_mod.create_experiment_idempotent(None, client, "/Shared/mason_traces/demo")
    # a newly created experiment is always managed -> no UC tables
    assert result == ("eid-1", tracing_mod.MLflowTraceTables())
    # the intermediate workspace folder is created before the experiment (mlflow won't make it)
    client.ensure_workspace_dir.assert_called_once_with("/Shared/mason_traces")


def test_create_experiment_idempotent_reuses_existing_without_mkdir():
    mlflow = mock.Mock()
    # A managed (non-UC) experiment carries no UC destination tag.
    mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="eid-2", tags={})
    client = mock.Mock()
    with mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow):
        assert tracing_mod.create_experiment_idempotent(None, client, "/Shared/x") == (
            "eid-2",
            tracing_mod.MLflowTraceTables(),
        )
    client.ensure_workspace_dir.assert_not_called()  # existing experiment -> no dir work
    mlflow.create_experiment.assert_not_called()


# The experiment tags MLflow sets on a UC-backed experiment (values verified against a real
# e2-dogfood UC experiment: per-kind base-table tags plus the "unified" tag naming a read-side VIEW).
_DEST_TAG = "mlflow.experiment.databricksTraceDestinationPath"
_SPAN_TAG = "mlflow.experiment.databricksTraceSpanStorageTable"
_LOG_TAG = "mlflow.experiment.databricksTraceLogStorageTable"
_ANNOTATION_TAG = "mlflow.experiment.databricksTraceAnnotationStorageTable"
_UNIFIED_TAG = "mlflow.experiment.databricksTraceStorageTable"


def test_create_experiment_idempotent_returns_uc_tables():
    # A hand-edited agent.toml can name a UC-backed experiment; deploy supports exporting to one, so
    # provisioning returns its id together with the UC OTEL tables the app must be granted MODIFY on.
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = types.SimpleNamespace(
        experiment_id="eid-uc",
        tags={
            _DEST_TAG: "cat.schema.pfx",
            _SPAN_TAG: "cat.schema.pfx_otel_spans",
            _LOG_TAG: "cat.schema.pfx_otel_logs",
            _ANNOTATION_TAG: "cat.schema.pfx_otel_annotations",
            _UNIFIED_TAG: "cat.schema.mlflow_experiment_trace_unified",
        },
    )
    client = mock.Mock()
    with mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow):
        result = tracing_mod.create_experiment_idempotent(None, client, "/Shared/uc")
    # The per-kind base tables; the "unified" VIEW tag is excluded.
    assert result == (
        "eid-uc",
        tracing_mod.MLflowTraceTables(
            spans="cat.schema.pfx_otel_spans",
            logs="cat.schema.pfx_otel_logs",
            annotations="cat.schema.pfx_otel_annotations",
        ),
    )
    client.ensure_workspace_dir.assert_not_called()  # existing experiment -> no dir work
    mlflow.create_experiment.assert_not_called()


# --- uc_trace_tables --------------------------------------------------------


def _uc_experiment(**extra_tags):
    """A UC-backed experiment (carries the UC destination tag) with any extra storage-table tags."""
    return types.SimpleNamespace(
        experiment_id="eid-uc", tags={_DEST_TAG: "cat.schema.pfx", **extra_tags}
    )


def test_uc_trace_tables_empty_for_managed_experiment():
    experiment = types.SimpleNamespace(experiment_id="eid-1", tags={})
    tables = tracing_mod.uc_trace_tables(experiment)
    assert tables == tracing_mod.MLflowTraceTables()
    assert not tables  # empty tables are falsy
    assert tables.otel_tables() == ()


def test_uc_trace_tables_reads_all_base_table_tags_excluding_unified_view():
    # Current layout: per-kind storage-table tags (spans/logs/annotations) are the source of truth;
    # the "unified" tag names a read-side VIEW and must NOT be granted MODIFY.
    experiment = _uc_experiment(
        **{
            _SPAN_TAG: "cat.schema.pfx_otel_spans",
            _LOG_TAG: "cat.schema.pfx_otel_logs",
            _ANNOTATION_TAG: "cat.schema.pfx_otel_annotations",
            _UNIFIED_TAG: "cat.schema.mlflow_experiment_trace_unified",
        }
    )
    tables = tracing_mod.uc_trace_tables(experiment)
    assert tables.spans == "cat.schema.pfx_otel_spans"
    assert tables.logs == "cat.schema.pfx_otel_logs"
    assert tables.annotations == "cat.schema.pfx_otel_annotations"
    assert tables.otel_tables() == (
        ("spans", "cat.schema.pfx_otel_spans"),
        ("logs", "cat.schema.pfx_otel_logs"),
        ("annotations", "cat.schema.pfx_otel_annotations"),
    )


def test_uc_trace_tables_reads_span_and_log_only_layout():
    # An older UC experiment with just spans + logs (no annotations tag) still excludes the unified tag.
    experiment = _uc_experiment(
        **{
            _SPAN_TAG: "cat.schema.pfx_otel_spans",
            _LOG_TAG: "cat.schema.pfx_otel_logs",
            _UNIFIED_TAG: "cat.schema.mlflow_experiment_trace_unified",
        }
    )
    tables = tracing_mod.uc_trace_tables(experiment)
    assert tables.annotations is None
    assert tables.otel_tables() == (
        ("spans", "cat.schema.pfx_otel_spans"),
        ("logs", "cat.schema.pfx_otel_logs"),
    )


def test_uc_trace_tables_falls_back_to_derived_names_for_three_part_path():
    # No per-kind storage tags: derive spans/logs/annotations from a 3-part destination path.
    experiment = types.SimpleNamespace(tags={_DEST_TAG: "cat.schema.pfx"})
    assert tracing_mod.uc_trace_tables(experiment) == tracing_mod.MLflowTraceTables(
        spans="cat.schema.pfx_otel_spans",
        logs="cat.schema.pfx_otel_logs",
        annotations="cat.schema.pfx_otel_annotations",
    )


def test_uc_trace_tables_falls_back_to_legacy_fixed_names_for_two_part_path():
    # Legacy schema-linked layout: a 2-part destination path yields the fixed table names
    # (annotations included for symmetry with the 3-part fallback - MODIFY on a table that may not
    # exist is harmless).
    experiment = types.SimpleNamespace(tags={_DEST_TAG: "cat.schema"})
    assert tracing_mod.uc_trace_tables(experiment) == tracing_mod.MLflowTraceTables(
        spans="cat.schema.mlflow_experiment_trace_otel_spans",
        logs="cat.schema.mlflow_experiment_trace_otel_logs",
        annotations="cat.schema.mlflow_experiment_trace_otel_annotations",
    )


def test_uc_trace_tables_empty_when_destination_tag_is_none_or_empty():
    # A present-but-valueless destination tag must not crash the fallback derivation.
    empty = tracing_mod.MLflowTraceTables()
    assert tracing_mod.uc_trace_tables(types.SimpleNamespace(tags={_DEST_TAG: None})) == empty
    assert tracing_mod.uc_trace_tables(types.SimpleNamespace(tags={_DEST_TAG: ""})) == empty


def test_uc_trace_tables_empty_when_destination_path_is_unusable():
    experiment = types.SimpleNamespace(tags={_DEST_TAG: "catalog-only"})
    assert tracing_mod.uc_trace_tables(experiment) == tracing_mod.MLflowTraceTables()


def test_get_experiment_by_id_maps_not_found_to_none():
    # mlflow's id lookup raises RESOURCE_DOES_NOT_EXIST for a missing experiment; normalize to None so
    # callers can treat it like the name lookup (which returns None).
    mlflow = mock.Mock()
    mlflow.get_experiment.side_effect = _not_found_exc()
    assert tracing_mod._get_experiment_by_id(mlflow, "nope") is None


def test_get_experiment_by_id_reraises_other_errors():
    # A non-"not found" error (auth, network) must propagate, not look like a missing experiment.
    from mlflow.exceptions import MlflowException

    mlflow = mock.Mock()
    mlflow.get_experiment.side_effect = MlflowException("permission denied")
    with pytest.raises(MlflowException):
        tracing_mod._get_experiment_by_id(mlflow, "eid-1")


# --- configure / disable ----------------------------------------------------


def test_bind_sets_experiment_name(tmp_path: pathlib.Path):
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
            tracing_mod.tracing_bind,
            ["--experiment-name", "/Shared/mason_traces/mine", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"experiment_name": "/Shared/mason_traces/mine"}
    assert AgentProject.load(tmp_path).trace_experiment_name == "/Shared/mason_traces/mine"


def test_bind_rejects_non_absolute_name(tmp_path: pathlib.Path):
    # An experiment name must be an absolute workspace path; a bare name is rejected up front.
    _project(tmp_path)
    result = CliRunner().invoke(
        tracing_mod.tracing_bind,
        ["--experiment-name", "not-a-path", "--source", str(tmp_path)],
        obj=_Ctx(),
    )
    assert result.exit_code != 0
    assert "absolute workspace path" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_name is None  # nothing persisted


def test_bind_accepts_uc_backed_experiment(tmp_path: pathlib.Path):
    # UC-backed experiments are supported for trace export (deploy grants their UC OTEL tables), so
    # binding a name persists it as-is; bind no longer consults the workspace for a name at all (any
    # UC resolution happens at deploy).
    _project(tmp_path)
    mlflow = mock.Mock()
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_bind,
            ["--experiment-name", "/Shared/uc", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"experiment_name": "/Shared/uc"}
    assert AgentProject.load(tmp_path).trace_experiment_name == "/Shared/uc"
    mlflow.get_experiment_by_name.assert_not_called()  # no workspace lookup for a name bind


def test_bind_requires_an_experiment(tmp_path: pathlib.Path):
    # bind needs an experiment (name or id), like `mason memory/sessions bind`; no args -> error,
    # nothing written. Pure agent.toml check, no mlflow call.
    _project(tmp_path)
    result = CliRunner().invoke(tracing_mod.tracing_bind, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert "--experiment-name or --experiment-id" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_name is None


def test_bind_by_experiment_id_stores_resolved_name(tmp_path: pathlib.Path):
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
            tracing_mod.tracing_bind,
            ["--experiment-id", "123", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"experiment_name": "/Shared/mason_traces/from-id"}
    # stored as the resolved NAME, never the id
    assert AgentProject.load(tmp_path).trace_experiment_name == "/Shared/mason_traces/from-id"


def test_bind_rejects_unknown_experiment_id(tmp_path: pathlib.Path):
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment.side_effect = _not_found_exc()  # mlflow raises for an unknown id
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_bind,
            ["--experiment-id", "nope", "--source", str(tmp_path)],
            obj=_Ctx(),
        )
    assert result.exit_code != 0
    assert "No MLflow experiment" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_name is None


def test_bind_rejects_both_name_and_id(tmp_path: pathlib.Path):
    _project(tmp_path)
    result = CliRunner().invoke(
        tracing_mod.tracing_bind,
        ["--experiment-name", "/Shared/x", "--experiment-id", "1", "--source", str(tmp_path)],
        obj=_Ctx(),
    )
    assert result.exit_code != 0
    assert "not both" in result.output
    assert AgentProject.load(tmp_path).trace_experiment_name is None


def test_unbind_removes_the_experiment_binding(tmp_path: pathlib.Path):
    _project(tmp_path, experiment_name="/Shared/mason_traces/x")
    result = CliRunner().invoke(
        tracing_mod.tracing_unbind, ["--source", str(tmp_path)], obj=_Ctx(output="json")
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"experiment_name": None}
    assert AgentProject.load(tmp_path).trace_experiment_name is None  # binding removed -> off


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
    mlflow.get_experiment.return_value = mock.Mock(tags={})  # managed experiment
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
    mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="p1", tags={})
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


# --- UC-backed reads (SQL warehouse) ------------------------------------------

_WAREHOUSE_ENV = "MLFLOW_TRACING_SQL_WAREHOUSE_ID"


def _bound_uc_project(tmp_path: pathlib.Path):
    """A project bound to a name that resolves to a UC-backed experiment."""
    _project(tmp_path, experiment_name="/Shared/uc")
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = types.SimpleNamespace(
        experiment_id="eid-uc", tags={_DEST_TAG: "cat.schema.pfx"}
    )
    return mlflow


def test_list_reads_uc_backed_experiment_through_warehouse(tmp_path, monkeypatch):
    # A UC-backed bound experiment is read from the workspace (no longer skipped to the local store)
    # through the SQL warehouse: --warehouse exports MLFLOW_TRACING_SQL_WAREHOUSE_ID before the search.
    mlflow = _bound_uc_project(tmp_path)
    monkeypatch.delenv(_WAREHOUSE_ENV, raising=False)
    seen = {}

    def _search(**kwargs):
        seen["warehouse"] = os.environ.get(_WAREHOUSE_ENV)  # captured at search time
        seen["locations"] = kwargs["locations"]
        return [_trace("tr-uc")]

    mlflow.search_traces.side_effect = _search
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--warehouse", "wh-1", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert seen == {"warehouse": "wh-1", "locations": ["eid-uc"]}
    assert json.loads(result.output)[0]["trace_id"] == "tr-uc"
    mlflow.get_experiment_by_name.assert_called_once_with("/Shared/uc")


def test_list_reads_uc_backed_experiment_with_warehouse_from_env(tmp_path, monkeypatch):
    # No --warehouse flag: the MLFLOW_TRACING_SQL_WAREHOUSE_ID env var is used instead.
    mlflow = _bound_uc_project(tmp_path)
    monkeypatch.setenv(_WAREHOUSE_ENV, "wh-env")
    mlflow.search_traces.return_value = []
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    assert mlflow.search_traces.call_args.kwargs["locations"] == ["eid-uc"]
    assert os.environ[_WAREHOUSE_ENV] == "wh-env"


def test_list_uc_backed_experiment_requires_a_warehouse(tmp_path, monkeypatch):
    # A UC-backed target with neither --warehouse nor the env var is a clean error, not a silent
    # fallthrough to the local dev store.
    mlflow = _bound_uc_project(tmp_path)
    monkeypatch.delenv(_WAREHOUSE_ENV, raising=False)
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx()
        )
    assert result.exit_code != 0
    assert "requires a SQL warehouse" in result.output
    mlflow.search_traces.assert_not_called()


def test_list_explicit_uc_experiment_requires_a_warehouse(tmp_path, monkeypatch):
    # An explicit --experiment-name that resolves UC-backed also requires a warehouse.
    _project(tmp_path)
    monkeypatch.delenv(_WAREHOUSE_ENV, raising=False)
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = types.SimpleNamespace(
        experiment_id="eid-uc", tags={_DEST_TAG: "cat.schema"}
    )
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--experiment-name", "/Shared/uc", "--source", str(tmp_path)],
            obj=_Ctx(),
        )
    assert result.exit_code != 0
    assert "requires a SQL warehouse" in result.output
    mlflow.search_traces.assert_not_called()


def test_get_reads_uc_backed_experiment_through_warehouse(tmp_path, monkeypatch):
    # get resolves through the same helper: a UC-backed bound experiment + --warehouse exports the
    # env var before get_trace.
    mlflow = _bound_uc_project(tmp_path)
    monkeypatch.delenv(_WAREHOUSE_ENV, raising=False)
    seen = {}

    def _get_trace(trace_id):
        seen["warehouse"] = os.environ.get(_WAREHOUSE_ENV)  # captured at fetch time
        return _trace(trace_id)

    mlflow.get_trace.side_effect = _get_trace
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_get,
            ["tr-uc", "--warehouse", "wh-2", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert seen["warehouse"] == "wh-2"
    assert json.loads(result.output)["trace_id"] == "tr-uc"


def test_uc_read_restores_a_preexisting_warehouse_env_var(tmp_path, monkeypatch):
    # The UC read exports MLFLOW_TRACING_SQL_WAREHOUSE_ID for the duration of the search only: a
    # pre-existing value is restored afterwards, not left clobbered by --warehouse.
    mlflow = _bound_uc_project(tmp_path)
    monkeypatch.setenv(_WAREHOUSE_ENV, "wh-sentinel")
    seen = {}

    def _search(**kwargs):
        seen["warehouse"] = os.environ.get(_WAREHOUSE_ENV)  # during the read: the --warehouse value
        return [_trace("tr-uc")]

    mlflow.search_traces.side_effect = _search
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--warehouse", "wh-1", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert seen["warehouse"] == "wh-1"  # exported for the read
    assert os.environ[_WAREHOUSE_ENV] == "wh-sentinel"  # restored after it


def test_uc_read_removes_the_warehouse_env_var_when_it_was_unset(tmp_path, monkeypatch):
    # With no pre-existing value, the --warehouse export is removed on exit, not left behind.
    mlflow = _bound_uc_project(tmp_path)
    monkeypatch.delenv(_WAREHOUSE_ENV, raising=False)
    mlflow.search_traces.return_value = []
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--warehouse", "wh-1", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert _WAREHOUSE_ENV not in os.environ


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
    mlflow.get_experiment_by_name.return_value = (
        None  # no project experiment -> reads the workspace
    )
    mlflow.get_trace.return_value = None
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_get, ["tr-x", "--source", str(tmp_path)], obj=_Ctx()
        )
    assert result.exit_code != 0
    assert "No trace found" in result.output


def test_get_by_explicit_experiment_id(tmp_path: pathlib.Path):
    # --experiment-id points get at that workspace store (no project resolution, no local fallback).
    _project(tmp_path)
    mlflow = mock.Mock()
    mlflow.get_experiment.return_value = mock.Mock(tags={})  # managed experiment
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
    mlflow.get_experiment.side_effect = _not_found_exc()  # mlflow raises for an unknown id
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
    # No workspace experiment yet, but a local `mason dev` store exists -> list reads the local traces
    # over a short-lived REST server (not by opening the sqlite file), consistent with where `mason dev`
    # traced pre-deploy, and tears the server down after.
    _project(tmp_path, experiment_name="/Shared/mason_traces/demo")
    (tmp_path / ".mason").mkdir()
    (tmp_path / ".mason" / "mlflow.db").write_text("")  # only needs to exist
    mlflow = mock.Mock()
    # workspace miss, then local hit (bare project-name experiment served by the local read server)
    mlflow.get_experiment_by_name.side_effect = [None, mock.Mock(experiment_id="local-1", tags={})]
    mlflow.search_traces.return_value = [_trace("tr-local")]
    fake_server = mock.Mock()
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
        mock.patch.object(
            tracing_mod, "_start_read_server", return_value=(fake_server, "http://127.0.0.1:5599")
        ),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["trace_id"] == "tr-local"
    # read over the local REST server, not the workspace
    assert any(
        str(c.args[0]).startswith("http://127.0.0.1")
        for c in mlflow.set_tracking_uri.call_args_list
    )
    fake_server.terminate.assert_called_once()  # torn down after the read


def test_get_reads_local_dev_store_when_not_provisioned(tmp_path: pathlib.Path):
    # get resolves its store the same way as list: the local dev store (over a short-lived REST server)
    # when the workspace experiment isn't provisioned yet.
    _project(tmp_path, experiment_name="/Shared/mason_traces/demo")
    (tmp_path / ".mason").mkdir()
    (tmp_path / ".mason" / "mlflow.db").write_text("")
    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.side_effect = [None, mock.Mock(experiment_id="local-1", tags={})]
    mlflow.get_trace.return_value = _trace("tr-local")
    fake_server = mock.Mock()
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_set_tracking_uri"),
        mock.patch.object(
            tracing_mod, "_start_read_server", return_value=(fake_server, "http://127.0.0.1:5599")
        ),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_get,
            ["tr-local", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["trace_id"] == "tr-local"
    assert any(
        str(c.args[0]).startswith("http://127.0.0.1")
        for c in mlflow.set_tracking_uri.call_args_list
    )
    fake_server.terminate.assert_called_once()


def test_list_degrades_when_local_read_server_unavailable(tmp_path: pathlib.Path):
    # The local store exists but its short-lived read server can't start (e.g. uv missing) -> list
    # degrades to showing nothing rather than erroring.
    _project(tmp_path)  # unbound -> straight to the local store
    (tmp_path / ".mason").mkdir()
    (tmp_path / ".mason" / "mlflow.db").write_text("")
    mlflow = mock.Mock()
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_start_read_server", return_value=(None, None)),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []
    mlflow.search_traces.assert_not_called()


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
    # pinned interpreter (avoids source-building pyarrow on 3.13) + broad mlflow range (uvx cache reuse
    # across mason releases; the server owns the schema and reads go over REST, so no exact-version pin)
    assert captured["cmd"][captured["cmd"].index("--python") + 1] == "3.12"
    assert "mlflow>=3.10,<4" in captured["cmd"]


def test_start_local_tracing_server_degrades_when_launch_fails(tmp_path: pathlib.Path, monkeypatch):
    # If the server process can't be spawned (e.g. uv missing), degrade to (None, {}) so `mason dev`
    # runs without traces rather than aborting.
    monkeypatch.setattr(tracing_mod, "_free_port", lambda: 5599)

    def _boom(cmd, **kwargs):
        raise OSError("uvx not found")

    monkeypatch.setattr(tracing_mod.subprocess, "Popen", _boom)
    server, env = tracing_mod.start_local_tracing_server(tmp_path)
    assert server is None and env == {}


def test_wait_for_server_false_when_process_exits():
    # If the server process dies before serving (e.g. an install/bind failure), detect the exit and
    # bail immediately rather than blocking for the whole timeout.
    server = mock.Mock()
    server.poll.return_value = 1  # already exited
    assert tracing_mod._wait_for_server("http://127.0.0.1:1", server, timeout=1) is False


def test_wait_for_server_true_when_health_responds(monkeypatch):
    # Once the health endpoint answers 200, the server is ready to query.
    server = mock.Mock()
    server.poll.return_value = None  # still running
    resp_cm = mock.MagicMock()
    resp_cm.__enter__.return_value = mock.Mock(status=200)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: resp_cm)
    assert tracing_mod._wait_for_server("http://127.0.0.1:5599", server) is True


def test_start_read_server_degrades_when_launch_fails(tmp_path: pathlib.Path, monkeypatch):
    # A short-lived read server that can't spawn (uv missing) degrades to (None, None) so `list`/`get`
    # show nothing rather than aborting.
    monkeypatch.setattr(tracing_mod, "_free_port", lambda: 5599)

    def _boom(cmd, **kwargs):
        raise OSError("uvx not found")

    monkeypatch.setattr(tracing_mod.subprocess, "Popen", _boom)
    server, base_url = tracing_mod._start_read_server(tmp_path / "mlflow.db")
    assert server is None and base_url is None
