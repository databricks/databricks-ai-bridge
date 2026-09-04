"""Unit tests for `mason tracing`: UC-schema validation, setup persistence, list resolution, and
the deploy-time UC-experiment provisioning.

MLflow-backed paths (`list`, `ensure_uc_experiment`) are exercised with a mocked `_mlflow`/UC symbols
(the hermetic test env carries no mlflow); the pure surface is tested directly.
"""

from __future__ import annotations

import json
import pathlib
import types
from unittest import mock

import pytest
from click.testing import CliRunner

from databricks_mason import tracing as tracing_mod
from databricks_mason.agent_project import AgentProject
from databricks_mason.errors import AgentCliError

_AGENT_TOML = 'schema_version = 1\n\n[agent]\nframework = "langgraph"\n'


class _Ctx:
    """Stand-in for CliContext: tracing reads .profile / .output, and .client() for the dev default."""

    def __init__(self, output: str = "text", profile=None, user="me@example.com"):
        self.output = output
        self.profile = profile
        self._user = user

    def client(self):
        return mock.Mock(current_user=self._user)


def _project(tmp_path: pathlib.Path, *, schema: str | None = None, warehouse: str | None = None):
    binding = ""
    if schema:
        binding = f'\n[trace_location]\nname = "{schema}"\n'
        if warehouse:
            binding += f'warehouse_id = "{warehouse}"\n'
    (tmp_path / "agent.toml").write_text(_AGENT_TOML + binding)
    return tmp_path


# --- validation --------------------------------------------------------------


def test_validate_uc_schema_accepts_catalog_dot_schema():
    assert tracing_mod.validate_uc_schema("catalog.schema") == "catalog.schema"
    assert tracing_mod.validate_uc_schema("  cat.sch  ") == "cat.sch"


def test_validate_uc_schema_rejects_bad_values():
    for bad in ("nodot", "cat.schema.table", "12345", "", "   ", "cat/schema"):
        with pytest.raises(AgentCliError):
            tracing_mod.validate_uc_schema(bad)


# --- experiment naming -------------------------------------------------------


def test_experiment_name():
    assert tracing_mod.experiment_name("me@x.com", "app") == "/Users/me@x.com/mason-traces/app"


def test_experiment_ui_url():
    assert (
        tracing_mod.experiment_ui_url("https://x.databricks.com", "123")
        == "https://x.databricks.com/ml/experiments/123/traces"
    )
    # trailing slash on the host is normalized; a missing/unknown host yields no link
    assert (
        tracing_mod.experiment_ui_url("https://x.databricks.com/", "123")
        == "https://x.databricks.com/ml/experiments/123/traces"
    )
    assert tracing_mod.experiment_ui_url(None, "123") is None
    assert tracing_mod.experiment_ui_url("unknown", "123") is None


# --- setup -------------------------------------------------------------------


def test_setup_persists_schema_and_warehouse_in_agent_toml(tmp_path: pathlib.Path):
    _project(tmp_path)
    result = CliRunner().invoke(
        tracing_mod.tracing_setup,
        ["--trace-location", "cat.schema", "--warehouse-id", "wh1", "--source", str(tmp_path)],
        obj=_Ctx(output="json"),
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"trace_location": "cat.schema", "warehouse_id": "wh1"}
    project = AgentProject.load(tmp_path)
    assert project.trace_location == "cat.schema"
    assert project.trace_warehouse == "wh1"


def test_setup_rejects_invalid_schema(tmp_path: pathlib.Path):
    _project(tmp_path)
    result = CliRunner().invoke(
        tracing_mod.tracing_setup,
        ["--trace-location", "cat.schema.table", "--source", str(tmp_path)],
        obj=_Ctx(),
    )
    assert result.exit_code != 0
    assert AgentProject.load(tmp_path).trace_location is None  # nothing persisted


# --- ensure_uc_experiment ----------------------------------------------------


def _fake_uc(mlflow, existing=None, create_id="e1"):
    mlflow.get_experiment_by_name.return_value = (
        types.SimpleNamespace(experiment_id=existing) if existing else None
    )
    mlflow.create_experiment.return_value = create_id


def test_ensure_uc_experiment_creates_and_links():
    mlflow = mock.Mock()
    _fake_uc(mlflow)
    set_location = mock.Mock()
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_configure"),
        mock.patch.object(
            tracing_mod, "_uc_trace_symbols", return_value=(mock.Mock(), set_location)
        ),
    ):
        experiment_id = tracing_mod.ensure_uc_experiment(None, "/Users/me/x", "cat.schema", "wh1")
    assert experiment_id == "e1"  # returns the id (for the experiment UI link), not the name
    mlflow.create_experiment.assert_called_once_with("/Users/me/x")
    assert set_location.call_args.kwargs["experiment_id"] == "e1"


def test_ensure_uc_experiment_idempotent_when_already_linked():
    mlflow = mock.Mock()
    _fake_uc(mlflow, existing="e9")
    set_location = mock.Mock(side_effect=RuntimeError("experiment is already linked to a location"))
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_configure"),
        mock.patch.object(
            tracing_mod, "_uc_trace_symbols", return_value=(mock.Mock(), set_location)
        ),
    ):
        # a re-deploy of an already-linked experiment is a no-op, not an error (returns the id)
        assert tracing_mod.ensure_uc_experiment(None, "/Users/me/x", "cat.schema", None) == "e9"


def test_ensure_uc_experiment_errors_on_existing_traces():
    mlflow = mock.Mock()
    _fake_uc(mlflow, existing="e9")
    set_location = mock.Mock(side_effect=RuntimeError("Experiment already contains traces."))
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_configure"),
        mock.patch.object(
            tracing_mod, "_uc_trace_symbols", return_value=(mock.Mock(), set_location)
        ),
    ):
        with pytest.raises(AgentCliError) as exc:
            tracing_mod.ensure_uc_experiment(None, "/Users/me/x", "cat.schema", None)
    assert "migrate" in (exc.value.hint or "").lower()


# --- list resolution ---------------------------------------------------------


def _fake_mlflow(traces):
    fake = mock.Mock()
    fake.search_traces.return_value = traces
    return fake


def _trace(trace_id):
    return types.SimpleNamespace(
        info=types.SimpleNamespace(
            trace_id=trace_id, status="OK", execution_time_ms=5, timestamp_ms=1
        )
    )


def test_list_uses_configured_uc_schema(tmp_path: pathlib.Path):
    _project(tmp_path, schema="proj.schema")
    fake = _fake_mlflow([_trace("tr-1")])
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=fake),
        mock.patch.object(tracing_mod, "_configure"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )
    assert result.exit_code == 0, result.output
    assert fake.search_traces.call_args.kwargs["locations"] == ["proj.schema"]
    assert json.loads(result.output)[0]["trace_id"] == "tr-1"


def test_list_flag_overrides_project(tmp_path: pathlib.Path):
    _project(tmp_path, schema="proj.schema")
    fake = _fake_mlflow([])
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=fake),
        mock.patch.object(tracing_mod, "_configure"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list,
            ["--trace-location", "flag.schema", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )
    assert result.exit_code == 0, result.output
    assert fake.search_traces.call_args.kwargs["locations"] == ["flag.schema"]


def test_list_errors_without_configured_location(tmp_path: pathlib.Path):
    # Tracing is UC-only + opt-in: with nothing configured and no flag, list refuses (no silent
    # fallback), pointing the user at `mason tracing setup`.
    _project(tmp_path)  # no UC schema configured
    result = CliRunner().invoke(
        tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
    )
    assert result.exit_code != 0
    assert "mason tracing setup" in result.output


def test_list_applies_configured_warehouse(tmp_path: pathlib.Path):
    # A UC schema is queried through the project's configured SQL warehouse.
    _project(tmp_path, schema="cat.schema", warehouse="wh-1")
    captured: dict = {}
    fake = _fake_mlflow([])
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=fake),
        mock.patch.object(
            tracing_mod, "_configure", side_effect=lambda m, p, w: captured.update(warehouse=w)
        ),
    ):
        res = CliRunner().invoke(
            tracing_mod.tracing_list, ["--source", str(tmp_path)], obj=_Ctx(output="json")
        )
    assert res.exit_code == 0, res.output
    assert captured["warehouse"] == "wh-1"
    assert fake.search_traces.call_args.kwargs["locations"] == ["cat.schema"]


# --- helpers -----------------------------------------------------------------


def test_project_trace_location_reads_schema_and_warehouse(tmp_path: pathlib.Path):
    _project(tmp_path, schema="cat.schema", warehouse="wh1")
    assert tracing_mod.project_trace_location(str(tmp_path)) == ("cat.schema", "wh1")


def test_project_trace_location_none_without_project(tmp_path: pathlib.Path):
    assert tracing_mod.project_trace_location(str(tmp_path)) == (None, None)


def test_status_str_handles_enum_like_and_none():
    class _EnumLike:
        name = "OK"

    assert tracing_mod._status_str(_EnumLike()) == "OK"
    assert tracing_mod._status_str(None) is None
