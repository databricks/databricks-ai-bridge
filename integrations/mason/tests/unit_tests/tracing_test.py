"""Unit tests for `mason tracing`: instrument snippet, destination parsing, mlflow guard.

The pure-Python surface (instrument, destination parsing) is covered directly. The
mlflow-backed commands are exercised only for their "mlflow not installed" guard, since the
hermetic test env does not carry mlflow on the deptree.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from databricks_mason import tracing as tracing_mod
from databricks_mason.errors import AgentCliError


class _Ctx:
    """Stand-in for CliContext: tracing commands read only .profile and .output."""

    def __init__(self, output: str = "text", profile=None):
        self.output = output
        self.profile = profile


def test_instrument_text_runs():
    result = CliRunner().invoke(
        tracing_mod.tracing_instrument,
        ["--destination", "cat.schema", "--experiment", "/Shared/x"],
        obj=_Ctx(),
    )
    assert result.exit_code == 0, result.output
    assert "Agent Tracing" in result.output


def test_instrument_json_snippet_contents():
    result = CliRunner().invoke(
        tracing_mod.tracing_instrument,
        ["--destination", "cat.schema", "--experiment", "/Shared/x"],
        obj=_Ctx(output="json"),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["destination"] == "cat.schema"
    assert payload["experiment"] == "/Shared/x"
    snippet = payload["snippet"]
    assert 'catalog_name="cat"' in snippet
    assert 'schema_name="schema"' in snippet
    assert "mlflow.openai.autolog" in snippet
    assert 'mlflow.set_experiment("/Shared/x")' in snippet


def test_instrument_defaults_to_placeholders_and_default_experiment():
    result = CliRunner().invoke(tracing_mod.tracing_instrument, [], obj=_Ctx(output="json"))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["experiment"] == tracing_mod._DEFAULT_EXPERIMENT
    assert "<catalog>" in payload["snippet"] and "<schema>" in payload["snippet"]


def test_split_destination_valid():
    assert tracing_mod._split_destination("my_cat.my_schema") == ("my_cat", "my_schema")


def test_split_destination_invalid_raises():
    for bad in ("nodot", ".schema", "catalog."):
        try:
            tracing_mod._split_destination(bad)
            raise AssertionError(f"expected AgentCliError for {bad!r}")
        except AgentCliError:
            pass


def test_ensure_experiment_creates_parent_dir_for_nested_path():
    from unittest import mock

    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = None  # doesn't exist yet
    mlflow.create_experiment.return_value = "eid-1"
    client = mock.Mock()

    eid = tracing_mod._ensure_experiment(mlflow, client, "/Users/me@x.com/mason-traces/demo")

    assert eid == "eid-1"
    # the intermediate workspace folder is created before the experiment (mlflow won't make it)
    client.ensure_workspace_dir.assert_called_once_with("/Users/me@x.com/mason-traces")


def test_ensure_experiment_reuses_existing_without_mkdir():
    from unittest import mock

    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="eid-2")
    client = mock.Mock()

    assert tracing_mod._ensure_experiment(mlflow, client, "/Shared/x") == "eid-2"
    client.ensure_workspace_dir.assert_not_called()  # existing experiment -> no dir work
    mlflow.create_experiment.assert_not_called()


def test_default_experiment_is_per_app_under_user_home():
    assert (
        tracing_mod.default_experiment("me@x.com", "my-agent")
        == "/Users/me@x.com/mason-traces/my-agent"
    )


def test_default_experiment_falls_back_to_shared_without_app():
    assert tracing_mod.default_experiment("me@x.com", None) == tracing_mod._DEFAULT_EXPERIMENT


def test_experiment_url_builds_traces_tab_link():
    url = tracing_mod._experiment_url("https://ws.databricks.com/", "123")
    assert url == "https://ws.databricks.com/ml/experiments/123?compareRunsMode=TRACES"


def test_link_trace_location_reports_already_linked_without_relink():
    def set_location(location, experiment_id):
        raise RuntimeError("experiment is already linked to a storage location")

    try:
        tracing_mod._link_trace_location(
            set_location, lambda **k: None, object(), "e1", relink=False
        )
        raise AssertionError("expected AgentCliError")
    except AgentCliError as exc:
        assert "--relink" in (exc.hint or "")


def test_link_trace_location_relinks_when_requested():
    calls = []

    def set_location(location, experiment_id):
        calls.append("set")
        if calls.count("set") == 1:  # first attempt fails as already-linked
            raise RuntimeError("already linked")

    def unset_location(location, experiment_id):
        calls.append("unset")

    tracing_mod._link_trace_location(set_location, unset_location, object(), "e1", relink=True)
    assert calls == ["set", "unset", "set"]  # try, unset existing, re-link


def test_link_trace_location_propagates_unrelated_errors():
    def set_location(location, experiment_id):
        raise RuntimeError("permission denied on catalog")

    try:
        tracing_mod._link_trace_location(
            set_location, lambda **k: None, object(), "e1", relink=True
        )
        raise AssertionError("expected the original error")
    except RuntimeError as exc:
        assert "permission denied" in str(exc)  # not swallowed as an already-linked case


def test_setup_requires_mlflow_when_absent():
    # mlflow is not on the hermetic test deptree, so setup should surface a clean CLI error
    # (non-zero exit) rather than a traceback.
    result = CliRunner().invoke(
        tracing_mod.tracing_setup, ["--catalog", "c", "--schema", "s"], obj=_Ctx()
    )
    assert result.exit_code != 0


def test_list_resolves_experiment_name_to_id_for_search():
    # search_traces selects by experiment_ids, not names, so list must resolve the name first.
    from unittest import mock

    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = mock.Mock(experiment_id="eid-9")
    mlflow.search_traces.return_value = []
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_configure"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--experiment", "/Shared/x", "--limit", "7"], obj=_Ctx()
        )

    assert result.exit_code == 0, result.output
    mlflow.get_experiment_by_name.assert_called_once_with("/Shared/x")
    _, kwargs = mlflow.search_traces.call_args
    assert kwargs["experiment_ids"] == ["eid-9"]
    assert kwargs["max_results"] == 7


def test_list_returns_empty_when_experiment_absent():
    # A not-yet-created experiment has no traces; list should show none, not error.
    from unittest import mock

    mlflow = mock.Mock()
    mlflow.get_experiment_by_name.return_value = None
    with (
        mock.patch.object(tracing_mod, "_mlflow", return_value=mlflow),
        mock.patch.object(tracing_mod, "_configure"),
    ):
        result = CliRunner().invoke(
            tracing_mod.tracing_list, ["--experiment", "/Shared/missing"], obj=_Ctx(output="json")
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []
    mlflow.search_traces.assert_not_called()


def test_status_str_handles_enum_like_and_none():
    class _EnumLike:
        name = "OK"

    assert tracing_mod._status_str(_EnumLike()) == "OK"
    assert tracing_mod._status_str(None) is None
