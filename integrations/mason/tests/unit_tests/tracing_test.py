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


def test_setup_requires_mlflow_when_absent():
    # mlflow is not on the hermetic test deptree, so setup should surface a clean CLI error
    # (non-zero exit) rather than a traceback.
    result = CliRunner().invoke(
        tracing_mod.tracing_setup, ["--catalog", "c", "--schema", "s"], obj=_Ctx()
    )
    assert result.exit_code != 0


def test_status_str_handles_enum_like_and_none():
    class _EnumLike:
        name = "OK"

    assert tracing_mod._status_str(_EnumLike()) == "OK"
    assert tracing_mod._status_str(None) is None
