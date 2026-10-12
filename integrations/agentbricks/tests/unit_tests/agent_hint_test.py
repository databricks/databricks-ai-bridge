"""Default skill hints preserve command behavior and machine-readable stdout."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from databricks_agentbricks import agent_hint
from databricks_agentbricks.cli.app import agentbricks
from databricks_agentbricks.skills import bundled_workflow_skill


@pytest.fixture(autouse=True)
def default_hint_environment(monkeypatch, suppress_cli_hints):
    monkeypatch.delenv("AGENTBRICKS_DISABLE_AGENT_HINT", raising=False)


@pytest.mark.parametrize(
    "environment",
    [{}, {"CI": "true"}, {"TERM_PROGRAM": "human-terminal"}, {"FUTURE_CODING_AGENT": "1"}],
)
def test_hint_is_enabled_without_agent_detection(monkeypatch, environment):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    message = agent_hint.workflow_hint()
    assert message is not None
    assert str(bundled_workflow_skill()) in message
    assert len(message.splitlines()) == 1
    assert "http://" not in message and "https://" not in message


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_opt_out_silences_hints(monkeypatch, value):
    monkeypatch.setenv("AGENTBRICKS_DISABLE_AGENT_HINT", value)
    assert agent_hint.workflow_hint() is None
    result = CliRunner().invoke(agentbricks, ["--help"])
    assert result.exit_code == 0
    assert result.stderr == ""


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_false_opt_out_values_keep_hints_enabled(monkeypatch, value):
    monkeypatch.setenv("AGENTBRICKS_DISABLE_AGENT_HINT", value)
    result = CliRunner().invoke(agentbricks, ["--help"])
    assert result.exit_code == 0
    assert result.stderr.count(str(bundled_workflow_skill())) == 1


def test_missing_resource_does_not_affect_command(monkeypatch):
    def missing():
        raise FileNotFoundError("skill not bundled")

    monkeypatch.setattr(agent_hint, "bundled_workflow_skill", missing)
    result = CliRunner().invoke(agentbricks, ["--help"])
    assert result.exit_code == 0
    assert result.stderr == ""


def test_hint_output_failure_does_not_affect_command(monkeypatch):
    echo = click.echo

    def broken_hint(message=None, *args, **kwargs):
        if kwargs.get("err") and "Agent Bricks workflow guidance" in str(message):
            raise OSError("stderr unavailable")
        return echo(message, *args, **kwargs)

    monkeypatch.setattr("click.echo", broken_hint)
    result = CliRunner().invoke(agentbricks, ["--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "arguments",
    [["--help"], ["--version"], ["init", "--help"], ["memory", "stores", "--help"]],
)
def test_help_and_nested_commands_emit_exactly_one_stderr_hint(monkeypatch, arguments):
    runner = CliRunner()
    with monkeypatch.context() as disabled:
        disabled.setenv("AGENTBRICKS_DISABLE_AGENT_HINT", "1")
        baseline = runner.invoke(agentbricks, arguments)
    result = runner.invoke(agentbricks, arguments)
    assert result.exit_code == baseline.exit_code == 0
    assert result.stdout == baseline.stdout
    assert result.stderr.count(str(bundled_workflow_skill())) == 1
    assert len(result.stderr.splitlines()) == 1


def test_hint_preserves_json_stdout(tmp_path):
    result = CliRunner().invoke(agentbricks, ["-o", "json", "init", str(tmp_path / "agent")])
    assert result.exit_code == 0, result.output
    assert Path(json.loads(result.stdout)["workflow_skill"]).is_file()
    assert str(bundled_workflow_skill()) in result.stderr


def test_existing_project_reads_bundled_guidance_without_installation(monkeypatch, tmp_path):
    project_file = tmp_path / "README.md"
    project_file.write_text("existing project")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(agentbricks, ["deploy", "--help"])
    assert result.exit_code == 0, result.output
    manifest = bundled_workflow_skill()
    assert str(manifest) in result.stderr
    assert manifest.is_file()
    assert list(tmp_path.iterdir()) == [project_file]
    assert project_file.read_text() == "existing project"


def test_hint_preserves_error_exit_status(monkeypatch):
    runner = CliRunner()
    with monkeypatch.context() as disabled:
        disabled.setenv("AGENTBRICKS_DISABLE_AGENT_HINT", "1")
        baseline = runner.invoke(agentbricks, ["unknown-command"])
    result = runner.invoke(agentbricks, ["unknown-command"])
    assert result.exit_code == baseline.exit_code != 0
    assert result.stdout == baseline.stdout
    assert result.stderr.count(str(bundled_workflow_skill())) == 1


@pytest.mark.parametrize(
    "command", ["dev", "deploy", "memory", "sessions", "tracing", "tools", "endpoint", "doctor"]
)
def test_hint_is_available_across_cli_workflows(command):
    result = CliRunner().invoke(agentbricks, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert result.stderr.count(str(bundled_workflow_skill())) == 1


def test_completion_context_is_silent(capsys):
    context = agentbricks.make_context("agentbricks", [], resilient_parsing=True)
    with context:
        assert context.resilient_parsing
    assert capsys.readouterr().err == ""


def test_package_import_does_not_emit_cli_hint():
    result = subprocess.run(
        [sys.executable, "-c", "import databricks_agentbricks; import databricks_agentkit"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Agent Bricks workflow guidance" not in result.stdout + result.stderr
