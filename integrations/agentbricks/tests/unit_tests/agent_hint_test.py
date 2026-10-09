"""Agent-only hints preserve command behavior and machine-readable stdout."""

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


def test_no_hint_for_human_or_generic_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert agent_hint.workflow_hint() is None
    result = CliRunner().invoke(agentbricks, ["--help"])
    assert result.exit_code == 0
    assert result.stderr == ""


@pytest.mark.parametrize("marker", agent_hint.AGENT_ENV_MARKERS)
def test_supported_agent_markers_expose_local_guidance(monkeypatch, marker):
    monkeypatch.setenv(marker, "1")
    message = agent_hint.workflow_hint()
    assert message is not None
    assert str(bundled_workflow_skill()) in message
    assert len(message.splitlines()) == 1
    assert "http://" not in message and "https://" not in message


def test_empty_marker_is_not_an_agent(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "")
    assert agent_hint.workflow_hint() is None


def test_cursor_value_specific_marker(monkeypatch):
    monkeypatch.setenv("CURSOR_EXTENSION_HOST_ROLE", "human-terminal")
    assert agent_hint.workflow_hint() is None
    monkeypatch.setenv("CURSOR_EXTENSION_HOST_ROLE", "agent-exec")
    assert agent_hint.workflow_hint() is not None


@pytest.mark.parametrize("tty", [True, False])
def test_kiro_detection_respects_human_terminal(monkeypatch, tty):
    monkeypatch.setenv("TERM_PROGRAM", "kiro")
    monkeypatch.setattr(agent_hint.sys.stdout, "isatty", lambda: tty)
    assert (agent_hint.workflow_hint() is None) == tty


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_opt_out_silences_hints(monkeypatch, value):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("AGENTBRICKS_DISABLE_AGENT_HINT", value)
    assert agent_hint.workflow_hint() is None


def test_missing_resource_does_not_affect_command(monkeypatch):
    monkeypatch.setenv("CODEX_THREAD_ID", "test")

    def missing():
        raise FileNotFoundError("skill not bundled")

    monkeypatch.setattr(agent_hint, "bundled_workflow_skill", missing)
    result = CliRunner().invoke(agentbricks, ["--help"])
    assert result.exit_code == 0
    assert result.stderr == ""


def test_hint_output_failure_does_not_affect_command(monkeypatch):
    monkeypatch.setenv("CODEX_THREAD_ID", "test")
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
    baseline = runner.invoke(agentbricks, arguments)
    monkeypatch.setenv("CODEX_THREAD_ID", "test")
    result = runner.invoke(agentbricks, arguments)
    assert result.exit_code == baseline.exit_code == 0
    assert result.stdout == baseline.stdout
    assert result.stderr.count(str(bundled_workflow_skill())) == 1
    assert len(result.stderr.splitlines()) == 1


def test_hint_preserves_json_stdout(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_CLI", "1")
    result = CliRunner().invoke(agentbricks, ["-o", "json", "init", str(tmp_path / "agent")])
    assert result.exit_code == 0, result.output
    assert Path(json.loads(result.stdout)["workflow_skill"]).is_file()
    assert str(bundled_workflow_skill()) in result.stderr


def test_existing_project_reads_bundled_guidance_without_installation(monkeypatch, tmp_path):
    project_file = tmp_path / "README.md"
    project_file.write_text("existing project")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_THREAD_ID", "test")
    result = CliRunner().invoke(agentbricks, ["deploy", "--help"])
    assert result.exit_code == 0, result.output
    manifest = bundled_workflow_skill()
    assert str(manifest) in result.stderr
    assert manifest.is_file()
    assert list(tmp_path.iterdir()) == [project_file]
    assert project_file.read_text() == "existing project"


def test_hint_preserves_error_exit_status(monkeypatch):
    runner = CliRunner()
    baseline = runner.invoke(agentbricks, ["unknown-command"])
    monkeypatch.setenv("CLAUDECODE", "1")
    result = runner.invoke(agentbricks, ["unknown-command"])
    assert result.exit_code == baseline.exit_code != 0
    assert result.stdout == baseline.stdout


@pytest.mark.parametrize(
    "command", ["dev", "deploy", "memory", "sessions", "tracing", "tools", "endpoint", "doctor"]
)
def test_hint_is_available_across_cli_workflows(monkeypatch, command):
    monkeypatch.setenv("CODEX_THREAD_ID", "test")
    result = CliRunner().invoke(agentbricks, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert result.stderr.count(str(bundled_workflow_skill())) == 1


def test_completion_context_is_silent(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_THREAD_ID", "test")
    context = agentbricks.make_context("agentbricks", [], resilient_parsing=True)
    with context:
        assert context.resilient_parsing
    assert capsys.readouterr().err == ""


def test_package_import_does_not_emit_cli_hint(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    result = subprocess.run(
        [sys.executable, "-c", "import databricks_agentbricks; import databricks_agentkit"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Agent Bricks workflow guidance" not in result.stdout + result.stderr
