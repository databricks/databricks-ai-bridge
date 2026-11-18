"""Resolve framework-neutral local commands while preserving legacy Apps projects."""

import json

import pytest

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.local_run import load_local_run


def test_dev_command_runs_without_an_apps_manifest(tmp_path):
    (tmp_path / "agent.toml").write_text(
        '[dev]\ncommand = ["python", "-m", "my_agent"]\n[dev.env]\nMODE = "local"\n'
    )
    local = load_local_run(tmp_path)
    assert local.command == ("python", "-m", "my_agent")
    assert local.environment == {"MODE": "local"}


def test_explicit_dev_configuration_ignores_deployment_manifest(tmp_path):
    (tmp_path / "app.yaml").write_text("invalid: [yaml")
    (tmp_path / "agent.toml").write_text('[dev]\ncommand = ["my-server"]\n')
    assert load_local_run(tmp_path).command == ("my-server",)


@pytest.mark.parametrize("command", ['["python", "app.py"]', '"python app.py"'])
def test_legacy_manifest_command_and_environment(tmp_path, command):
    (tmp_path / "agent.toml").write_text('schema_version = 1\n[agent]\nframework = "openai"\n')
    (tmp_path / "app.yaml").write_text(
        f"command: {command}\nenv:\n"
        "  - {name: LITERAL, value: text}\n"
        "  - {name: NUMBER, value: 42}\n"
        "  - {name: SECRET, valueFrom: resource}\n"
    )
    local = load_local_run(tmp_path)
    assert local.command == ("python", "app.py")
    assert local.environment == {"LITERAL": "text", "NUMBER": "42", "SECRET": None}


@pytest.mark.parametrize(
    "command", [[], [""], ["python", 2], "python app.py", ["replace_with_your_command"]]
)
def test_invalid_dev_command_is_rejected_without_falling_back(tmp_path, command):
    (tmp_path / "agent.toml").write_text(f"[dev]\ncommand = {json.dumps(command)}\n")
    (tmp_path / "app.yaml").write_text("command: [fallback]\n")
    with pytest.raises(AgentCliError, match="command"):
        load_local_run(tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        'dev = "invalid"',
        '[dev]\ncommand = ["x"]\nenv = "invalid"',
        '[dev]\ncommand = ["x"]\n[dev.env]\nPORT = 123',
        "invalid = [",
    ],
)
def test_invalid_dev_configuration_is_actionable(tmp_path, body):
    (tmp_path / "agent.toml").write_text(body)
    with pytest.raises(AgentCliError):
        load_local_run(tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "command: []",
        "command: 123",
        "command: [replace_with_your_command]",
        "command: [python]\nenv: [invalid]",
        "command: [python]\nenv: [{name: SECRET}]",
    ],
)
def test_invalid_legacy_configuration_is_rejected(tmp_path, body):
    (tmp_path / "app.yaml").write_text(body)
    with pytest.raises(AgentCliError):
        load_local_run(tmp_path)


def test_missing_configuration_has_init_hint(tmp_path):
    with pytest.raises(AgentCliError, match="No local start command") as error:
        load_local_run(tmp_path)
    assert error.value.hint is not None
    assert "agentbricks init" in error.value.hint
