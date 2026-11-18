"""Project environment preparation and native command launch contracts."""

import os
import subprocess
from unittest.mock import Mock

import pytest

from databricks_agentbricks.clients import local_process_client as module
from databricks_agentbricks.clients.local_process_client import LocalProcessClient
from databricks_agentbricks.diagnostics import CommandDiagnostics
from databricks_agentbricks.errors import AgentCliError


def test_launch_uses_project_environment_profile_and_selected_port(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/cli/venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/other/venv")
    monkeypatch.setenv("UV_NO_SYNC", "1")
    monkeypatch.setenv("PORT", "1111")
    monkeypatch.setenv("FROM_SHELL", "shell")
    (tmp_path / ".env").write_text("FROM_DOTENV=dotenv\nFROM_SHELL=dotenv\nRESOURCE=local-secret\n")
    (tmp_path / ".venv").mkdir()
    runner = Mock(return_value=subprocess.CompletedProcess([], 0))
    client = LocalProcessClient("demo", runner=runner)
    ready = Mock()
    monkeypatch.setattr(module, "accepts_connections", lambda port: port == 9000)
    client.run(
        tmp_path,
        ["python", "app.py"],
        {"MODE": "dev", "RESOURCE": None},
        prepare_environment=False,
        port=9000,
        on_ready=ready,
    )
    args, kwargs = runner.call_args
    assert args == (["python", "app.py"],)
    assert kwargs["cwd"] == str(tmp_path)
    env = kwargs["env"]
    assert env["PORT"] == "9000"
    assert env["HOST"] == "127.0.0.1"
    assert env["DATABRICKS_APP_URL"] == "http://localhost:9000"
    assert env["DATABRICKS_CONFIG_PROFILE"] == "demo"
    assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv")
    assert env["UV_NO_SYNC"] == "1"
    assert env["PATH"].split(os.pathsep)[0] == str(
        tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    )
    assert env["FROM_DOTENV"] == "dotenv"
    assert env["FROM_SHELL"] == "shell"
    assert env["RESOURCE"] == "local-secret"
    assert kwargs["ready_probe"]()
    assert not client.diagnostics.app_listened
    kwargs["on_ready"]()
    assert client.diagnostics.app_listened
    ready.assert_called_once()


def test_uv_project_preparation_is_explicit_and_does_not_rebuild_on_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_NO_SYNC", "1")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="example"\nversion="0.0.0"\n')
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"].copy()))
        if command == ["uv", "sync"]:
            assert "VIRTUAL_ENV" not in kwargs["env"]
            assert "UV_NO_SYNC" not in kwargs["env"]
            (tmp_path / ".venv").mkdir()
        return subprocess.CompletedProcess(command, 0)

    client = LocalProcessClient(runner=run)
    client.run(tmp_path, ["uv", "run", "start-server"], {}, prepare_environment=True, port=8000)
    assert [c[0] for c in calls] == [["uv", "sync"], ["uv", "run", "start-server"]]
    assert calls[-1][1]["UV_NO_SYNC"] == "1"
    assert calls[-1][1]["VIRTUAL_ENV"] == str(tmp_path / ".venv")
    calls.clear()
    client.run(tmp_path, ["uv", "run", "start-server"], {}, prepare_environment=False, port=8000)
    assert len(calls) == 1


def test_requirements_project_gets_its_own_venv(tmp_path):
    (tmp_path / "requirements.txt").write_text("example-package\n")
    runner = Mock(return_value=subprocess.CompletedProcess([], 0))
    LocalProcessClient(runner=runner).run(
        tmp_path, ["python", "app.py"], {}, prepare_environment=True, port=8000
    )
    commands = [call.args[0] for call in runner.call_args_list]
    assert commands[0] == ["uv", "venv", str(tmp_path / ".venv")]
    assert commands[1][:3] == ["uv", "pip", "install"]
    assert commands[1][-2:] == ["-r", "requirements.txt"]
    assert commands[2] == ["python", "app.py"]


def test_missing_local_resource_value_fails_without_launching(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_RESOURCE", raising=False)
    runner = Mock()
    with pytest.raises(AgentCliError, match="LOCAL_RESOURCE.*needs a local value"):
        LocalProcessClient(runner=runner).run(
            tmp_path, ["server"], {"LOCAL_RESOURCE": None}, prepare_environment=False, port=8000
        )
    runner.assert_not_called()


@pytest.mark.parametrize("build_failure", [True, False])
def test_failed_build_or_command_preserves_exit_and_output(tmp_path, build_failure):
    if build_failure:
        (tmp_path / "pyproject.toml").touch()
    runner = Mock(
        return_value=subprocess.CompletedProcess([], 17, stdout="missing module", stderr=" \n")
    )
    facts = CommandDiagnostics()
    with pytest.raises(AgentCliError) as error:
        LocalProcessClient(runner=runner, diagnostics=facts).run(
            tmp_path, ["server"], {}, prepare_environment=build_failure, port=8000
        )
    assert error.value.subprocess_exit_code == 17
    assert error.value.failure_detail == "missing module"
    assert facts.subprocess_exit_code == 17
    runner.assert_called_once()


def test_missing_executable_has_actionable_error(tmp_path):
    runner = Mock(side_effect=FileNotFoundError("no executable"))
    with pytest.raises(AgentCliError, match="Could not start the local command"):
        LocalProcessClient(runner=runner).run(
            tmp_path, ["server"], {}, prepare_environment=False, port=8000
        )
