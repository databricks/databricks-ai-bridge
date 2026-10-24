"""Unit tests for the runner behind `agentbricks dev`: the manifest command on one port."""

from __future__ import annotations

import os
import pathlib
import signal
import sys
import threading
from unittest import mock

import pytest

from databricks_agentbricks.clients.local_app_runner import LocalAppRunner, _run_process
from databricks_agentbricks.errors import AgentCliError


def _manifest(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    path = tmp_path / "app.agentbricksdev.yaml"
    path.write_text(body)
    return path


@pytest.fixture
def config_file(tmp_path: pathlib.Path, monkeypatch) -> pathlib.Path:
    path = tmp_path / "databrickscfg"
    path.write_text(
        "[DEFAULT]\nhost = https://default.example.com\nworkspace_id = 999\n\n"
        "[dogfood]\nhost = https://dogfood.example.com\n\n"
        "[vanity]\nhost = https://account.example.com\nworkspace_id = 123\n"
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(path))
    return path


def test_runs_manifest_command_with_manifest_env_and_profile_auth(
    tmp_path, config_file, monkeypatch
):
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
    # FR-6: the agent gets the selected profile's auth and the manifest env, as run-local gave it.
    manifest = _manifest(
        tmp_path,
        "command: [uv, run, start-server]\n"
        "env:\n  - name: KEEP\n    value: 1\n  - name: SECRET\n    valueFrom: my-secret\n",
    )
    (tmp_path / "pyproject.toml").write_text("")
    run = mock.Mock(return_value=0)
    warn = mock.Mock()
    LocalAppRunner("dogfood", run_process=run, warn=warn).run(
        tmp_path, manifest, prepare_environment=True, port=8002
    )
    assert [call.args[0] for call in run.call_args_list] == [
        ["uv", "sync"],
        ["uv", "run", "start-server"],
    ]
    assert run.call_args.kwargs["cwd"] == str(tmp_path)
    env = run.call_args.kwargs["env"]
    assert env["KEEP"] == "1"
    assert "SECRET" not in env
    warn.assert_called_once_with("Skipping env SECRET: valueFrom isn't available locally.")
    assert env["DATABRICKS_CONFIG_PROFILE"] == "dogfood"
    assert env["DATABRICKS_HOST"] == "https://dogfood.example.com"
    # [DEFAULT] is a profile of its own, not defaults merged into every profile.
    assert "DATABRICKS_WORKSPACE_ID" not in env
    assert env["DATABRICKS_APP_NAME"] == tmp_path.resolve().name


def test_account_host_profile_exposes_workspace_id(tmp_path, config_file, monkeypatch):
    monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
    manifest = _manifest(tmp_path, "command: [start-server]\n")
    run = mock.Mock(return_value=0)
    LocalAppRunner("vanity", run_process=run).run(
        tmp_path, manifest, prepare_environment=False, port=8000
    )
    env = run.call_args.kwargs["env"]
    assert env["DATABRICKS_HOST"] == "https://account.example.com"
    assert env["DATABRICKS_WORKSPACE_ID"] == "123"


def test_without_profile_leaves_auth_to_the_project(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    manifest = _manifest(tmp_path, "command: [start-server]\n")
    run = mock.Mock(return_value=0)
    LocalAppRunner(None, run_process=run).run(
        tmp_path, manifest, prepare_environment=False, port=8000
    )
    assert "DATABRICKS_CONFIG_PROFILE" not in run.call_args.kwargs["env"]


def test_two_runs_each_get_only_their_own_port(tmp_path):
    # FR-1 / FR-2: each run is told its own port and nothing else; there is no shared fixed port.
    envs = []
    for port in (8000, 8002):
        source = tmp_path / str(port)
        source.mkdir()
        manifest = _manifest(source, "command: [uv, run, start-server]\n")
        run = mock.Mock(return_value=0)
        LocalAppRunner(None, run_process=run).run(
            source, manifest, prepare_environment=False, port=port
        )
        assert run.call_args.args[0] == ["uv", "run", "start-server"]
        envs.append(run.call_args.kwargs["env"])
    assert [env["PORT"] for env in envs] == ["8000", "8002"]
    assert [env["DATABRICKS_APP_URL"] for env in envs] == [
        "http://localhost:8000",
        "http://localhost:8002",
    ]
    assert not any("8001" in value for env in envs for value in env.values())


def test_failed_prepare_stops_before_the_command(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    manifest = _manifest(tmp_path, "command: [start-server]\n")
    run = mock.Mock(return_value=1)
    with pytest.raises(AgentCliError, match="Could not build the agent's environment."):
        LocalAppRunner(None, run_process=run).run(
            tmp_path, manifest, prepare_environment=True, port=8000
        )
    run.assert_called_once()
    assert run.call_args.args[0] == ["uv", "sync"]


def test_run_process_returns_the_exit_code(tmp_path):
    code = _run_process(
        [sys.executable, "-c", "import os, sys; sys.exit(int(os.environ['CODE']))"],
        cwd=str(tmp_path),
        env={**os.environ, "CODE": "3"},
    )
    assert code == 3


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_run_process_forwards_sigterm_and_exits_cleanly(tmp_path):
    # A stop sent to `agentbricks dev` (an IDE stop button, `timeout`) stops the agent too, and
    # returns so dev can clean up its manifest and tracing server.
    previous = signal.getsignal(signal.SIGTERM)
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        code = _run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=str(tmp_path),
            env=os.environ,
        )
    finally:
        timer.cancel()
    assert code == 0
    assert signal.getsignal(signal.SIGTERM) is previous
