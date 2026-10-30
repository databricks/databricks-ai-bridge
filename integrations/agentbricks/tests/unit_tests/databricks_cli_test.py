"""Unit tests for the databricks CLI wrapper (`_databricks`)."""

from __future__ import annotations

import sys
import types

from databricks_agentbricks.clients import databricks_cli as dc


def test_databricks_failure_hides_the_apps_subcommand(monkeypatch):
    # AgentBricks drives `databricks apps` as an implementation detail; a failure must be reported in
    # AgentBricks's terms (the `action`) and must NOT echo the raw subcommand or --profile at the customer.
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="boom from the CLI")

    monkeypatch.setattr(dc.subprocess, "run", fake_run)
    try:
        dc._databricks(
            ["apps", "run-local", "--prepare-environment"],
            "e2-dogfood",
            capture=True,
            action="Could not start the agent locally.",
        )
        raise AssertionError("expected AgentCliError")
    except dc.AgentCliError as exc:
        assert exc.message == "Could not start the agent locally."
        assert "apps" not in exc.message and "e2-dogfood" not in exc.message  # no leak
        assert exc.hint == "boom from the CLI"  # underlying stderr preserved for debugging


def test_databricks_failure_without_action_is_generic(monkeypatch):
    # No action label: still no raw subcommand/profile echo — just a generic message.
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="err")

    monkeypatch.setattr(dc.subprocess, "run", fake_run)
    try:
        dc._databricks(["apps", "deploy", "x"], "prof", capture=True)
        raise AssertionError("expected AgentCliError")
    except dc.AgentCliError as exc:
        assert "apps" not in exc.message and "prof" not in exc.message


def test_to_stderr_keeps_stdout_for_the_agentbricks_payload(monkeypatch):
    # `--output json` prints Agent Bricks' payload on stdout; the CLI's own output must not land there.
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return types.SimpleNamespace(returncode=0, stdout=None, stderr=None)

    monkeypatch.setattr(dc.subprocess, "run", fake_run)
    dc._databricks(["sync", "src", "/Workspace/dst"], "prof", to_stderr=True)
    assert seen["stdout"] == 2
    dc._databricks(["apps", "logs", "x"], "prof")
    assert seen["stdout"] is None


def test_tee_relays_output_to_stderr_and_returns_its_tail(monkeypatch, capsys):
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            calls.append((cmd, kwargs))
            self.stdout = iter(["Uploading…\n", "Error: Another operation is in progress.\n"])
            self.returncode = 1

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(dc.subprocess, "Popen", FakePopen)

    result = dc._databricks(["apps", "deploy", "x"], "prof", tee=True, check=False)

    captured = capsys.readouterr()
    assert captured.out == ""  # stdout stays clean for -o json
    assert "Another operation is in progress" in captured.err
    assert result.returncode == 1
    assert "Another operation is in progress" in result.stdout
    cmd, kwargs = calls[0]
    assert cmd == ["databricks", "apps", "deploy", "x", "--profile", "prof"]
    assert kwargs["stderr"] is dc.subprocess.STDOUT


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_run_until_ready_calls_on_ready_once_the_probe_succeeds(monkeypatch):
    monkeypatch.setattr(dc, "_READY_POLL_S", 0.01)
    probes = iter([False, False, True])
    ready: list[bool] = []
    result = dc._run_until_ready(
        _python("import time; time.sleep(0.3)"),
        cwd=None,
        env=None,
        probe=lambda: next(probes),
        on_ready=lambda: ready.append(True),
    )
    assert ready == [True]
    assert result.returncode == 0


def test_run_until_ready_skips_on_ready_when_the_command_exits_first(monkeypatch):
    # A server that fails during startup never becomes ready: no success callback, exit code kept.
    monkeypatch.setattr(dc, "_READY_POLL_S", 0.01)
    ready: list[bool] = []
    result = dc._run_until_ready(
        _python("import sys; sys.exit(3)"),
        cwd=None,
        env=None,
        probe=lambda: False,
        on_ready=lambda: ready.append(True),
    )
    assert ready == []
    assert result.returncode == 3


def test_databricks_with_ready_probe_uses_the_readiness_runner(monkeypatch):
    seen: dict = {}

    def fake_until_ready(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(dc, "_run_until_ready", fake_until_ready)
    probe, on_ready = (lambda: True), (lambda: None)
    dc._databricks(["apps", "run-local"], "prof", ready_probe=probe, on_ready=on_ready)
    assert seen["cmd"] == ["databricks", "apps", "run-local", "--profile", "prof"]
    assert seen["probe"] is probe and seen["on_ready"] is on_ready
