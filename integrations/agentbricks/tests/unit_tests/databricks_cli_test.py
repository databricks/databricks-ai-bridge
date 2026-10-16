"""Unit tests for the databricks CLI wrapper (`_databricks`)."""

from __future__ import annotations

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
