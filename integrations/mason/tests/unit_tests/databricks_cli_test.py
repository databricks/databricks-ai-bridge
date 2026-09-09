"""Unit tests for the databricks CLI wrapper (`_databricks`)."""

from __future__ import annotations

import types

from databricks_mason import databricks_cli as dc


def test_databricks_failure_hides_the_apps_subcommand(monkeypatch):
    # Mason drives `databricks apps` as an implementation detail; a failure must be reported in
    # Mason's terms (the `action`) and must NOT echo the raw subcommand or --profile at the customer.
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
