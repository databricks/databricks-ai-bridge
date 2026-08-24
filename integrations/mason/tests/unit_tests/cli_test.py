"""Unit tests for the mason command tree: flattened sessions + login/logout wired in."""

from __future__ import annotations

from databricks_mason import cli


def test_sessions_verbs_are_flat_no_redundant_subgroup():
    names = set(cli.sessions.commands)
    # Session verbs are direct subcommands of `sessions` (no `mason sessions sessions`).
    assert {"create", "list", "get", "update", "delete", "fork"} <= names
    assert "sessions" not in names
    # Sub-resources remain their own groups.
    assert {"stores", "items"} <= names


def test_root_registers_login_and_logout():
    names = set(cli.mason.commands)
    assert {"login", "logout", "memory", "sessions", "tracing", "deploy", "deployments"} <= names
