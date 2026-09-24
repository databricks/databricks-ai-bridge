"""Tests for the runtime store resolvers (databricks_mason.runtime.tool_manifest).

The runtime resolves stores from the explicit arg → env only; it never reads ``agent.toml`` (that
file is the CLI's authoring source, which `ab deploy`/`dev` resolve into env). These tests pin
that contract, including that a store declared in ``agent.toml`` is ignored at runtime.
"""

from __future__ import annotations

import pathlib

from databricks_mason.runtime.tool_manifest import (
    resolve_memory_store,
    resolve_session_store,
)


def _write(root: pathlib.Path, body: str) -> None:
    (root / "agent.toml").write_text(body, encoding="utf-8")


def test_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_STORE", "env-mem")
    monkeypatch.setenv("AGENT_SESSION_STORE", "env-sess")

    assert resolve_memory_store("arg-mem") == "arg-mem"
    assert resolve_session_store("arg-sess") == "arg-sess"


def test_env_resolves_when_no_arg(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_STORE", "mem-id-123")
    monkeypatch.setenv("AGENT_SESSION_STORE", "sess")

    assert resolve_memory_store() == "mem-id-123"
    assert resolve_session_store() == "sess"


def test_none_when_unset(monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)

    assert resolve_memory_store() is None
    assert resolve_session_store() is None


def test_agent_toml_is_ignored_at_runtime(tmp_path: pathlib.Path, monkeypatch):
    # A store declared in agent.toml must NOT be read by the runtime — env is the only carrier.
    _write(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\n'
        '\n[memory_store]\nname = "mem"\nid = "mem-id-123"\n\n[session_store]\nname = "sess"\n',
    )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)

    assert resolve_memory_store() is None
    assert resolve_session_store() is None
