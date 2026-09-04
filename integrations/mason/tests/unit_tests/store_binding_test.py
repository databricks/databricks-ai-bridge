"""Tests for the runtime store-binding reader (databricks_mason.runtime.tool_manifest)."""

from __future__ import annotations

import pathlib

from databricks_mason.runtime.tool_manifest import (
    resolve_memory_store,
    resolve_session_store,
    store_binding,
)


def _write(root: pathlib.Path, body: str) -> None:
    (root / "agent.toml").write_text(body, encoding="utf-8")


def test_store_binding_reads_declared_tables(tmp_path: pathlib.Path, monkeypatch):
    _write(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\n'
        '\n[memory_store]\nname = "mem"\n\n[session_store]\nname = "sess"\n',
    )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    assert store_binding("memory_store") == "mem"
    assert store_binding("session_store") == "sess"


def test_store_binding_none_when_table_absent(tmp_path: pathlib.Path, monkeypatch):
    _write(tmp_path, 'schema_version = 1\n\n[agent]\nframework = "openai"\n')
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    assert store_binding("memory_store") is None
    assert store_binding("session_store") is None


def test_store_binding_none_when_no_manifest(tmp_path: pathlib.Path, monkeypatch):
    # No agent.toml anywhere and no MASON_PROJECT_ROOT with one → never raises, just None.
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    assert store_binding("session_store") is None


def test_store_binding_prefers_id_when_requested(tmp_path: pathlib.Path, monkeypatch):
    _write(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\n'
        '\n[memory_store]\nname = "mem"\nid = "mem-id-123"\n',
    )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    assert store_binding("memory_store", prefer="id") == "mem-id-123"
    assert store_binding("memory_store") == "mem"  # default still returns the display name


def test_resolve_memory_store_returns_bound_id(tmp_path: pathlib.Path, monkeypatch):
    # The entries API is keyed by id, so the runtime resolves the memory store to its bound id.
    _write(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\n'
        '\n[memory_store]\nname = "mem"\nid = "mem-id-123"\n\n[session_store]\nname = "sess"\n',
    )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)

    assert resolve_memory_store() == "mem-id-123"
    assert resolve_session_store() == "sess"  # sessions use the name verbatim


def test_resolve_memory_store_falls_back_to_name_without_id(tmp_path: pathlib.Path, monkeypatch):
    # A hand-written binding with only a name still resolves (to the name) rather than None.
    _write(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\n\n[memory_store]\nname = "mem"\n',
    )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)

    assert resolve_memory_store() == "mem"
