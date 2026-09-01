"""Unit tests for CLI ergonomics fixes: optional --store, friendly source-type,
delete confirmation, flag aliases, sessions get guard, and fork positional arg."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from databricks_mason.errors import AgentCliError
from databricks_mason.memory import _normalize_source_type, entries, stores
from databricks_mason.sessions import sessions


class _Ctx:
    def __init__(self, client, output="text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


class _MemClient:
    host = "https://example.databricks.com"

    def __init__(self):
        self.calls = []

    def create_memory_store(self, display_name, description=None):
        self.calls.append(("create", display_name, description))
        return {"name": "memory-stores/new", "display_name": display_name}

    def delete_memory_store(self, name):
        self.calls.append(("delete", name))
        return {}

    def get_memory_entry(self, store, entry):
        self.calls.append(("get_entry", store, entry))
        return {"name": entry, "path": "/p.md", "actor_id": "a"}

    def delete_memory_entry(self, store, entry):
        self.calls.append(("delete_entry", store, entry))
        return {}


# --- friendly source type ----------------------------------------------------


def test_source_type_normalization():
    assert _normalize_source_type("agent") == "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT"
    assert _normalize_source_type("AGENT") == "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT"
    assert _normalize_source_type(None) is None
    # The full enum still passes through for backward compatibility.
    assert (
        _normalize_source_type("MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT")
        == "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT"
    )
    with pytest.raises(AgentCliError):
        _normalize_source_type("bogus")


# --- optional --store on entry commands (69227) ------------------------------


def test_entries_get_accepts_full_resource_name_without_store():
    client = _MemClient()
    result = CliRunner().invoke(
        entries, ["get", "memory-stores/S/entries/E"], obj=_Ctx(client)
    )
    assert result.exit_code == 0, result.output
    assert ("get_entry", None, "memory-stores/S/entries/E") in client.calls


def test_entries_get_bare_id_without_store_errors():
    client = _MemClient()
    result = CliRunner().invoke(entries, ["get", "bare-id"], obj=_Ctx(client))
    assert result.exit_code != 0
    assert "Provide --store" in result.output
    assert client.calls == []  # never hit the API


# --- delete confirmation (69234) ---------------------------------------------


def test_store_delete_aborts_without_confirmation():
    client = _MemClient()
    result = CliRunner().invoke(stores, ["delete", "abc"], obj=_Ctx(client), input="n\n")
    assert result.exit_code != 0  # aborted
    assert client.calls == []


def test_store_delete_proceeds_with_yes_flag():
    client = _MemClient()
    result = CliRunner().invoke(stores, ["delete", "abc", "--yes"], obj=_Ctx(client))
    assert result.exit_code == 0, result.output
    assert ("delete", "abc") in client.calls


def test_entry_delete_confirms_with_short_flag():
    client = _MemClient()
    result = CliRunner().invoke(
        entries, ["delete", "memory-stores/S/entries/E", "-y"], obj=_Ctx(client)
    )
    assert result.exit_code == 0, result.output
    assert ("delete_entry", None, "memory-stores/S/entries/E") in client.calls


# --- flag alias (69232) ------------------------------------------------------


def test_memory_store_create_accepts_name_alias():
    client = _MemClient()
    result = CliRunner().invoke(stores, ["create", "--name", "acme"], obj=_Ctx(client))
    assert result.exit_code == 0, result.output
    assert ("create", "acme", None) in client.calls


# --- sessions get guard (69231) ----------------------------------------------


class _SessClient:
    host = "https://example.databricks.com"

    def __init__(self):
        self.calls = []

    def get_session(self, session_id, store=None):
        self.calls.append(("get", session_id, store))
        return {"session_id": session_id, "session_store_name": store}

    def fork_session(self, store, source, actor_id, up_to=None, session_id=None, metadata=None):
        self.calls.append(("fork", store, source, actor_id))
        return {"session": {"session_id": "forked", "session_store_name": store}}


def test_sessions_get_without_store_errors_clearly():
    client = _SessClient()
    result = CliRunner().invoke(sessions, ["get", "sid"], obj=_Ctx(client))
    assert result.exit_code != 0
    assert "Provide --store" in result.output
    assert client.calls == []


# --- fork positional source id (69233) ---------------------------------------


def test_fork_accepts_positional_source_id():
    client = _SessClient()
    result = CliRunner().invoke(
        sessions, ["fork", "src-sid", "--store", "s", "--actor-id", "a"], obj=_Ctx(client)
    )
    assert result.exit_code == 0, result.output
    assert ("fork", "s", "src-sid", "a") in client.calls


def test_fork_still_accepts_flag():
    client = _SessClient()
    result = CliRunner().invoke(
        sessions,
        ["fork", "--source-session-id", "src-sid", "--store", "s", "--actor-id", "a"],
        obj=_Ctx(client),
    )
    assert result.exit_code == 0, result.output
    assert ("fork", "s", "src-sid", "a") in client.calls


def test_fork_missing_source_errors():
    client = _SessClient()
    result = CliRunner().invoke(sessions, ["fork", "--store", "s", "--actor-id", "a"], obj=_Ctx(client))
    assert result.exit_code != 0
    assert client.calls == []


def test_fork_ambiguous_source_errors():
    client = _SessClient()
    result = CliRunner().invoke(
        sessions,
        ["fork", "pos-sid", "--source-session-id", "opt-sid", "--store", "s", "--actor-id", "a"],
        obj=_Ctx(client),
    )
    assert result.exit_code != 0
    assert client.calls == []
