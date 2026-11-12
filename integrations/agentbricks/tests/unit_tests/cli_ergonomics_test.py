"""Unit tests for CLI ergonomics fixes: optional --store, friendly source-type,
delete confirmation, flag aliases, sessions get guard, and fork positional arg."""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from databricks_agentbricks import errors
from databricks_agentbricks.cli.memory import _normalize_source_type, entries, stores
from databricks_agentbricks.cli.sessions import sessions
from databricks_agentbricks.errors import AgentCliError


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
    result = CliRunner().invoke(entries, ["get", "memory-stores/S/entries/E"], obj=_Ctx(client))
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
    result = CliRunner().invoke(
        sessions, ["fork", "--store", "s", "--actor-id", "a"], obj=_Ctx(client)
    )
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


# --- -o/--output after the command -------------------------------------------


class _StoreClient:
    host = "https://example.databricks.com"

    def create_session_store(self, name, description=None, metadata=None):
        return {"name": f"session-stores/{name}", "description": description}

    def create_memory_store(self, display_name, description=None):
        return {"name": "memory-stores/new", "display_name": display_name}


@pytest.fixture
def root(monkeypatch):
    """The real root group with a fake client; resets the process-global error mode afterwards."""
    from databricks_agentbricks.cli import app

    monkeypatch.setattr(app.CliContext, "client", lambda self: _StoreClient())
    yield app.agentbricks
    errors.set_output_mode("text")


def _leaf_paths(group, prefix=()):
    for name, command in group.commands.items():
        if isinstance(command, click.Group):
            yield from _leaf_paths(command, (*prefix, name))
        else:
            yield (*prefix, name)


@pytest.mark.parametrize(
    "args",
    [
        ["sessions", "stores", "create", "--name", "s"],
        ["memory", "stores", "create", "--name", "m"],
    ],
)
def test_output_after_command_matches_root_placement(root, args):
    before = CliRunner().invoke(root, ["-o", "json", *args])
    after = CliRunner().invoke(root, [*args, "-o", "json"])
    assert after.exit_code == before.exit_code == 0, after.output
    assert after.stdout == before.stdout
    assert json.loads(after.stdout)


@pytest.mark.parametrize("flag", [["--output", "json"], ["--output=json"], ["-ojson"]])
@pytest.mark.parametrize("first", [True, False])
def test_output_spellings_anywhere_among_options(root, flag, first):
    rest = ["--name", "s", "--description", "d"]
    args = [*flag, *rest] if first else [*rest, *flag]
    result = CliRunner().invoke(root, ["sessions", "stores", "create", *args])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["name"] == "session-stores/s"


def test_output_after_command_wins_over_root(root):
    cmd = ["sessions", "stores", "create", "--name", "s"]
    as_json = CliRunner().invoke(root, ["-o", "text", *cmd, "-o", "json"])
    assert json.loads(as_json.stdout)["name"] == "session-stores/s"
    as_text = CliRunner().invoke(root, ["-o", "json", *cmd, "-o", "text"])
    assert as_text.exit_code == 0
    assert not as_text.stdout.lstrip().startswith("{")


@pytest.mark.parametrize(
    "args",
    [
        ["sessions", "stores", "create", "--bogus", "-o", "json"],
        ["sessions", "stores", "create", "-o", "json"],
        ["memory", "stores", "create", "--bogus", "--output=json"],
    ],
)
def test_usage_errors_render_json_with_output_after_command(root, args):
    result = CliRunner().invoke(root, args)
    assert result.exit_code == 2
    error = json.loads(result.stderr[result.stderr.index("{") :])["error"]
    assert error["message"] and error["hint"]


@pytest.mark.parametrize(
    "args",
    [
        ["sessions", "frobnicate", "-o", "json"],
        ["frobnicate", "-o", "json"],
        ["--bogus", "sessions", "stores", "list", "-o", "json"],
    ],
)
def test_failures_before_the_command_still_render_json_with_trailing_output(root, args):
    """An unknown command or bad root option fires before any callback, so rendering follows the
    raw line's trailing `-o json` too."""
    result = CliRunner().invoke(root, args)
    assert result.exit_code != 0
    error = json.loads(result.stderr[result.stderr.index("{") :])["error"]
    assert error["message"] and error["hint"]


def test_telemetry_records_output_after_command_like_before_it(root, monkeypatch):
    from databricks_agentbricks.cli import telemetry

    records = []
    monkeypatch.setattr(
        telemetry,
        "emit_command",
        lambda ctx, **kwargs: records.append(telemetry._build_log(ctx, **kwargs).to_dict()),
    )
    result = CliRunner().invoke(root, ["sessions", "stores", "create", "--name", "s", "-o", "json"])
    assert result.exit_code == 0, result.output
    parameters = {entry["name"]: entry for entry in records[0]["parameters"]}
    assert parameters["agentbricks.output"] == {
        "name": "agentbricks.output",
        "choice_value": "AGENTBRICKS_CLI_JSON",
    }


def test_every_leaf_accepts_output_and_lists_it_in_help(root):
    leaves = list(_leaf_paths(root))
    assert len(leaves) > 50
    for path in leaves:
        result = CliRunner().invoke(root, [*path, "-o", "json", "--help"])
        assert result.exit_code == 0, (path, result.output)
        assert "-o, --output [text|json]" in result.stdout, path


def test_invalid_output_after_command_lists_choices(root):
    result = CliRunner().invoke(root, ["sessions", "stores", "create", "--name", "s", "-o", "yaml"])
    assert result.exit_code == 2
    assert "'yaml' is not one of 'text', 'json'" in result.stderr
