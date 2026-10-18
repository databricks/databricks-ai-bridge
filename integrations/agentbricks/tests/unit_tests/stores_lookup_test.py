"""Finding memory and session stores in a crowded workspace.

Covers `stores get` by display name, `stores list --name` / `--all` across pages, the
"more available" footer, untruncated identifiers, and the deploy summary's lookup commands.
"""

from __future__ import annotations

import io
import json
import types
from dataclasses import replace

import pytest
from click.testing import CliRunner
from rich.console import Console

from databricks_agentbricks.cli import memory as memory_cli
from databricks_agentbricks.cli import sessions as sessions_cli
from databricks_agentbricks.clients.conversation_store_client import MemoryStoreClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.deploy import present_deploy_result
from databricks_agentbricks.services.deploy_service import DeployResult


def _memory_store(store_id: str, display_name: str, **over) -> dict:
    store = {
        "name": f"memory-stores/{store_id}",
        "display_name": display_name,
        "owner_user_id": "1001",
        "create_time": "2026-10-17T15:22:00Z",
        "update_time": "2026-10-17T15:22:00Z",
    }
    store.update(over)
    return store


def _session_store(name: str, **over) -> dict:
    store = {
        "session_store_name": name,
        "session_store_id": f"id-{name}",
        "creator_user_id": "1001",
        "create_time": "2026-10-17T15:22:00Z",
        "update_time": "2026-10-17T15:22:00Z",
    }
    store.update(over)
    return store


class _PagedApi:
    """Fake API client whose list endpoints serve `per_page` stores at a time, whatever was asked."""

    host = "https://ws.example.com"

    def __init__(self, memory=(), sessions=(), *, per_page=2, get_error_code="NOT_FOUND"):
        self.memory = list(memory)
        self.sessions = list(sessions)
        self.per_page = per_page
        self.get_error_code = get_error_code
        self.calls: list[tuple] = []

    def _page(self, items, key, page_size, page_token):
        start = int(page_token or 0)
        end = start + self.per_page
        page = {key: items[start:end]}
        if end < len(items):
            page["next_page_token"] = str(end)
        return page

    def list_memory_stores(self, page_size=None, page_token=None):
        self.calls.append(("list_memory_stores", page_size, page_token))
        return self._page(self.memory, "managed_memory_stores", page_size, page_token)

    def list_session_stores(self, page_size=None, page_token=None):
        self.calls.append(("list_session_stores", page_size, page_token))
        return self._page(self.sessions, "session_stores", page_size, page_token)

    def get_memory_store(self, name):
        self.calls.append(("get_memory_store", name))
        store_id = name.removeprefix("memory-stores/")
        for store in self.memory:
            if store["name"] == f"memory-stores/{store_id}":
                return store
        raise AgentCliError(f"Store {name} not found", error_code=self.get_error_code)

    def get_session_store(self, name):
        self.calls.append(("get_session_store", name))
        for store in self.sessions:
            if store["session_store_name"] == name.removeprefix("session-stores/"):
                return store
        raise AgentCliError(f"Store {name} not found", error_code=self.get_error_code)


class _Ctx:
    def __init__(self, api, output="text"):
        self.output = output
        self._api = api
        self.api_client_provider = types.SimpleNamespace(get=lambda: api)

    def client(self):
        return self._api


def _run(group, args, api, output="text"):
    return CliRunner().invoke(group, args, obj=_Ctx(api, output=output))


def _flat(text: str) -> str:
    return " ".join(text.split())


def _crowded_memory() -> list[dict]:
    # Three pages of two: the agent's store is on the last page.
    return [
        _memory_store("a1", "someone-elses-memory"),
        _memory_store("a2", "Hello-World-memory", owner_user_id="2002"),
        _memory_store("a3", "other-memory"),
        _memory_store("a4", "another-memory"),
        _memory_store("a5", "unrelated"),
        _memory_store("a6", "hello-world-memory-x9"),
    ]


# --- memory stores get ---------------------------------------------------------


@pytest.mark.parametrize("arg", ["a3", "memory-stores/a3"])
def test_memory_get_by_id_or_resource_name_is_one_get(arg):
    api = _PagedApi(_crowded_memory())
    result = _run(memory_cli.stores, ["get", arg], api)
    assert result.exit_code == 0, result.output
    assert "other-memory" in result.output
    assert api.calls == [("get_memory_store", arg)]


def test_memory_get_resolves_display_name_across_pages():
    api = _PagedApi(_crowded_memory())
    result = _run(memory_cli.stores, ["get", "hello-world-memory-x9"], api)
    assert result.exit_code == 0, result.output
    assert "memory-stores/a6" in result.output
    assert api.calls[0] == ("get_memory_store", "hello-world-memory-x9")
    # Fell back to listing every page at the API maximum.
    assert [c for c in api.calls if c[0] == "list_memory_stores"] == [
        ("list_memory_stores", 100, None),
        ("list_memory_stores", 100, "2"),
        ("list_memory_stores", 100, "4"),
    ]

    json_result = _run(memory_cli.stores, ["get", "hello-world-memory-x9"], api, output="json")
    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output) == _crowded_memory()[5]


@pytest.mark.parametrize("code", ["RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"])
def test_memory_get_falls_back_on_invalid_id_codes(code):
    api = _PagedApi(_crowded_memory(), get_error_code=code)
    result = _run(memory_cli.stores, ["get", "other-memory"], api)
    assert result.exit_code == 0, result.output
    assert "memory-stores/a3" in result.output


def test_memory_get_name_with_slash_skips_the_get():
    api = _PagedApi([_memory_store("b1", "team/agent-memory")])
    result = _run(memory_cli.stores, ["get", "team/agent-memory"], api)
    assert result.exit_code == 0, result.output
    assert "memory-stores/b1" in result.output
    assert all(c[0] == "list_memory_stores" for c in api.calls)


def test_memory_get_permission_denied_is_not_swallowed():
    api = _PagedApi(_crowded_memory(), get_error_code="PERMISSION_DENIED")
    result = _run(memory_cli.stores, ["get", "hello-world-memory-x9"], api)
    assert result.exit_code != 0
    assert "not found" in result.output
    assert not any(c[0] == "list_memory_stores" for c in api.calls)


def test_memory_get_several_matches_lists_each_and_fails():
    stores = _crowded_memory() + [
        _memory_store("b7", "dup", owner_user_id="3003", create_time="2026-10-01T09:10:00Z"),
        _memory_store("b8", "other"),
        _memory_store("b9", "dup", owner_user_id="4004", create_time="2026-10-02T09:10:00Z"),
    ]
    result = _run(memory_cli.stores, ["get", "dup"], _PagedApi(stores))
    assert result.exit_code == 1
    out = _flat(result.output)
    assert "2 memory stores are named 'dup'" in out
    assert "memory-stores/b7 owner 3003 created Oct" in out
    assert "memory-stores/b9 owner 4004 created Oct" in out
    assert "Pass one resource name, e.g. agentbricks memory stores get memory-stores/b7" in out


def test_memory_get_no_match_hints_list_name():
    result = _run(memory_cli.stores, ["get", "agent-memroy"], _PagedApi(_crowded_memory()))
    assert result.exit_code == 1
    out = _flat(result.output)
    assert "No memory store named 'agent-memroy'" in out
    assert "agentbricks memory stores list --name agent-memroy" in out


def test_resolve_still_returns_the_first_match_and_stops_paging():
    stores = [_memory_store("c1", "dup"), _memory_store("c2", "x"), _memory_store("c3", "dup")]
    api = _PagedApi(stores)
    client = MemoryStoreClient(types.SimpleNamespace(get=lambda: api))  # type: ignore[arg-type]
    assert client.resolve("dup") == stores[0]
    assert len(api.calls) == 1
    assert client.find_by_display_name("dup") == [stores[0], stores[2]]


# --- sessions stores get -------------------------------------------------------


def test_session_get_by_name_is_unchanged():
    api = _PagedApi(sessions=[_session_store("agent-sessions")])
    result = _run(sessions_cli.stores, ["get", "agent-sessions"], api)
    assert result.exit_code == 0, result.output
    assert "session-stores/agent-sessions" in result.output
    assert api.calls == [("get_session_store", "agent-sessions")]


def test_session_get_not_found_hints_list_name():
    api = _PagedApi(sessions=[_session_store("agent-sessions")])
    result = _run(sessions_cli.stores, ["get", "agent-sesions"], api)
    assert result.exit_code == 1
    out = _flat(result.output)
    assert "No session store named 'agent-sesions'" in out
    assert "agentbricks sessions stores list --name agent-sesions" in out


def test_session_get_other_errors_are_not_rewritten():
    api = _PagedApi(get_error_code="PERMISSION_DENIED")
    result = _run(sessions_cli.stores, ["get", "agent-sessions"], api)
    assert result.exit_code == 1
    assert "No session store named" not in result.output


# --- stores list --name / --all ------------------------------------------------


def _crowded_sessions() -> list[dict]:
    return [
        _session_store("hello-sessions-1"),
        _session_store("other-a"),
        _session_store("other-b"),
        _session_store("other-c"),
        _session_store("HELLO-sessions-2"),
    ]


def test_memory_list_name_filters_case_insensitively_across_pages():
    api = _PagedApi(_crowded_memory())
    result = _run(memory_cli.stores, ["list", "--name", "HeLLo"], api)
    assert result.exit_code == 0, result.output
    assert "Hello-World-memory" in result.output
    assert "hello-world-memory-x9" in result.output
    assert "other-memory" not in result.output and "unrelated" not in result.output
    assert "name contains 'HeLLo'" in result.output
    assert "2 items" in result.output
    assert "Show next page?" not in result.output and "--page-token" not in result.output
    assert {c[1] for c in api.calls} == {100}


def test_sessions_list_name_filters_case_insensitively_across_pages():
    api = _PagedApi(sessions=_crowded_sessions())
    result = _run(sessions_cli.stores, ["list", "--name", "hello"], api)
    assert result.exit_code == 0, result.output
    assert "hello-sessions-1" in result.output and "HELLO-sessions-2" in result.output
    assert "other-" not in result.output
    assert len(api.calls) == 3


def test_memory_list_all_returns_every_store_in_order():
    result = _run(memory_cli.stores, ["list", "--all"], _PagedApi(_crowded_memory()))
    assert result.exit_code == 0, result.output
    names = [s["display_name"] for s in _crowded_memory()]
    positions = [result.output.index(f"{name} ") for name in names]
    assert positions == sorted(positions)
    assert "6 items" in result.output


@pytest.mark.parametrize(
    ("group", "api", "key", "args", "expected"),
    [
        (memory_cli.stores, _PagedApi(_crowded_memory()), "managed_memory_stores", ["--all"], 6),
        (
            memory_cli.stores,
            _PagedApi(_crowded_memory()),
            "managed_memory_stores",
            ["--name", "hello"],
            2,
        ),
        (
            sessions_cli.stores,
            _PagedApi(sessions=_crowded_sessions()),
            "session_stores",
            ["--all"],
            5,
        ),
        (
            sessions_cli.stores,
            _PagedApi(sessions=_crowded_sessions()),
            "session_stores",
            ["--name", "hello"],
            2,
        ),
    ],
)
def test_list_filters_print_one_json_document_without_token(group, api, key, args, expected):
    result = _run(group, ["list", *args], api, output="json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert list(payload) == [key]
    assert len(payload[key]) == expected


@pytest.mark.parametrize("group", [memory_cli.stores, sessions_cli.stores])
@pytest.mark.parametrize("flags", [["--name", "x"], ["--all"]])
def test_page_token_with_filters_is_a_usage_error(group, flags):
    api = _PagedApi(_crowded_memory(), _crowded_sessions())
    result = _run(group, ["list", *flags, "--page-token", "t"], api)
    assert result.exit_code == 2
    assert "--page-token can't be combined" in result.output
    assert api.calls == []


def test_filters_ignore_page_size():
    api = _PagedApi(_crowded_memory())
    result = _run(memory_cli.stores, ["list", "--all", "--page-size", "5"], api)
    assert result.exit_code == 0, result.output
    assert {c[1] for c in api.calls} == {100}


# --- unfiltered paging ---------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "stores", "key"),
    [
        (memory_cli.stores, _crowded_memory(), "managed_memory_stores"),
        (sessions_cli.stores, _crowded_sessions(), "session_stores"),
    ],
)
def test_unfiltered_json_is_one_raw_page(group, stores, key):
    api = _PagedApi(stores, stores)
    result = _run(group, ["list"], api, output="json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {key: stores[:2], "next_page_token": "2"}
    assert api.calls == [(f"list_{key.removeprefix('managed_')}", 25, None)]


@pytest.mark.parametrize(
    ("group", "api"),
    [
        (memory_cli.stores, _PagedApi(_crowded_memory())),
        (sessions_cli.stores, _PagedApi(sessions=_crowded_sessions())),
    ],
)
def test_unfiltered_text_ends_with_more_available_line(group, api):
    result = _run(group, ["list"], api)
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().splitlines()[-1] == (
        "2 shown; more available: --page-token 2, or use --all / --name"
    )
    assert "More results available" not in result.output
    assert "2 items" not in result.output


def test_unfiltered_last_page_keeps_the_item_count():
    result = _run(memory_cli.stores, ["list"], _PagedApi(_crowded_memory()[:2]))
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().splitlines()[-1] == "2 items"


# --- untruncated identifiers ---------------------------------------------------


def test_piped_memory_list_keeps_long_identifiers_on_one_line():
    display_name = "hello-world-custom-agent-memory-" + "x" * 38  # 70 chars
    store_id = "7ab1c2d3-0000-4e5f-9a8b-112233445566-abcdefghijklmn"  # 50 chars
    description = "a long description " * 5
    api = _PagedApi([_memory_store(store_id, display_name, description=description)])
    result = _run(memory_cli.stores, ["list"], api)  # CliRunner stdout isn't a terminal
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if display_name in line)
    assert store_id in row
    assert "…" in row  # the description is still shortened
    assert description.strip() not in row


def test_piped_sessions_list_keeps_long_names_on_one_line():
    name = "hello-world-custom-agent-sessions-" + "y" * 40
    api = _PagedApi(sessions=[_session_store(name, description="d")])
    result = _run(sessions_cli.stores, ["list", "--all"], api)
    assert result.exit_code == 0, result.output
    assert any(name in line for line in result.output.splitlines())


def test_terminal_table_folds_long_identifiers_instead_of_ellipsis():
    name = "hello-world-custom-agent-memory-" + "z" * 40
    buffer = io.StringIO()
    con = Console(file=buffer, width=60, force_terminal=True, color_system=None)
    render.resource_table(
        "Stores",
        [("Name", "left"), ("Description", "left")],
        [[name, "short description"]],
        con=con,
        no_wrap=[0],
    )
    out = buffer.getvalue()
    assert "…" not in out
    assert name in "".join(line.split("   ")[0].strip() for line in out.splitlines())


# --- help ----------------------------------------------------------------------


@pytest.mark.parametrize("group", ["memory", "sessions"])
def test_help_documents_filters_and_name_lookup(group):
    from databricks_agentbricks.cli.app import agentbricks

    listing = CliRunner().invoke(agentbricks, [group, "stores", "list", "--help"])
    assert listing.exit_code == 0, listing.output
    for text in ("--name TEXT", "--all", "--page-token", "not with --name or --all"):
        assert text in _flat(listing.output)
    assert f"agentbricks {group} stores list --name" in listing.output

    get = CliRunner().invoke(agentbricks, [group, "stores", "get", "--help"])
    assert get.exit_code == 0, get.output
    expected = "display name" if group == "memory" else "the name in agent.toml"
    assert expected in _flat(get.output)


# --- deploy summary ------------------------------------------------------------


def _deploy_result(**over) -> DeployResult:
    result = DeployResult(
        deployment="agent-bricks-demo",
        source="/tmp/agent",
        url="https://demo.example",
        workspace_path="/Workspace/demo",
        env={},
        client_host="https://workspace.example",
        memory_store=None,
        session_store=None,
        trace_experiment_id="123",
        uc_trace_tables=[],
        trace_setup_error=None,
        trace_grant_error=None,
        memory_grant_error=None,
        session_grant_error=None,
        memory_grant_attempted=False,
        session_grant_attempted=False,
        tool_access=None,
        created_app_yaml=False,
        pip_index_url=None,
        instance_count=None,
        uses_runtime_api=True,
    )
    return replace(result, **over)


def test_deploy_summary_prints_store_lookup_commands(capsys):
    present_deploy_result(
        _deploy_result(memory_store="hello memory", session_store="hello-sessions"), output=None
    )
    out = _flat(capsys.readouterr().out)
    assert "agentbricks memory stores get 'hello memory'" in out
    assert "agentbricks sessions stores get hello-sessions" in out


def test_deploy_summary_without_stores_has_no_lookup_commands(capsys):
    present_deploy_result(_deploy_result(), output=None)
    assert "stores get" not in capsys.readouterr().out


def test_deploy_json_is_unchanged_by_lookup_commands(capsys):
    present_deploy_result(
        _deploy_result(memory_store="hello-memory", session_store="hello-sessions"), output="json"
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["memory_store"] == "hello-memory"
    assert payload["session_store"] == "hello-sessions"
    assert "stores get" not in json.dumps(payload)
