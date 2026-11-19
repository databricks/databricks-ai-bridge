"""Exercise history commands through the real CLI and transport, mocking only the SDK boundary."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from databricks.sdk.errors import DatabricksError

from databricks_agentbricks.cli.app import agentbricks
from databricks_agentkit._api_client import _AgentBricksApiClient

BASE = "/api/2.0/agents/memory-stores/store-1/changes"
SOURCE = "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_"


@pytest.fixture
def workspace(monkeypatch):
    # Authentication and HTTP are external; command parsing, rendering, and transport stay real.
    workspace = Mock()
    workspace.config.host = "https://workspace.example.com"
    workspace.api_client.do.return_value = {}
    monkeypatch.setattr("databricks_agentkit._api_client._workspace_client", lambda _: workspace)
    return workspace


def run(args, output="json"):
    return CliRunner().invoke(
        agentbricks, ["--profile", "test", "--output", output, "memory", "changes", *args]
    )


@pytest.mark.parametrize("store", ["store-1", "memory-stores/store-1"])
@pytest.mark.parametrize("path_option", ["--path", "--path-prefix"])
def test_list_filters_and_response_are_preserved(workspace, store, path_option):
    response = {
        "memory_changes": [{"name": "memory-stores/store-1/changes/c1"}],
        "next_page_token": "a+/=",
    }
    workspace.api_client.do.return_value = response
    result = run(
        [
            "list",
            "--store",
            store,
            "--actor-id",
            "alice",
            "--source-type",
            "dreamer",
            path_option,
            "/notes/été%_",
            "--memory-entry-id",
            "e1",
            "--page-size",
            "2",
            "--page-token",
            "opaque+/=",
        ]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == response
    workspace.api_client.do.assert_called_once_with(
        "GET",
        BASE,
        body=None,
        query={
            "actor_id": "alice",
            "source_type": SOURCE + "DREAMER",
            "path_prefix": "/notes/été%_",
            "memory_entry_id": "e1",
            "page_size": 2,
            "page_token": "opaque+/=",
        },
    )


@pytest.mark.parametrize(
    "source", ["agent", "DREAMER", "unspecified", SOURCE + "AGENT", (SOURCE + "DREAMER").lower()]
)
def test_source_aliases(workspace, source):
    result = run(["list", "--store", "store-1", "--source-type", source])
    assert result.exit_code == 0, result.output
    expected = SOURCE + source.upper().removeprefix(SOURCE)
    assert workspace.api_client.do.call_args.kwargs["query"]["source_type"] == expected


def test_omitted_source_means_all_sources_and_default_page_size(workspace):
    result = run(["list", "--store", "store-1"])
    assert result.exit_code == 0, result.output
    workspace.api_client.do.assert_called_once_with("GET", BASE, query={"page_size": 10}, body=None)


def test_get_sends_only_the_change_name_and_preserves_versions(workspace):
    # GetMemoryChange takes a name and nothing else, so a query here would be dropped or rejected.
    response = {
        "name": "memory-stores/store-1/changes/c2",
        "source_type": SOURCE + "DREAMER",
        "entries_updated": 1,
        "entries": [
            {
                "memory_entry": "memory-stores/store-1/entries/e1",
                "path": "/notes/été%_.md",
                "content": "[bold]literal[/bold]\nnew",
                "description": "",
                "previous_change_id": "c1",
            },
        ],
    }
    workspace.api_client.do.return_value = response
    result = run(["get", "c2", "--store", "store-1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == response
    workspace.api_client.do.assert_called_once_with("GET", BASE + "/c2", query=None, body=None)


def test_entry_history_is_listed_by_entry_identity_without_current_path(workspace):
    # A path can change between versions, so one entry's history is selected by its id alone.
    result = run(["list", "--store", "store-1", "--memory-entry-id", "e1"])
    assert result.exit_code == 0, result.output
    workspace.api_client.do.assert_called_once_with(
        "GET", BASE, query={"memory_entry_id": "e1", "page_size": 10}, body=None
    )


@pytest.mark.parametrize(
    "args",
    [
        ["list"],
        ["get", "c1"],
        ["get", "--store", "store-1"],
        ["list", "--store", "store-1", "--source-type", "manual"],
        *[
            ["list", "--store", "store-1", "--page-size", size]
            for size in ["0", "-1", "101", "2147483648", "1.5", "oops"]
        ],
        # Get is keyed by the change name alone: list's filters and paging are not accepted.
        *[
            ["get", "c1", "--store", "store-1", *option]
            for option in [
                ["--actor-id", "alice"],
                ["--memory-entry-id", "e1"],
                ["--path", "/note.md"],
                ["--page-size", "10"],
                ["--page-token", "token"],
            ]
        ],
    ],
)
def test_usage_errors_make_no_requests(workspace, args):
    result = run(args)
    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert "error" in json.loads(result.stderr)
    workspace.api_client.do.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        ["list", "--store", ""],
        ["list", "--store", "store?x=1"],
        ["list", "--store", "../store"],
        ["list", "--store", "x" * 256],
        ["get", "c1#fragment", "--store", "store-1"],
        ["get", "../c1", "--store", "store-1"],
        ["list", "--store", "store-1", "--memory-entry-id", "e1/extra"],
        *[
            ["list", "--store", "store-1", "--path", path]
            for path in ["relative", "/a//b", "/a/../b", "/a/./b", "/a\x00b"]
        ],
    ],
)
def test_invalid_resource_and_path_filters_fail_before_http(workspace, args):
    result = run(args)
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert "error" in json.loads(result.stderr)
    workspace.api_client.do.assert_not_called()


@pytest.mark.parametrize("size", [0, -1, 101, 2**31, True, 1.5])
def test_transport_rejects_invalid_page_sizes(workspace, size):
    client = _AgentBricksApiClient(workspace_client=workspace)
    with pytest.raises(ValueError):
        client.list_memory_changes("store-1", page_size=size)
    workspace.api_client.do.assert_not_called()


@pytest.mark.parametrize("command", [["list"], ["get", "c1"]])
@pytest.mark.parametrize(
    "code", ["FEATURE_DISABLED", "NOT_FOUND", "PERMISSION_DENIED", "BAD_REQUEST"]
)
def test_api_errors_are_nonzero_and_not_hidden(workspace, command, code):
    workspace.api_client.do.side_effect = DatabricksError("History request failed", error_code=code)
    result = run([*command, "--store", "store-1"])
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == code
    assert "Traceback" not in result.stderr
    assert workspace.api_client.do.call_count == 1


def test_text_get_renders_every_entry_version_literally(workspace):
    workspace.api_client.do.return_value = {
        "entries": [
            {"path": "/note.md", "content": "[bold]literal[/bold]", "previous_change_id": "c0"},
            {"path": "/baseline.md", "change_type": "MEMORY_CHANGE_TYPE_BASELINE"},
            {"path": "/deleted.md", "change_type": "MEMORY_CHANGE_TYPE_DELETE"},
        ],
    }
    result = run(["get", "c1", "--store", "store-1"], output="text")
    assert result.exit_code == 0, result.output
    for text in [
        "[bold]literal[/bold]",
        "c0",
        "/baseline.md",
        "/deleted.md",
        "3 entry versions",
    ]:
        assert text in result.stdout
    assert "--page-token" not in result.stdout
    assert workspace.api_client.do.call_count == 1


@pytest.mark.parametrize("command,payload", [(["list"], {}), (["get", "c1"], {"sessions_read": 0})])
@pytest.mark.parametrize("output", ["text", "json"])
def test_empty_and_metadata_only_changes(workspace, command, payload, output):
    workspace.api_client.do.return_value = payload
    result = run([*command, "--store", "store-1"], output=output)
    assert result.exit_code == 0, result.output
    if output == "json":
        assert json.loads(result.stdout) == payload
    else:
        assert "--page-token" not in result.stdout


def test_list_text_displays_provenance_counts_and_pagination(workspace):
    workspace.api_client.do.return_value = {
        "memory_changes": [
            {
                "name": "memory-stores/store-1/changes/c1",
                "source_type": SOURCE + "AGENT",
                "session_id": "[s1]",
                "entries_added": 1,
                "entries_updated": 2,
                "entries_deleted": 3,
            }
        ],
        "next_page_token": "next-token",
    }
    result = run(["list", "--store", "store-1"], output="text")
    assert result.exit_code == 0, result.output
    for text in ["c1", "AGENT", "[s1]", "+1 ~2 -3", "--page-token next-token"]:
        assert text in result.stdout


def test_history_source_options_do_not_allow_dreamer_entry_creation(workspace):
    result = CliRunner().invoke(
        agentbricks,
        [
            "memory",
            "entries",
            "create",
            "--store",
            "store-1",
            "--actor-id",
            "alice",
            "--path",
            "/note.md",
            "--content",
            "note",
            "--source-type",
            "dreamer",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Choose one of: agent, unspecified" in result.stderr
    workspace.api_client.do.assert_not_called()


def test_list_text_reads_changes_from_the_memory_changes_field(workspace):
    # ListMemoryChangesResponse names its list memory_changes; a table built from any other key
    # would show a store with history as empty.
    workspace.api_client.do.return_value = {
        "changes": [{"name": "memory-stores/store-1/changes/wrong-key"}],
        "memory_changes": [{"name": "memory-stores/store-1/changes/right-key"}],
    }
    result = run(["list", "--store", "store-1"], output="text")
    assert result.exit_code == 0, result.output
    assert "right-key" in result.stdout
    assert "wrong-key" not in result.stdout
