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
            {"path": "/note.md", "content": "[bold]literal[/bold]"},
            {"path": "/baseline.md", "change_type": "MEMORY_CHANGE_TYPE_BASELINE"},
            {"path": "/deleted.md", "change_type": "MEMORY_CHANGE_TYPE_DELETE"},
        ],
    }
    result = run(["get", "c1", "--store", "store-1"], output="text")
    assert result.exit_code == 0, result.output
    for text in [
        "[bold]literal[/bold]",
        "/baseline.md",
        "/deleted.md",
        "3 entry versions",
    ]:
        assert text in result.stdout
    assert "--page-token" not in result.stdout
    # None of these entries has a predecessor, so there is nothing more to read.
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
    for text in ["c1", "AGENT", "session:[s1]", "+1 ~2 -3", "More changes available"]:
        assert text in result.stdout
    # The token is long and opaque; text output must not ask people to copy it.
    assert "next-token" not in result.stdout


@pytest.mark.parametrize(
    "header,reference",
    [
        ({"session_id": "case-755"}, "session:case-755"),
        ({"pipeline_run": "memory-pipelines/p1/runs/42"}, "run:42"),
        ({"request_id": "req-1"}, "request:req-1"),
    ],
)
def test_list_text_labels_each_reference_with_its_kind(workspace, header, reference):
    # Session, run, and request ids look nothing alike; the label says which one a row shows.
    workspace.api_client.do.return_value = {
        "memory_changes": [{"name": "memory-stores/store-1/changes/c1", **header}]
    }
    result = run(["list", "--store", "store-1"], output="text")
    assert result.exit_code == 0, result.output
    assert reference in result.stdout


def test_list_text_shows_the_actor_from_each_change(workspace):
    # List rows have no actor, so the table reads it from the change's entry versions.
    def do(method, path, **kwargs):
        if path == BASE:
            return {"memory_changes": [{"name": "memory-stores/store-1/changes/c1"}]}
        assert path == BASE + "/c1"
        return {"entries": [{"actor_id": "alice"}, {"actor_id": "bob"}, {"actor_id": "alice"}]}

    workspace.api_client.do.side_effect = do
    result = run(["list", "--store", "store-1"], output="text")
    assert result.exit_code == 0, result.output
    assert "alice, bob" in result.stdout


def test_list_json_makes_no_extra_requests(workspace):
    workspace.api_client.do.return_value = {
        "memory_changes": [{"name": "memory-stores/store-1/changes/c1"}]
    }
    assert run(["list", "--store", "store-1"]).exit_code == 0
    assert workspace.api_client.do.call_count == 1


def test_list_text_on_a_terminal_offers_the_next_page(workspace, monkeypatch):
    monkeypatch.setattr("databricks_agentbricks.cli.memory_changes._interactive", lambda: True)
    pages = {
        None: {"memory_changes": [{"name": "x/changes/first"}], "next_page_token": "t2"},
        "t2": {"memory_changes": [{"name": "x/changes/second"}]},
    }

    def do(method, path, query=None, **kwargs):
        return pages[(query or {}).get("page_token")] if path == BASE else {}

    workspace.api_client.do.side_effect = do
    result = CliRunner().invoke(
        agentbricks,
        ["--profile", "test", "memory", "changes", "list", "--store", "store-1"],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert "first" in result.stdout and "second" in result.stdout
    assert "t2" not in result.stdout


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


ENTRY = "memory-stores/store-1/entries/e1"


def _changes(by_id):
    def do(method, path, **kwargs):
        return by_id[path.rsplit("/", 1)[-1]]

    return do


def test_diff_shows_what_the_change_did_to_each_entry(workspace):
    # "What did this change do?" is the question history exists to answer; get must answer it
    # from the predecessor's content, not just restate the new content.
    workspace.api_client.do.side_effect = _changes(
        {
            "c2": {
                "entries": [
                    {
                        "memory_entry": ENTRY,
                        "path": "/profile.md",
                        "previous_change_id": "c1",
                        "content": "Name: Maya\nRole: lead\n[bold]literal[/bold]",
                    }
                ]
            },
            "c1": {
                "entries": [
                    {"memory_entry": "memory-stores/store-1/entries/other", "content": "unrelated"},
                    {"memory_entry": ENTRY, "content": "Name: Maya\nRole: engineer"},
                ]
            },
        }
    )
    result = run(["get", "c2", "--store", "store-1"], output="text")
    assert result.exit_code == 0, result.output
    for line in ["/profile.md diff", "  Name: Maya", "- Role: engineer", "+ Role: lead"]:
        assert line in result.stdout
    # The entry's fields already name the previous change; the diff title does not repeat it.
    assert "Entry  ›  /profile.md" in result.stdout
    assert result.stdout.count("c1") == 1
    assert "+ [bold]literal[/bold]" in result.stdout
    assert "unrelated" not in result.stdout


@pytest.mark.parametrize(
    "entry,previous,expected",
    [
        ({"content": "a\nb"}, None, ["+ a", "+ b"]),
        (
            {"change_type": "MEMORY_CHANGE_TYPE_DELETE", "previous_change_id": "c1"},
            {"content": "a\nb"},
            ["- a", "- b"],
        ),
    ],
)
def test_diff_of_a_new_or_deleted_entry_is_all_added_or_all_removed(
    workspace, entry, previous, expected
):
    by_id = {"c2": {"entries": [{"memory_entry": ENTRY, **entry}]}}
    if previous is not None:
        by_id["c1"] = {"entries": [{"memory_entry": ENTRY, **previous}]}
    workspace.api_client.do.side_effect = _changes(by_id)
    result = run(["get", "c2", "--store", "store-1"], output="text")
    assert result.exit_code == 0, result.output
    for line in expected:
        assert line in result.stdout


def test_diff_fails_instead_of_showing_a_wrong_diff(workspace):
    # A predecessor without this entry would otherwise diff against nothing and present the whole
    # entry as newly added.
    workspace.api_client.do.side_effect = _changes(
        {
            "c2": {
                "entries": [{"memory_entry": ENTRY, "previous_change_id": "c1", "content": "new"}]
            },
            "c1": {"entries": []},
        }
    )
    result = run(["get", "c2", "--store", "store-1"], output="text")
    assert result.exit_code == 1, result.output
    assert "has no version of" in result.stderr
    assert "+ new" not in result.stdout


def test_diff_without_this_versions_content_does_not_read_as_a_delete(workspace):
    workspace.api_client.do.side_effect = _changes(
        {
            "c2": {"entries": [{"memory_entry": ENTRY, "previous_change_id": "c1"}]},
            "c1": {"entries": [{"memory_entry": ENTRY, "content": "kept"}]},
        }
    )
    result = run(["get", "c2", "--store", "store-1"], output="text")
    assert result.exit_code == 0, result.output
    assert "no content to compare" in result.stdout
    assert "- kept" not in result.stdout


def test_json_get_does_not_read_predecessors(workspace):
    # JSON is the stored change; the diff and its extra requests belong to text output only.
    workspace.api_client.do.return_value = {
        "entries": [{"memory_entry": ENTRY, "previous_change_id": "c1", "content": "new"}]
    }
    result = run(["get", "c2", "--store", "store-1"])
    assert result.exit_code == 0, result.output
    assert workspace.api_client.do.call_count == 1
