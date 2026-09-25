"""Contracts for project-independent available-tool discovery."""

import json

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli import app as cli
from databricks_agentbricks.errors import AgentCliError


class _Client:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []

    def list_mcp_services(self, schema, page_token=None):
        self.calls.append((schema, page_token))
        page = next(self.pages)
        if isinstance(page, Exception):
            raise page
        return page


def _invoke(monkeypatch, client, *args, output="json"):
    monkeypatch.setattr(cli.CliContext, "client", lambda self: client)
    return CliRunner().invoke(cli.agentbricks, ["--output", output, "tools", "list", *args])


def test_available_tools_without_project_normalizes_and_paginates(monkeypatch, tmp_path):
    client = _Client(
        [
            {
                "mcp_services": [
                    {"name": "mcp-services/system.ai.web_search"},
                    {"name": "mcp-services/system.ai.sandbox"},
                ],
                "next_page_token": "next",
            },
            {"mcp_services": [{"name": "system.ai.web_search"}, {"name": "system.ai.slack"}]},
        ]
    )
    monkeypatch.chdir(tmp_path)
    result = _invoke(monkeypatch, client)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 2
    assert payload["complete"] is True
    assert payload["errors"] == []
    assert payload["mcp_schema"] == "system.ai"
    assert [tool["name"] for tool in payload["available_tools"]] == [
        "sandbox",
        "uc-function",
        "genie-one",
        "genie-agent",
        "system.ai.slack",
        "system.ai.web_search",
    ]
    assert client.calls == [("system.ai", None), ("system.ai", "next")]
    assert "tools" not in payload


def test_kind_mcp_replaces_default_scope(monkeypatch):
    client = _Client([{"mcp_services": [{"name": "main.tools.search"}]}])
    result = _invoke(monkeypatch, client, "--kind", "mcp", "--schema", "main.tools")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mcp_schema"] == "main.tools"
    assert [tool["name"] for tool in payload["available_tools"]] == ["main.tools.search"]
    assert (
        payload["available_tools"][0]["add_command"]
        == "agentbricks tools add mcp main.tools.search"
    )
    assert client.calls == [("main.tools", None)]


@pytest.mark.parametrize(
    ("kind", "add_command"),
    [
        ("sandbox", "agentbricks tools add sandbox --scope table:catalog.schema.table"),
        ("uc-function", "agentbricks tools add uc-function catalog.schema.function"),
        ("genie-one", "agentbricks tools add genie-one"),
        ("genie-agent", "agentbricks tools add genie-agent SPACE_ID"),
    ],
)
def test_local_recipe_needs_no_client_or_project(monkeypatch, tmp_path, kind, add_command):
    def unexpected_client(self):
        pytest.fail("local recipes must not authenticate")

    monkeypatch.setattr(cli.CliContext, "client", unexpected_client)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli.agentbricks, ["-o", "json", "tools", "list", "--kind", kind])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mcp_schema"] is None
    assert [tool["kind"] for tool in payload["available_tools"]] == [kind]
    assert payload["available_tools"][0]["add_command"] == add_command


@pytest.mark.parametrize(
    "args",
    [
        ["--schema", "main.tools"],
        ["--kind", "sandbox", "--schema", "main.tools"],
        ["--kind", "mcp", "--schema", "main/tools"],
        ["--kind", "mcp", "--schema", "main"],
        ["--kind", "mcp", "--schema", ""],
        ["--source", "."],
    ],
)
def test_invalid_scope_and_removed_source_fail_before_api(monkeypatch, args):
    client = _Client([])
    result = _invoke(monkeypatch, client, *args)
    assert result.exit_code != 0
    assert client.calls == []


def test_discovered_sandbox_uses_scoped_recipe(monkeypatch):
    client = _Client([{"mcp_services": [{"name": "system.ai.sandbox"}]}])
    result = _invoke(monkeypatch, client, "--kind", "mcp")
    assert result.exit_code == 0, result.output
    tool = json.loads(result.stdout)["available_tools"][0]
    assert tool["kind"] == "mcp"
    assert tool["add_command"] == "agentbricks tools add sandbox --scope table:catalog.schema.table"


def test_empty_discovery_is_complete(monkeypatch):
    result = _invoke(monkeypatch, _Client([{}]), "--kind", "mcp")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["complete"] is True
    assert payload["available_tools"] == []


@pytest.mark.parametrize(
    "pages",
    [
        [{"mcp_services": [{"id": "missing-name"}]}],
        [{"mcp_services": [None]}],
        [{"mcp_services": [{"name": "mcp-services/"}]}],
        [{"mcp_services": [{"name": "system.ai"}]}],
        [{"mcp_services": [], "next_page_token": 42}],
        [{"next_page_token": "same"}, {"next_page_token": "same"}],
    ],
)
def test_malformed_discovery_cannot_claim_completeness(monkeypatch, pages):
    client = _Client(pages)
    result = _invoke(monkeypatch, client)
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["complete"] is False
    assert payload["errors"]
    assert [tool["kind"] for tool in payload["available_tools"]] == [
        "sandbox",
        "uc-function",
        "genie-one",
        "genie-agent",
    ]


@pytest.mark.parametrize("output", ["json", "text"])
def test_failed_discovery_preserves_recipes_and_reports_incomplete(monkeypatch, output):
    client = _Client([AgentCliError("Discovery denied", error_code="PERMISSION_DENIED")])
    result = _invoke(monkeypatch, client, output=output)
    assert result.exit_code == 1
    if output == "json":
        payload = json.loads(result.stdout)
        assert payload["complete"] is False
        assert [tool["kind"] for tool in payload["available_tools"]] == [
            "sandbox",
            "uc-function",
            "genie-one",
            "genie-agent",
        ]
        assert payload["errors"][0]["message"] == "Discovery denied"
        assert payload["errors"][0]["code"] == "PERMISSION_DENIED"
        assert result.stderr == ""
    else:
        assert "sandbox" in result.stdout
        assert "incomplete" in result.stdout.lower()
        assert "Discovery denied" in result.stderr


def test_discovery_text_labels_scope_and_manifest(monkeypatch):
    result = _invoke(monkeypatch, _Client([{}]), output="text")
    assert result.exit_code == 0, result.output
    assert "system.ai" in result.stdout
    assert "agent.toml" in result.stdout
    assert "recipe" in result.stdout.lower()
    assert "agentbricks tools list --kind mcp" in result.stdout
    assert "agentbricks tools list --kind mcp --schema catalog.schema" in result.stdout


def test_command_tree_has_no_legacy_or_nested_mcp_group():
    assert "mcp" not in cli.agentbricks.commands
    assert "mcp" not in cli.tools.commands


@pytest.mark.parametrize("path", [("tools",), ("tools", "list")])
def test_discovery_help_explains_available_not_configured(path):
    result = CliRunner().invoke(cli.agentbricks, [*path, "--help"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.stdout.split())
    assert "agent.toml" in text
    assert "available" in text.lower()
    assert "agentbricks mcp list" not in text
    assert "list configured tools" not in text


def test_list_help_explains_filters_and_project_independence():
    result = CliRunner().invoke(cli.agentbricks, ["tools", "list", "--help"])
    text = " ".join(result.stdout.split())
    for expected in (
        "--kind",
        "--schema",
        "catalog.schema",
        "system.ai",
        "genie-one",
        "genie-agent",
        "No agent project",
    ):
        assert expected in text
    assert "--source DIRECTORY" not in text
