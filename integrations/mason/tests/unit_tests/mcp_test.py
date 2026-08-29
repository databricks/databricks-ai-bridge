"""Unit tests for ``mason mcp list`` discovery."""

from __future__ import annotations

import json

from click.testing import CliRunner

from databricks_mason.mcp import mcp


class _Client:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []

    def list_mcp_services(self, schema, page_token=None):
        self.calls.append((schema, page_token))
        return next(self.pages)


class _Ctx:
    def __init__(self, client, output="text"):
        self._client = client
        self.output = output

    def client(self):
        return self._client


def test_list_json_normalizes_sorts_deduplicates_and_paginates():
    client = _Client(
        [
            {
                "mcp_services": [
                    {
                        "name": "mcp-services/system.ai.web_search",
                        "id": "web-id",
                        "comment": "Search the web",
                    },
                    {"name": "mcp-services/system.ai.slack", "id": "slack-id"},
                    {"id": "missing-name"},
                ],
                "next_page_token": "page-2",
            },
            {
                "mcp_services": [
                    {"name": "mcp-services/system.ai.slack", "id": "duplicate"},
                    {"name": "mcp-services/system.ai.github", "id": "github-id"},
                ]
            },
        ]
    )

    result = CliRunner().invoke(mcp, ["list"], obj=_Ctx(client, output="json"))

    assert result.exit_code == 0, result.output
    assert client.calls == [("system.ai", None), ("system.ai", "page-2")]
    assert json.loads(result.output) == {
        "schema_version": 1,
        "mcp_services": [
            {"name": "system.ai.github", "id": "github-id"},
            {"name": "system.ai.slack", "id": "slack-id"},
            {
                "name": "system.ai.web_search",
                "id": "web-id",
                "comment": "Search the web",
            },
        ],
    }


def test_list_text_shows_copyable_add_command_and_schema_override():
    client = _Client(
        [{"mcp_services": [{"name": "mcp-services/main.tools.ticket_search", "id": "ticket-id"}]}]
    )

    result = CliRunner().invoke(
        mcp,
        ["list", "--schema", "main.tools"],
        obj=_Ctx(client),
    )

    assert result.exit_code == 0, result.output
    assert client.calls == [("main.tools", None)]
    assert "main.tools.ticket_search" in result.output
    assert "mason tools add mcp main.tools.ticket_search" in result.output


def test_list_rejects_invalid_schema_before_api_call():
    client = _Client([])

    result = CliRunner().invoke(mcp, ["list", "--schema", "system"], obj=_Ctx(client))

    assert result.exit_code != 0
    assert "catalog.schema" in result.output
    assert client.calls == []
