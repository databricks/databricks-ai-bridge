"""Unit tests for MasonClient paths, verbs, bodies, and error mapping."""

from __future__ import annotations

from unittest import mock

from databricks_mason.client import (
    MasonClient,
    _workspace_client,
    memory_entry_path,
    memory_store_path,
)
from databricks_mason.errors import AgentCliError


def _client(workspace_client):
    inst = workspace_client.return_value
    inst.config.host = "https://ws.example.com"
    inst.api_client.do.return_value = {}
    return MasonClient(profile="p"), inst.api_client.do


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_create_memory_store(workspace_client):
    c, do = _client(workspace_client)
    c.create_memory_store("acme", "desc")
    do.assert_called_once_with(
        "POST",
        "/api/agents/v1/memory-stores",
        query=None,
        body={"display_name": "acme", "description": "desc"},
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_list_memory_stores_query(workspace_client):
    c, do = _client(workspace_client)
    c.list_memory_stores(page_size=10)
    do.assert_called_once_with(
        "GET", "/api/agents/v1/memory-stores", query={"page_size": 10}, body=None
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_list_mcp_services_query(workspace_client):
    c, do = _client(workspace_client)

    c.list_mcp_services("system.ai", page_token="next")

    do.assert_called_once_with(
        "GET",
        "/api/2.1/unity-catalog/mcp-services",
        query={"parent": "schemas/system.ai", "page_token": "next"},
        body=None,
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_get_memory_store_normalizes_id(workspace_client):
    c, do = _client(workspace_client)
    c.get_memory_store("abc123")
    do.assert_called_once_with("GET", "/api/agents/v1/memory-stores/abc123", query=None, body=None)


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_search_memory_entries(workspace_client):
    c, do = _client(workspace_client)
    c.search_memory_entries("s1", "alice", "style", limit=5)
    do.assert_called_once_with(
        "POST",
        "/api/agents/v1/memory-stores/s1/entries:search",
        query=None,
        body={"actor_id": "alice", "query": "style", "limit": 5},
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_create_session_puts_session_id_in_query(workspace_client):
    c, do = _client(workspace_client)
    c.create_session("store1", "alice", session_id="sid")
    do.assert_called_once_with(
        "POST",
        "/api/agents/v1/session-stores/store1/sessions",
        query={"session_id": "sid"},
        body={"actor_id": "alice"},
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_get_session_scoped_vs_unscoped(workspace_client):
    c, do = _client(workspace_client)
    c.get_session("sid", store="store1")
    c.get_session("sid")
    assert do.call_args_list[0].args[1] == "/api/agents/v1/session-stores/store1/sessions/sid"
    assert do.call_args_list[1].args[1] == "/api/agents/v1/sessions/sid"


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_append_wraps_items_in_data(workspace_client):
    c, do = _client(workspace_client)
    c.append_session_items("store1", "sid", [{"role": "user", "content": "hi"}])
    do.assert_called_once_with(
        "POST",
        "/api/agents/v1/session-stores/store1/sessions/sid/items:append",
        query=None,
        body={"items": [{"data": {"role": "user", "content": "hi"}}]},
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_delete_session_force_flag(workspace_client):
    c, do = _client(workspace_client)
    c.delete_session("store1", "sid", force=True)
    do.assert_called_once_with(
        "DELETE",
        "/api/agents/v1/session-stores/store1/sessions/sid",
        query={"force": True},
        body=None,
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_preview_error_is_mapped_with_hint(workspace_client):
    c, do = _client(workspace_client)

    class PreviewApiError(RuntimeError):
        error_code = "NOT_IMPLEMENTED"

    err = PreviewApiError("not implemented")
    do.side_effect = err
    try:
        c.list_memory_stores()
        raise AssertionError("expected AgentCliError")
    except AgentCliError as mapped:
        assert mapped.error_code == "NOT_IMPLEMENTED"
        assert mapped.hint is not None


def test_account_routed_profile_uses_configured_host_and_workspace_header():
    resolved = mock.Mock()
    resolved.config.host = "https://workspace.example.com"
    resolved.config.workspace_id = "123"
    routed = mock.Mock()

    with (
        mock.patch("databricks_mason.client.WorkspaceClient", side_effect=[resolved, routed]) as wc,
        mock.patch(
            "databricks_mason.client._profile_host",
            return_value="https://account.example.com",
        ),
    ):
        client = _workspace_client("p")

    assert client is routed
    assert wc.call_args_list == [
        mock.call(profile="p"),
        mock.call(
            profile="p",
            host="https://account.example.com",
            custom_headers={"X-Databricks-Org-Id": "123"},
        ),
    ]


def test_path_helpers():
    assert memory_store_path("abc") == "memory-stores/abc"
    assert memory_store_path("memory-stores/abc") == "memory-stores/abc"
    assert memory_entry_path("s", "e") == "memory-stores/s/entries/e"
    assert memory_entry_path("s", "memory-stores/s/entries/e") == "memory-stores/s/entries/e"
