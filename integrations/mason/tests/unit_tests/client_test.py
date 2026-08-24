"""Unit tests for AgentApiClient: path/verb/body per RPC + preview-error mapping.

Uses only unittest.mock (no `import pytest`): the Bazel db_py_test runner supplies a
vendored pytest that is not importable as `pytest`, so tests rely on the mock decorator
and built-in behavior rather than pytest fixtures.
"""

from __future__ import annotations

from unittest import mock

from databricks_mason.client import AgentApiClient, memory_entry_path, memory_store_path
from databricks_mason.errors import AgentCliError


def _client(workspace_client):
    inst = workspace_client.return_value
    inst.config.host = "https://ws.example.com"
    inst.api_client.do.return_value = {}
    return AgentApiClient(profile="p"), inst.api_client.do


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_create_memory_store(workspace_client):
    c, do = _client(workspace_client)
    c.create_memory_store("acme", "desc")
    do.assert_called_once_with(
        "POST", "/api/agents/v1/memory-stores", query=None, body={"display_name": "acme", "description": "desc"}
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_list_memory_stores_query(workspace_client):
    c, do = _client(workspace_client)
    c.list_memory_stores(page_size=10)
    do.assert_called_once_with("GET", "/api/agents/v1/memory-stores", query={"page_size": 10}, body=None)


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
        "POST", "/api/agents/v1/session-stores/store1/sessions", query={"session_id": "sid"}, body={"actor_id": "alice"}
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
        "DELETE", "/api/agents/v1/session-stores/store1/sessions/sid", query={"force": True}, body=None
    )


@mock.patch("databricks_mason.client.WorkspaceClient")
def test_preview_error_is_mapped_with_hint(workspace_client):
    c, do = _client(workspace_client)
    err = RuntimeError("not implemented")
    err.error_code = "NOT_IMPLEMENTED"
    do.side_effect = err
    try:
        c.list_memory_stores()
        assert False, "expected AgentCliError"
    except AgentCliError as mapped:
        assert mapped.error_code == "NOT_IMPLEMENTED"
        assert mapped.hint is not None


def test_path_helpers():
    assert memory_store_path("abc") == "memory-stores/abc"
    assert memory_store_path("memory-stores/abc") == "memory-stores/abc"
    assert memory_entry_path("s", "e") == "memory-stores/s/entries/e"
    assert memory_entry_path("s", "memory-stores/s/entries/e") == "memory-stores/s/entries/e"
