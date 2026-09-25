"""Unit tests for the public AgentKit client and private API transport."""

from __future__ import annotations

import pathlib
import subprocess
import types
from datetime import datetime, timezone
from unittest import mock

import pytest

from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit import AgentKitClient
from databricks_agentkit._api_client import (
    _AgentBricksApiClient,
    _workspace_client,
    memory_entry_path,
    memory_store_path,
    session_store_path,
)


def _client(workspace_client):
    inst = workspace_client.return_value
    inst.config.host = "https://ws.example.com"
    inst.api_client.do.return_value = {}
    return _AgentBricksApiClient(profile="p"), inst.api_client.do


@pytest.mark.parametrize(
    "client_class",
    [AgentKitClient],
    ids=["agentkit"],
)
@mock.patch("databricks_agentkit.client._AgentBricksApiClient")
def test_client_wraps_workspace_client(api_client, client_class):
    workspace_client = mock.Mock()

    client = client_class(workspace_client)

    api_client.assert_called_once_with(workspace_client=workspace_client)
    assert client.memory_stores._api is api_client.return_value
    assert client.session_stores._api is api_client.return_value


def test_private_api_client_exposes_its_workspace_client():
    workspace_client = mock.Mock()

    client = _AgentBricksApiClient(workspace_client=workspace_client)

    assert client.workspace_client is workspace_client


@mock.patch("databricks.sdk.WorkspaceClient")
def test_create_memory_store_uses_transitional_request(workspace_client):
    c, do = _client(workspace_client)
    c.create_memory_store("acme", "desc")
    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/memory-stores",
        query={"managed_memory_store_id": "acme"},
        body={"display_name": "acme", "description": "desc"},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_create_runtime_store_uses_v2_api_and_app_owner(workspace_client):
    client, do = _client(workspace_client)

    client.create_runtime_store(
        "agent-bricks-app-abc123", "sp-123", app_name="agent-bricks-app", retry_transient=True
    )

    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/runtime-stores",
        query={"runtime_store_id": "agent-bricks-app-abc123"},
        body={"owner": {"app": {"name": "agent-bricks-app", "service_principal_id": "sp-123"}}},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_get_and_delete_runtime_store_use_v2_resource_names(workspace_client):
    client, do = _client(workspace_client)
    client.get_runtime_store("agent-bricks-app")
    client.delete_runtime_store("agent-bricks-app")
    assert do.call_args_list == [
        mock.call("GET", "/api/2.0/agents/runtime-stores/agent-bricks-app", query=None, body=None),
        mock.call(
            "DELETE",
            "/api/2.0/agents/runtime-stores/agent-bricks-app",
            query=None,
            body=None,
        ),
    ]


@mock.patch("databricks.sdk.WorkspaceClient")
@pytest.mark.parametrize(
    ("description", "metadata", "expected_body"),
    [
        (None, None, {}),
        (
            "Support history",
            {"environment": "poc"},
            {"description": "Support history", "metadata": {"environment": "poc"}},
        ),
        ("", {}, {"description": "", "metadata": {}}),
    ],
)
def test_create_session_store_uses_transitional_request(
    workspace_client, description, metadata, expected_body
):
    client, do = _client(workspace_client)
    do.return_value = {"session_store_name": "support-history", "session_store_id": "store-uuid"}

    store = client.create_session_store("support-history", description, metadata)

    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/session-stores",
        query={"session_store_name": "support-history", "session_store_id": "support-history"},
        body=expected_body,
    )
    assert store.session_store_name == "support-history"
    assert store.session_store_id == "store-uuid"


@pytest.mark.parametrize(
    ("identity", "expected_id"),
    [
        pytest.param(
            {"session_store_name": "support-history", "session_store_id": "store-uuid"},
            "store-uuid",
            id="legacy",
        ),
        pytest.param({"name": "session-stores/support-history"}, None, id="canonical"),
        pytest.param(
            {
                "name": "session-stores/support-history",
                "session_store_name": "support-history",
                "session_store_id": "store-uuid",
            },
            "store-uuid",
            id="both",
        ),
    ],
)
def test_session_store_response_compatibility(identity, expected_id):
    """Mock workspace HTTP responses to exercise the real transport and resource parsers."""
    workspace = mock.Mock()
    client = AgentKitClient(workspace)
    payload = {
        **identity,
        "description": "Support history",
        "metadata": {"environment": "poc"},
        "creator_user_id": "123",
        "create_time": "2026-08-14T01:02:03Z",
        "update_time": "2026-08-14T02:03:04.500Z",
    }
    updated_payload = {**payload, "metadata": {"environment": "prod"}}
    workspace.api_client.do.side_effect = [
        payload,
        payload,
        {"session_stores": [payload]},
        updated_payload,
        {"session_id": "case-456", "actor_id": "customer-123"},
        {},
    ]

    created = client.session_stores.create("support-history")
    fetched = client.session_stores.get(created.name)
    listed = list(client.session_stores.list())
    updated = created.update(metadata={"environment": "prod"})
    session = created.add(actor_id="customer-123", session_id="case-456")
    updated.delete()

    for store in [created, fetched, *listed, updated]:
        assert store.name == "support-history"
        assert store.session_store_id == expected_id
        assert store.description == "Support history"
        assert store.creator_user_id == "123"
        assert store.create_time == datetime(2026, 8, 14, 1, 2, 3, tzinfo=timezone.utc)
        assert store.update_time == datetime(2026, 8, 14, 2, 3, 4, 500000, tzinfo=timezone.utc)
    assert len(listed) == 1
    assert created.metadata == {"environment": "poc"}
    assert updated.metadata == {"environment": "prod"}
    assert session.store_name == "support-history"
    assert [call.args for call in workspace.api_client.do.call_args_list] == [
        ("POST", "/api/2.0/agents/session-stores"),
        ("GET", "/api/2.0/agents/session-stores/support-history"),
        ("GET", "/api/2.0/agents/session-stores"),
        ("PATCH", "/api/2.0/agents/session-stores/support-history"),
        ("POST", "/api/2.0/agents/session-stores/support-history/sessions"),
        ("DELETE", "/api/2.0/agents/session-stores/support-history"),
    ]


@mock.patch("databricks.sdk.WorkspaceClient")
def test_list_memory_stores_query(workspace_client):
    c, do = _client(workspace_client)
    c.list_memory_stores(page_size=10)
    do.assert_called_once_with(
        "GET", "/api/2.0/agents/memory-stores", query={"page_size": 10}, body=None
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_list_mcp_services_query(workspace_client):
    c, do = _client(workspace_client)

    c.list_mcp_services("system.ai", page_token="next")

    do.assert_called_once_with(
        "GET",
        "/api/2.1/unity-catalog/mcp-services",
        query={"parent": "schemas/system.ai", "page_token": "next"},
        body=None,
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_get_mcp_service_uses_encoded_resource_path(workspace_client):
    client, do = _client(workspace_client)
    do.return_value = {"name": "mcp-services/main.tools.search"}

    result = client.get_mcp_service("main.tools.search")

    assert result == {"name": "mcp-services/main.tools.search"}
    do.assert_called_once_with(
        "GET", "/api/2.1/unity-catalog/mcp-services/main.tools.search", query=None, body=None
    )
    do.reset_mock()
    client.get_mcp_service("main.tools.search/extra?query=value")
    do.assert_called_once_with(
        "GET",
        "/api/2.1/unity-catalog/mcp-services/main.tools.search%2Fextra%3Fquery%3Dvalue",
        query=None,
        body=None,
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_get_memory_store_normalizes_id(workspace_client):
    c, do = _client(workspace_client)
    c.get_memory_store("abc123")
    do.assert_called_once_with("GET", "/api/2.0/agents/memory-stores/abc123", query=None, body=None)


@mock.patch("databricks.sdk.WorkspaceClient")
def test_update_memory_store_retains_empty_description(workspace_client):
    client, do = _client(workspace_client)

    client.update_memory_store("abc123", description="")

    do.assert_called_once_with(
        "PATCH",
        "/api/2.0/agents/memory-stores/abc123",
        query={"update_mask": "description"},
        body={"description": ""},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_get_memory_entry_passes_read_mask(workspace_client):
    client, do = _client(workspace_client)

    client.get_memory_entry("abc123", "entry1", read_mask="name,content")

    do.assert_called_once_with(
        "GET",
        "/api/2.0/agents/memory-stores/abc123/entries/entry1",
        query={"read_mask": "name,content"},
        body=None,
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_create_memory_entry_serializes_write_mode(workspace_client):
    client, do = _client(workspace_client)

    client.create_memory_entry(
        "s1",
        "alice",
        "/preferences/style.md",
        content="more",
        write_mode="MANAGED_MEMORY_ENTRY_WRITE_MODE_APPEND",
    )

    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/memory-stores/s1/entries",
        query=None,
        body={
            "actor_id": "alice",
            "path": "/preferences/style.md",
            "content": "more",
            "write_mode": "MANAGED_MEMORY_ENTRY_WRITE_MODE_APPEND",
        },
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_search_memory_entries(workspace_client):
    c, do = _client(workspace_client)
    c.search_memory_entries("s1", "alice", "style", limit=5)
    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/memory-stores/s1/entries:search",
        query=None,
        body={"actor_id": "alice", "query": "style", "limit": 5},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_search_memory_entries_with_resource_filters(workspace_client):
    client, do = _client(workspace_client)

    client.search_memory_entries(
        "s1",
        "alice",
        "style",
        page_size=5,
        path_prefix="/preferences/",
        session_id="session1",
        read_mask="name,path,content",
    )

    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/memory-stores/s1/entries:search",
        query=None,
        body={
            "actor_id": "alice",
            "query": "style",
            "page_size": 5,
            "path_prefix": "/preferences/",
            "session_id": "session1",
            "read_mask": "name,path,content",
        },
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_create_session_puts_session_id_in_query(workspace_client):
    c, do = _client(workspace_client)
    c.create_session("store1", "alice", session_id="sid")
    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/session-stores/store1/sessions",
        query={"session_id": "sid"},
        body={"actor_id": "alice"},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_get_session_uses_store_scoped_path(workspace_client):
    c, do = _client(workspace_client)
    c.get_session("sid", store="store1")
    assert do.call_args.args[1] == "/api/2.0/agents/session-stores/store1/sessions/sid"


@mock.patch("databricks.sdk.WorkspaceClient")
def test_append_wraps_items_in_data(workspace_client):
    c, do = _client(workspace_client)
    c.append_session_items("store1", "sid", [{"role": "user", "content": "hi"}])
    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/session-stores/store1/sessions/sid/items:append",
        query=None,
        body={"items": [{"data": {"role": "user", "content": "hi"}}]},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_extract_memories_posts_to_2_0_extractions(workspace_client):
    c, do = _client(workspace_client)
    c.extract_memories("store1", "sid", "mem", instructions="only prefs")
    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/session-stores/store1/sessions/sid/extractions",
        query=None,
        # memory_store is normalized to its resource form; dry_run defaults off (omitted).
        body={"memory_store": "memory-stores/mem", "instructions": "only prefs"},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_extract_memories_dry_run_and_bare_memory_store(workspace_client):
    c, do = _client(workspace_client)
    c.extract_memories("store1", "sid", "memory-stores/mem", dry_run=True)
    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/session-stores/store1/sessions/sid/extractions",
        query=None,
        body={"memory_store": "memory-stores/mem", "dry_run": True},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_extract_memories_wraps_entries(workspace_client):
    c, do = _client(workspace_client)
    do.return_value = {
        "name": "extractions/abc",
        "entries": [{"name": "memory-stores/m/entries/e1", "path": "/prefs/style.md"}],
    }
    result = c.extract_memories("store1", "sid", "mem")
    assert result.name == "extractions/abc"
    assert [e.path for e in result.entries] == ["/prefs/style.md"]


@mock.patch("databricks.sdk.WorkspaceClient")
def test_delete_session_without_force(workspace_client):
    c, do = _client(workspace_client)
    c.delete_session("store1", "sid")
    do.assert_called_once_with(
        "DELETE",
        "/api/2.0/agents/session-stores/store1/sessions/sid",
        query=None,
        body=None,
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_delete_session_with_force(workspace_client):
    c, do = _client(workspace_client)
    c.delete_session("store1", "sid", force=True)
    do.assert_called_once_with(
        "DELETE",
        "/api/2.0/agents/session-stores/store1/sessions/sid",
        query={"force": True},
        body=None,
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_grant_session_store_permission(workspace_client):
    c, do = _client(workspace_client)
    c.grant_session_store_permission("sess-1", "sp-abc")
    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/session-stores/sess-1/permissions:grant",
        query=None,
        body={"principal": {"type": "SERVICE_PRINCIPAL", "name": "sp-abc"}, "permission": "WRITE"},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_grant_memory_store_permission_takes_resource_id_and_level(workspace_client):
    c, do = _client(workspace_client)
    c.grant_memory_store_permission("memory-stores/uuid-x", "sp-abc", permission="READ")
    do.assert_called_once_with(
        "POST",
        "/api/2.0/agents/memory-stores/uuid-x/permissions:grant",
        query=None,
        body={"principal": {"type": "SERVICE_PRINCIPAL", "name": "sp-abc"}, "permission": "READ"},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
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


@mock.patch(
    "databricks_agentkit._api_client._workspace_client", side_effect=RuntimeError("no auth")
)
def test_auth_error_hint_explains_profile_selection_and_login(_workspace_client):
    with pytest.raises(AgentCliError) as exc_info:
        _AgentBricksApiClient()

    hint = exc_info.value.hint
    assert hint is not None
    assert "`agentbricks --profile <name> <command>`" in hint
    assert "`agentbricks login --profile <name>`" in hint
    assert "`databricks auth login --profile <name>`" not in hint


def test_profile_auth_is_forwarded_when_multiple_profiles_share_a_host(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "databrickscfg"
    config_path.write_text(
        "[duplicate]\n"
        "host = https://workspace.cloud.databricks.com\n"
        "account_id = account-id\n"
        "discovery_url = https://workspace.cloud.databricks.com/oidc/.well-known/oauth-authorization-server\n"
        "auth_type = databricks-cli\n"
        "\n"
        "[selected]\n"
        "host = https://workspace.cloud.databricks.com\n"
        "account_id = account-id\n"
        "discovery_url = https://workspace.cloud.databricks.com/oidc/.well-known/oauth-authorization-server\n"
        "auth_type = databricks-cli\n"
    )
    for name in (
        "DATABRICKS_AUTH_TYPE",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_CONFIG_PROFILE",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(config_path))
    monkeypatch.setenv("DATABRICKS_CLI_PATH", "/fake/databricks")
    monkeypatch.setenv("DATABRICKS_DISABLE_ASYNC_TOKEN_REFRESH", "true")

    def run_cli(args: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        if args[1:] == ["version", "--output", "json"]:
            stdout = b'{"Major": 0, "Minor": 300, "Patch": 0}'
        else:
            assert args[1:3] == ["auth", "token"]
            assert "--host" not in args
            assert args[args.index("--profile") + 1] == "selected"
            stdout = (
                b'{"access_token": "test-token", "token_type": "Bearer", '
                b'"expiry": "2099-01-01T00:00:00Z"}'
            )
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=b"")

    host_metadata = types.SimpleNamespace(
        account_id=None,
        workspace_id=None,
        oidc_endpoint="",
        cloud=None,
        token_federation_default_oidc_audiences=[],
    )
    # Keep the test hermetic while exercising the SDK's real CLI command selection.
    with (
        mock.patch("databricks.sdk.config.get_host_metadata", return_value=host_metadata),
        mock.patch("databricks.sdk.credentials_provider._run_subprocess", side_effect=run_cli),
    ):
        client = _workspace_client("selected")

    assert client.config.profile == "selected"
    # The interactive client caps the SDK's retry budget so throttling fails fast, not after ~5 min.
    assert client.api_client._api_client._retry_timeout_seconds == 60


class _TransientError(RuntimeError):
    error_code = "CANCELLED"


@mock.patch("databricks_agentkit._api_client.time.sleep")
@mock.patch("databricks.sdk.WorkspaceClient")
def test_delete_runtime_store_retries_transient_errors(workspace_client, sleep):
    client, do = _client(workspace_client)
    do.side_effect = [_TransientError("interrupted"), {}]

    assert client.delete_runtime_store("agent-bricks-app") == {}
    assert do.call_count == 2
    sleep.assert_called_once()


@mock.patch("databricks_agentkit._api_client.time.sleep")
@mock.patch("databricks.sdk.WorkspaceClient")
def test_do_retries_transient_error_then_succeeds(workspace_client, sleep):
    client, do = _client(workspace_client)
    do.side_effect = [_TransientError(None), {"session_store_name": "s"}]

    store = client.create_session_store("s", retry_transient=True)

    assert store["session_store_name"] == "s"
    assert do.call_count == 2
    sleep.assert_called_once()


@mock.patch("databricks_agentkit._api_client.time.sleep")
@mock.patch("databricks.sdk.WorkspaceClient")
@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("create_memory_store", ("m",)),
        ("create_session_store", ("s",)),
    ],
)
def test_create_store_does_not_retry_transient_error_by_default(
    workspace_client, sleep, method_name, args
):
    client, do = _client(workspace_client)
    do.side_effect = _TransientError(None)

    with pytest.raises(AgentCliError):
        getattr(client, method_name)(*args)

    assert do.call_count == 1
    sleep.assert_not_called()


@mock.patch("databricks_agentkit._api_client.time.sleep")
@mock.patch("databricks.sdk.WorkspaceClient")
def test_do_stops_after_max_attempts(workspace_client, sleep):
    client, do = _client(workspace_client)
    do.side_effect = _TransientError(None)

    with pytest.raises(AgentCliError) as exc_info:
        client.list_memory_stores()

    assert do.call_count == 2
    assert exc_info.value.error_code == "CANCELLED"
    assert "None" not in exc_info.value.message


@mock.patch("databricks_agentkit._api_client.time.sleep")
@mock.patch("databricks.sdk.WorkspaceClient")
def test_do_does_not_retry_non_transient_error(workspace_client, sleep):
    client, do = _client(workspace_client)

    class NotFoundError(RuntimeError):
        error_code = "NOT_FOUND"

    do.side_effect = NotFoundError("missing")

    with pytest.raises(AgentCliError):
        client.list_memory_stores()

    assert do.call_count == 1
    sleep.assert_not_called()


@mock.patch("databricks_agentkit._api_client.time.sleep")
@mock.patch("databricks.sdk.WorkspaceClient")
def test_do_does_not_retry_transient_error_for_pop(workspace_client, sleep):
    client, do = _client(workspace_client)
    do.side_effect = _TransientError(None)

    with pytest.raises(AgentCliError):
        client.pop_session_item("store1", "sid")

    assert do.call_count == 1
    sleep.assert_not_called()


def test_account_routed_profile_uses_configured_host_and_workspace_header():
    resolved = mock.Mock()
    resolved.config.host = "https://workspace.example.com"
    resolved.config.workspace_id = "123"
    routed = mock.Mock()

    with (
        mock.patch("databricks.sdk.WorkspaceClient", side_effect=[resolved, routed]) as wc,
        mock.patch(
            "databricks_agentkit._api_client._profile_host",
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


def test_memory_store_path_rejects_empty_and_wrong_type():
    for bad in ["", "   ", "/", "memory-stores/"]:
        with pytest.raises(AgentCliError):
            memory_store_path(bad)
    with pytest.raises(AgentCliError):
        memory_store_path("session-stores/abc")


def test_session_store_path_rejects_empty():
    for bad in ["", "   ", "/", "session-stores/"]:
        with pytest.raises(AgentCliError):
            session_store_path(bad)
    assert session_store_path("s1") == "session-stores/s1"
    assert session_store_path("session-stores/s1") == "session-stores/s1"


def test_memory_entry_path_rejects_empty_entry():
    with pytest.raises(AgentCliError):
        memory_entry_path("s", "")


@mock.patch("databricks.sdk.WorkspaceClient")
def test_update_memory_store_no_fields_raises_without_calling_api(workspace_client):
    client, do = _client(workspace_client)
    with pytest.raises(AgentCliError):
        client.update_memory_store("abc")
    do.assert_not_called()


@mock.patch("databricks.sdk.WorkspaceClient")
def test_update_session_store_no_fields_raises_without_calling_api(workspace_client):
    client, do = _client(workspace_client)
    with pytest.raises(AgentCliError):
        client.update_session_store("s1")
    do.assert_not_called()


@mock.patch("databricks.sdk.WorkspaceClient")
def test_update_memory_entry_sends_update_mask(workspace_client):
    client, do = _client(workspace_client)

    client.update_memory_entry("s", "e", content="updated", description="")

    do.assert_called_once_with(
        "PATCH",
        "/api/2.0/agents/memory-stores/s/entries/e",
        query={"update_mask": "content,description"},
        body={"content": "updated", "description": ""},
    )


@mock.patch("databricks.sdk.WorkspaceClient")
def test_update_memory_entry_no_fields_raises_without_calling_api(workspace_client):
    client, do = _client(workspace_client)
    with pytest.raises(AgentCliError):
        client.update_memory_entry("s", "memory-stores/s/entries/e")
    do.assert_not_called()


@mock.patch("databricks.sdk.WorkspaceClient")
def test_delete_session_store_normalizes_path(workspace_client):
    client, do = _client(workspace_client)
    client.delete_session_store("session-stores/s1")
    do.assert_called_once_with("DELETE", "/api/2.0/agents/session-stores/s1", query=None, body=None)
