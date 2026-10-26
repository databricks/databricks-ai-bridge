"""Permission SDK and CLI tests with only the external HTTP boundary replaced."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli.app import agentbricks
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit import AgentKitClient, Principal, StorePermission
from databricks_agentkit._api_client import _AgentBricksApiClient


def _permission(name="reader@example.com", level="READ", kind="USER"):
    return {"principal": {"principal_type": kind, "name": name}, "permission": level}


@pytest.fixture(params=["session", "memory"])
def target(request):
    kind = request.param
    resource = f"{kind}-stores/test-store"
    # Replace workspace HTTP only; SDK resource methods and API transport execute normally.
    do = Mock(return_value={})
    workspace = Mock(api_client=SimpleNamespace(do=do))
    client = AgentKitClient(workspace)
    api = _AgentBricksApiClient(workspace_client=workspace)
    metadata = (
        {"session_store_name": "test-store"}
        if kind == "session"
        else {"name": resource, "display_name": "Test store"}
    )
    do.return_value = metadata
    store = getattr(client, f"{kind}_stores").get("test-store")
    do.reset_mock()
    do.return_value = {}
    return SimpleNamespace(
        kind=kind,
        resource=resource,
        path=f"/api/2.0/agents/{resource}",
        key=f"{kind}_store_permissions",
        api=api,
        store=store,
        do=do,
    )


def test_sdk_grant_preserves_existing_default_and_supports_users(target):
    target.store.grant_permission("app-id")
    target.store.grant_permission("reader@example.com", principal_type="USER", permission="READ")
    assert target.do.call_args_list == [
        call(
            "POST",
            target.path + "/permissions:grant",
            query=None,
            body={
                "principal": {"principal_type": "SERVICE_PRINCIPAL", "name": "app-id"},
                "permission": "WRITE",
            },
        ),
        call("POST", target.path + "/permissions:grant", query=None, body=_permission()),
    ]


@pytest.mark.parametrize("level", ["READ", "WRITE", "PERMISSION_LEVEL_UNSPECIFIED", None])
def test_sdk_get_returns_typed_requester_permission(target, level):
    response = _permission(level=level)
    if level is None:
        response.pop("permission")
    target.do.return_value = response
    result = target.store.get_permission()
    assert isinstance(result, StorePermission)
    assert result.principal == Principal(principal_type="USER", name="reader@example.com")
    assert result.permission == (level or "PERMISSION_LEVEL_UNSPECIFIED")
    target.do.assert_called_once_with(
        "GET", target.path + "/permissions/self", query=None, body=None
    )


def test_sdk_list_follows_pages_and_wraps_both_principal_types(target):
    target.do.side_effect = [
        {target.key: [_permission()], "next_page_token": "next"},
        {target.key: [_permission("app-id", "WRITE", "SERVICE_PRINCIPAL")]},
    ]
    grants = list(target.store.list_permissions(page_size=1000))
    assert [(g.principal.name, g.permission) for g in grants] == [
        ("reader@example.com", "READ"),
        ("app-id", "WRITE"),
    ]
    assert grants[1].principal.principal_type == "SERVICE_PRINCIPAL"
    assert target.do.call_args_list == [
        call("GET", target.path + "/permissions", query={"page_size": 1000}, body=None),
        call(
            "GET",
            target.path + "/permissions",
            query={"page_size": 1000, "page_token": "next"},
            body=None,
        ),
    ]


def test_sdk_empty_list_and_revoke_payload(target):
    assert list(target.store.list_permissions()) == []
    target.do.reset_mock()
    target.store.revoke_permission("app-id")
    target.store.revoke_permission("reader@example.com", principal_type="USER")
    assert target.do.call_args_list == [
        call(
            "POST",
            target.path + "/permissions:revoke",
            query=None,
            body={"principal": {"principal_type": "SERVICE_PRINCIPAL", "name": "app-id"}},
        ),
        call(
            "POST",
            target.path + "/permissions:revoke",
            query=None,
            body={"principal": _permission()["principal"]},
        ),
    ]


@pytest.mark.parametrize("value", [0, -1, 1001, True, 1.5, "10"])
def test_sdk_permission_list_rejects_bad_page_size_before_http(target, value):
    with pytest.raises(ValueError, match="page_size"):
        list(target.store.list_permissions(page_size=value))
    target.do.assert_not_called()


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        ("grant", {"permission": "NONE"}),
        ("grant", {"permission": "PERMISSION_LEVEL_UNSPECIFIED"}),
        ("grant", {"principal_type": "GROUP"}),
        ("revoke", {"principal_type": "GROUP"}),
    ],
)
def test_sdk_rejects_invalid_grants_and_principal_types(target, operation, kwargs):
    with pytest.raises(ValueError):
        getattr(target.store, f"{operation}_permission")("app-id", **kwargs)
    target.do.assert_not_called()


@pytest.mark.parametrize("operation", ["grant", "revoke"])
def test_sdk_rejects_empty_principal(target, operation):
    with pytest.raises(ValueError, match="non-empty"):
        getattr(target.store, f"{operation}_permission")("   ")
    target.do.assert_not_called()


def _run(target, monkeypatch, args, *, output="json", input=None):
    monkeypatch.setattr(
        "databricks_agentbricks.clients.api_client_provider.ApiClientProvider.get",
        lambda self: target.api,
    )
    return CliRunner().invoke(
        agentbricks,
        [
            "--profile",
            "test-profile",
            "--output",
            output,
            "sessions" if target.kind == "session" else "memory",
            "stores",
            "permissions",
            *args,
        ],
        input=input,
    )


@pytest.mark.parametrize("store", ["test-store", "resource-name"])
def test_cli_get_uses_existing_store_identifiers_without_metadata_get(target, monkeypatch, store):
    name = target.resource if store == "resource-name" else store
    target.do.return_value = _permission(level="PERMISSION_LEVEL_UNSPECIFIED")
    result = _run(target, monkeypatch, ["get", name])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == target.do.return_value
    target.do.assert_called_once_with(
        "GET", target.path + "/permissions/self", query=None, body=None
    )


def test_cli_get_text_explains_no_access(target, monkeypatch):
    target.do.return_value = _permission(level="PERMISSION_LEVEL_UNSPECIFIED")
    result = _run(target, monkeypatch, ["get", "test-store"], output="text")
    assert result.exit_code == 0, result.output
    assert "No access" in result.output and "reader@example.com" in result.output


@pytest.mark.parametrize(
    "flag,kind,name",
    [
        ("--user", "USER", "reader@example.com"),
        ("--service-principal", "SERVICE_PRINCIPAL", "app-id"),
    ],
)
def test_cli_grant_and_revoke_use_canonical_principal_fields(target, monkeypatch, flag, kind, name):
    result = _run(target, monkeypatch, ["grant", "test-store", flag, name, "--permission", "read"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {}
    assert target.do.call_args.kwargs["body"] == _permission(name, "READ", kind)
    result = _run(target, monkeypatch, ["revoke", "test-store", flag, name, "--yes"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {}
    assert target.do.call_args == call(
        "POST",
        target.path + "/permissions:revoke",
        query=None,
        body={"principal": {"principal_type": kind, "name": name}},
    )


@pytest.mark.parametrize(
    "args",
    [
        ["grant", "test-store", "--permission", "READ"],
        ["grant", "test-store", "--user", "a", "--service-principal", "b", "--permission", "READ"],
        ["grant", "test-store", "--user", "a"],
        ["grant", "test-store", "--user", "a", "--permission", "NONE"],
        ["grant", "test-store", "--user", " ", "--permission", "READ"],
        ["revoke", "test-store", "--yes"],
        ["revoke", "test-store", "--user", "a", "--service-principal", "b", "--yes"],
        ["get", "test-store", "--user", "a"],
        ["list", "test-store", "--page-size", "1001"],
        ["list", "test-store", "--page-size", "0"],
        ["list", "test-store", "--all", "--page-token", "token"],
    ],
)
def test_cli_usage_errors_do_not_send_http_requests(target, monkeypatch, args):
    result = _run(target, monkeypatch, args)
    assert result.exit_code != 0
    target.do.assert_not_called()


def test_cli_revoke_requires_confirmation(target, monkeypatch):
    result = _run(
        target, monkeypatch, ["revoke", "test-store", "--user", "reader@example.com"], input="n\n"
    )
    assert result.exit_code != 0
    target.do.assert_not_called()
    result = _run(
        target,
        monkeypatch,
        ["revoke", "test-store", "--user", "reader@example.com"],
        input="y\n",
        output="text",
    )
    assert result.exit_code == 0, result.output
    assert "Revoked READ and WRITE" in result.output
    assert target.do.call_count == 1


@pytest.mark.parametrize("all_pages", [False, True])
def test_cli_list_pagination_json(target, monkeypatch, all_pages):
    first: dict = {target.key: [_permission()], "next_page_token": "next"}
    second = {target.key: [_permission("app-id", "WRITE", "SERVICE_PRINCIPAL")]}
    target.do.side_effect = [first, second]
    result = _run(target, monkeypatch, ["list", "test-store", *(["--all"] if all_pages else [])])
    assert result.exit_code == 0, result.output
    expected = {target.key: first[target.key] + second[target.key]} if all_pages else first
    assert json.loads(result.output) == expected
    assert target.do.call_count == (2 if all_pages else 1)
    if all_pages:
        assert target.do.call_args.kwargs["query"] == {"page_size": 25, "page_token": "next"}


def test_cli_list_piped_text_prints_next_page_hint(target, monkeypatch):
    target.do.return_value = {target.key: [_permission()], "next_page_token": "next"}
    result = _run(target, monkeypatch, ["list", "test-store"], output="text")
    assert result.exit_code == 0, result.output
    assert "reader@example.com" in result.output and "--page-token next" in result.output
    assert target.do.call_count == 1


def test_cli_list_resumes_from_token_and_accepts_max_page_size(target, monkeypatch):
    target.do.return_value = {target.key: []}
    result = _run(
        target, monkeypatch, ["list", "test-store", "--page-token", "resume", "--page-size", "1000"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {target.key: []}
    assert target.do.call_args.kwargs["query"] == {"page_size": 1000, "page_token": "resume"}


def test_cli_list_text_all_fetches_every_page_without_prompt(target, monkeypatch):
    target.do.side_effect = [
        {target.key: [_permission()], "next_page_token": "next"},
        {target.key: [_permission("app-id", "WRITE", "SERVICE_PRINCIPAL")]},
    ]
    result = _run(target, monkeypatch, ["list", "test-store", "--all"], output="text")
    assert result.exit_code == 0, result.output
    assert "reader@example.com" in result.output and "app-id" in result.output
    assert target.do.call_count == 2


@pytest.mark.parametrize("code", ["PERMISSION_DENIED", "NOT_FOUND", "NOT_IMPLEMENTED"])
@pytest.mark.parametrize("operation", ["get", "list", "grant", "revoke"])
def test_cli_preserves_backend_errors_without_retrying(target, monkeypatch, code, operation):
    target.do.side_effect = AgentCliError("backend rejected request", error_code=code)
    args = [operation, "test-store"]
    if operation in {"grant", "revoke"}:
        args += ["--user", "reader@example.com"]
        args += ["--permission", "READ"] if operation == "grant" else ["--yes"]
    result = _run(target, monkeypatch, args)
    assert result.exit_code != 0
    assert json.loads(result.output)["error"]["code"] == code
    # Memory Get preserves its existing display-name fallback on NOT_FOUND.
    if target.kind == "memory" and operation == "get" and code == "NOT_FOUND":
        assert target.do.call_count == 3
    else:
        assert target.do.call_count == 1


def test_memory_permission_get_reuses_display_name_resolution(monkeypatch):
    do = Mock(
        side_effect=[
            AgentCliError("invalid ID", error_code="INVALID_PARAMETER_VALUE"),
            AgentCliError("invalid ID", error_code="INVALID_PARAMETER_VALUE"),
            {
                "managed_memory_stores": [
                    {"name": "memory-stores/id-1", "display_name": "Friendly name"}
                ]
            },
            _permission(),
        ]
    )
    api = _AgentBricksApiClient(workspace_client=Mock(api_client=SimpleNamespace(do=do)))
    target = SimpleNamespace(kind="memory", api=api)
    result = _run(target, monkeypatch, ["get", "Friendly name"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == _permission()
    assert do.call_args.args[1] == "/api/2.0/agents/memory-stores/id-1/permissions/self"


def test_memory_permission_get_rejects_ambiguous_display_names(monkeypatch):
    do = Mock(
        side_effect=[
            AgentCliError("missing", error_code="NOT_FOUND"),
            AgentCliError("missing", error_code="NOT_FOUND"),
            {
                "managed_memory_stores": [
                    {"name": f"memory-stores/id-{i}", "display_name": "Duplicate"} for i in range(2)
                ]
            },
        ]
    )
    api = _AgentBricksApiClient(workspace_client=Mock(api_client=SimpleNamespace(do=do)))
    result = _run(SimpleNamespace(kind="memory", api=api), monkeypatch, ["get", "Duplicate"])
    assert result.exit_code != 0
    assert "2 memory stores" in result.output
    assert do.call_count == 3
