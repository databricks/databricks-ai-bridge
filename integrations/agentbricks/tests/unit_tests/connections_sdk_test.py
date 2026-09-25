from __future__ import annotations

import asyncio
import json
import math
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import MagicMock

import httpx
import pytest

from databricks_agentbricks.agent_project import AgentProject, ConnectionSpec
from databricks_agentkit.auth import connections as connections_mod
from databricks_agentkit.auth import context
from databricks_agentkit.auth.connections import (
    ConnectionError,
    ConnectionHTTPError,
    ConnectionResponse,
    ConnectionTimeoutError,
)
from databricks_agentkit.auth.context import _bind_request_auth


@pytest.fixture
def project(tmp_path, monkeypatch):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.write()
    monkeypatch.setenv("AGENTBRICKS_PROJECT_ROOT", str(tmp_path))
    return project


def _bind(
    project: AgentProject,
    *,
    transport: Literal["mcp", "http"] = "http",
    principal: Literal["user", "app"] = "user",
):
    spec = ConnectionSpec(
        name="provider",
        uc_connection="main.connections.provider",
        transport=transport,
        principal=principal,
    )
    project.add_connection(spec)
    project.write()
    return spec


def _raw_response(
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    content: bytes = b'{"marker":"ok"}',
):
    return SimpleNamespace(
        status_code=status_code,
        headers=headers or {"content-type": "application/json", "x-provider": "fixture"},
        content=content,
    )


def _record_transport(monkeypatch, response=None):
    calls = []

    def perform(workspace, **kwargs):
        calls.append((workspace, kwargs))
        return response or _raw_response()

    monkeypatch.setattr(connections_mod, "_perform_request", perform)
    return calls


def _request_auth(workspace=None):
    request_auth = MagicMock()
    request_auth.client_for.return_value = workspace or SimpleNamespace(name="user-workspace")
    return request_auth


def test_lookup_is_lazy_and_unknown_alias_is_credential_free(project):
    _bind(project)
    request_auth = _request_auth()

    with _bind_request_auth(request_auth):
        client = context.connections.client("provider")

    assert client.alias == "provider"
    request_auth.client_for.assert_not_called()
    with pytest.raises(ConnectionError, match="not declared") as caught:
        context.connections.client("missing")
    assert "agent.toml" not in repr(caught.value)


@pytest.mark.asyncio
async def test_http_request_uses_user_client_and_uc_proxy(project, monkeypatch):
    _bind(project)
    calls = _record_transport(monkeypatch)
    request_auth = _request_auth()

    with _bind_request_auth(request_auth):
        response = await context.connections.client("provider").request(
            "get",
            "/accounts",
            params={"query": "acme"},
            json={"limit": 5},
            headers={"Accept": "application/json", "X-Provider-Feature": "search"},
            timeout=1.5,
        )

    assert response.json() == {"marker": "ok"}
    request_auth.client_for.assert_called_once_with("user")
    workspace, call = calls[0]
    assert workspace is request_auth.client_for.return_value
    assert call == {
        "method": "GET",
        "path": "/api/2.0/unity-catalog/connections/main.connections.provider/proxy/accounts",
        "params": {"query": "acme"},
        "json_body": {"limit": 5},
        "headers": {"Accept": "application/json", "X-Provider-Feature": "search"},
        "timeout": 1.5,
    }


@pytest.mark.asyncio
async def test_http_empty_path_uses_proxy_root(project, monkeypatch):
    _bind(project)
    calls = _record_transport(monkeypatch)
    request_auth = _request_auth()

    with _bind_request_auth(request_auth):
        await context.connections.client("provider").request("GET", "")

    assert calls[0][1]["path"] == (
        "/api/2.0/unity-catalog/connections/main.connections.provider/proxy"
    )


@pytest.mark.asyncio
async def test_app_principal_resolves_ambient_workspace_at_request_time(project, monkeypatch):
    _bind(project, principal="app")
    ambient = SimpleNamespace(name="app-workspace")
    workspace_client = MagicMock(return_value=ambient)
    monkeypatch.setattr(connections_mod, "workspace_client", workspace_client)
    calls = _record_transport(monkeypatch)

    client = context.connections.client("provider")
    workspace_client.assert_not_called()
    await client.request("GET", "/accounts")

    workspace_client.assert_called_once_with()
    assert calls[0][0] is ambient


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "GET", "DELETE"])
@pytest.mark.parametrize("path", ["", "/"])
async def test_mcp_streamable_http_methods_route_to_uc_connection_proxy(
    project, monkeypatch, method, path
):
    _bind(project, transport="mcp")
    calls = _record_transport(monkeypatch)
    request_auth = _request_auth()

    with _bind_request_auth(request_auth):
        response = await context.connections.client("provider").request(
            method,
            path,
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        )

    assert response.status_code == 200
    assert calls[0][1]["method"] == method
    assert calls[0][1]["path"] == (
        "/api/2.0/unity-catalog/connections/main.connections.provider/proxy"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "method", "path", "match"),
    [
        ("mcp", "PUT", "", "method"),
        ("mcp", "POST", "/tools", "path"),
        ("http", "CONNECT", "/accounts", "method"),
        ("http", "GET\r\nX-Evil: true", "/accounts", "method"),
    ],
)
async def test_invalid_method_or_mcp_path_fails_before_transport(
    project, monkeypatch, transport, method, path, match
):
    _bind(project, transport=transport)
    calls = _record_transport(monkeypatch)

    with _bind_request_auth(_request_auth()):
        with pytest.raises(ConnectionError, match=match):
            await context.connections.client("provider").request(method, path)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "https://evil.example/accounts",
        "//evil.example/accounts",
        "/accounts?token=sentinel",
        "/accounts#fragment",
        "/../admin",
        "/safe/../../admin",
        "/%2e%2e/admin",
        "/%252e%252e/admin",
        "/safe\\..\\admin",
        "/accounts\x00",
    ],
)
async def test_http_path_security_fails_before_transport(project, monkeypatch, path):
    _bind(project)
    calls = _record_transport(monkeypatch)

    with _bind_request_auth(_request_auth()):
        with pytest.raises(ConnectionError, match="path") as caught:
            await context.connections.client("provider").request("GET", path)
    assert calls == []
    assert "sentinel" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header",
    [
        "Authorization",
        "authorization",
        "Proxy-Authorization",
        "Cookie",
        "Host",
        "X-Databricks-Org-Id",
        "x-databricks-anything",
    ],
)
async def test_security_sensitive_headers_fail_before_transport(project, monkeypatch, header):
    _bind(project)
    calls = _record_transport(monkeypatch)

    with _bind_request_auth(_request_auth()):
        with pytest.raises(ConnectionError, match="header") as caught:
            await context.connections.client("provider").request(
                "GET", "/accounts", headers={header: "SENTINEL-TOKEN"}
            )
    assert calls == []
    assert "SENTINEL-TOKEN" not in str(caught.value)
    assert "SENTINEL-TOKEN" not in repr(caught.value)


def test_response_exposes_immutable_safe_body_helpers():
    response = ConnectionResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        content=b'{"snowman":"\xe2\x98\x83"}',
    )

    assert response.headers == {"content-type": "application/json"}
    assert isinstance(response.headers, MappingProxyType)
    with pytest.raises(TypeError):
        response.headers["x-new"] = "blocked"  # type: ignore[index]
    assert response.text == '{"snowman":"☃"}'
    assert response.json() == {"snowman": "☃"}
    response.raise_for_status()


def test_response_preserves_non_json_and_non_success_status():
    response = ConnectionResponse(
        status_code=429,
        headers={"content-type": "text/plain", "retry-after": "3"},
        content=b"provider throttled",
    )

    assert response.text == "provider throttled"
    with pytest.raises(json.JSONDecodeError):
        response.json()
    with pytest.raises(ConnectionHTTPError, match="429") as caught:
        response.raise_for_status()
    assert caught.value.status_code == 429
    assert "provider throttled" not in str(caught.value)


@pytest.mark.asyncio
async def test_raw_non_success_response_remains_inspectable(project, monkeypatch):
    _bind(project)
    _record_transport(
        monkeypatch,
        _raw_response(status_code=403, content=b'{"error":"consent required"}'),
    )

    with _bind_request_auth(_request_auth()):
        response = await context.connections.client("provider").request("GET", "/accounts")

    assert response.status_code == 403
    assert response.json() == {"error": "consent required"}


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, math.inf, math.nan])
async def test_invalid_timeout_fails_before_transport(project, monkeypatch, timeout):
    _bind(project)
    calls = _record_transport(monkeypatch)

    with _bind_request_auth(_request_auth()):
        with pytest.raises(ConnectionError, match="timeout"):
            await context.connections.client("provider").request(
                "GET", "/accounts", timeout=timeout
            )
    assert calls == []


@pytest.mark.asyncio
async def test_timeout_is_typed_and_credential_free(project, monkeypatch):
    _bind(project)

    def slow_transport(workspace, **kwargs):
        time.sleep(0.05)
        return _raw_response(content=b"SENTINEL-RESPONSE")

    monkeypatch.setattr(connections_mod, "_perform_request", slow_transport)
    with _bind_request_auth(_request_auth()):
        with pytest.raises(ConnectionTimeoutError, match="timed out") as caught:
            await context.connections.client("provider").request("GET", "/accounts", timeout=0.001)
    assert "SENTINEL-RESPONSE" not in repr(caught.value)


@pytest.mark.asyncio
async def test_transport_error_is_credential_free(project, monkeypatch):
    _bind(project)

    def broken_transport(workspace, **kwargs):
        raise RuntimeError("SENTINEL-UPSTREAM-ERROR")

    monkeypatch.setattr(connections_mod, "_perform_request", broken_transport)
    with _bind_request_auth(_request_auth()):
        with pytest.raises(ConnectionError, match="proxy request failed") as caught:
            await context.connections.client("provider").request("GET", "/accounts")
    assert "SENTINEL-UPSTREAM-ERROR" not in str(caught.value)
    assert "SENTINEL-UPSTREAM-ERROR" not in repr(caught.value)


@pytest.mark.asyncio
async def test_user_connection_requires_active_context_and_resets_after_exit(project, monkeypatch):
    _bind(project)
    calls = _record_transport(monkeypatch)
    client = context.connections.client("provider")

    with pytest.raises(ConnectionError, match="active request"):
        await client.request("GET", "/accounts")
    request_auth = _request_auth()
    with pytest.raises(RuntimeError, match="handler failed"):
        with _bind_request_auth(request_auth):
            raise RuntimeError("handler failed")
    with pytest.raises(ConnectionError, match="active request"):
        await client.request("GET", "/accounts")
    assert calls == []


@pytest.mark.asyncio
async def test_request_context_is_task_local(project, monkeypatch):
    _bind(project)
    calls = _record_transport(monkeypatch)
    first = _request_auth(SimpleNamespace(name="first"))
    second = _request_auth(SimpleNamespace(name="second"))

    async def invoke(request_auth):
        with _bind_request_auth(request_auth):
            await asyncio.sleep(0)
            await context.connections.client("provider").request("GET", "/accounts")

    await asyncio.gather(invoke(first), invoke(second))

    assert {workspace.name for workspace, _ in calls} == {"first", "second"}
    first.client_for.assert_called_once_with("user")
    second.client_for.assert_called_once_with("user")


@pytest.mark.asyncio
async def test_closed_request_auth_fails_before_proxy(project, monkeypatch):
    _bind(project)
    calls = _record_transport(monkeypatch)
    request_auth = _request_auth()
    request_auth.client_for.side_effect = RuntimeError("request-user authentication is closed")

    with _bind_request_auth(request_auth):
        with pytest.raises(ConnectionError, match="authentication failed") as caught:
            await context.connections.client("provider").request("GET", "/accounts")
    assert calls == []
    assert "closed" not in str(caught.value)


def test_raw_transport_uses_workspace_auth_without_exposing_it(monkeypatch, caplog):
    config = SimpleNamespace(
        host="https://workspace.example",
        authenticate=MagicMock(return_value={"Authorization": "Bearer SENTINEL-TOKEN"}),
        _custom_headers={"X-Databricks-Org-Id": "123"},
        http_timeout_seconds=45,
    )
    workspace = SimpleNamespace(config=config)
    request = MagicMock(
        return_value=httpx.Response(
            201,
            headers={"content-type": "application/json"},
            content=b'{"created":true}',
        )
    )
    monkeypatch.setattr(connections_mod.httpx, "request", request)

    raw = connections_mod._perform_request(
        cast(Any, workspace),
        method="POST",
        path="/api/2.0/unity-catalog/connections/main.connections.provider/proxy/accounts",
        params={"dry_run": True},
        json_body={"name": "Acme"},
        headers={"Accept": "application/json"},
        timeout=None,
    )

    assert raw.status_code == 201
    assert raw.content == b'{"created":true}'
    request.assert_called_once_with(
        "POST",
        "https://workspace.example/api/2.0/unity-catalog/connections/main.connections.provider/proxy/accounts",
        params={"dry_run": True},
        json={"name": "Acme"},
        headers={
            "Accept": "application/json",
            "X-Databricks-Org-Id": "123",
            "Authorization": "Bearer SENTINEL-TOKEN",
        },
        timeout=45,
        follow_redirects=False,
    )
    assert "SENTINEL-TOKEN" not in caplog.text
