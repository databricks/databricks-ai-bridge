"""OAuth discovery, DCR, and UC Connection creation tests."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk.service.catalog import ConnectionInfo, ConnectionType

from databricks_agentbricks.connection_registration import (
    HttpJsonResponse,
    _default_request,
    register_connection_via_dcr,
)
from databricks_agentbricks.errors import AgentCliError

RESOURCE_URL = "https://mcp.example.test/.well-known/oauth-protected-resource/mcp"
AUTH_SERVER = "https://auth.example.test/tenant"
AUTH_METADATA_URL = "https://auth.example.test/tenant/.well-known/oauth-authorization-server"
REGISTRATION_URL = "https://auth.example.test/register"


def _workspace(*, get_error: Exception | None = None):
    connections = Mock()
    connections.get.side_effect = get_error or NotFound("absent")
    connections.create.return_value = ConnectionInfo(
        name="github",
        full_name="main.agent_connections.github",
        connection_type=ConnectionType.HTTP,
        options={"is_mcp_connection": "true"},
    )
    return SimpleNamespace(
        config=SimpleNamespace(host="https://workspace.example.test"),
        connections=connections,
    )


def _request(method: str, url: str, json_body=None) -> HttpJsonResponse:
    if url == "https://mcp.example.test/mcp":
        return HttpJsonResponse(
            401,
            {"WWW-Authenticate": f'Bearer resource_metadata="{RESOURCE_URL}" scope="repo read"'},
            None,
        )
    if url == RESOURCE_URL:
        return HttpJsonResponse(
            200,
            {},
            {
                "authorization_servers": [AUTH_SERVER],
                "scopes_supported": ["metadata-scope"],
            },
        )
    if url == AUTH_METADATA_URL:
        return HttpJsonResponse(
            200,
            {},
            {
                "registration_endpoint": REGISTRATION_URL,
                "authorization_endpoint": "https://auth.example.test/authorize",
                "token_endpoint": "https://auth.example.test/token",
            },
        )
    if method == "POST" and url == REGISTRATION_URL:
        assert json_body == {
            "client_name": "Databricks Agent Bricks",
            "redirect_uris": ["https://workspace.example.test/login/oauth/http.html"],
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        return HttpJsonResponse(
            201,
            {},
            {"client_id": "registered-client", "client_secret": "registered-secret"},
        )
    return HttpJsonResponse(404, {}, None)


def test_default_request_identifies_agent_bricks_to_oauth_provider():
    response = SimpleNamespace(status=401, headers={}, read=lambda: b"")

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        result = _default_request("GET", "https://mcp.example.test/mcp", None)

    request = urlopen.call_args.args[0]
    assert result.status_code == 401
    assert request.get_header("User-agent") == "Databricks-Agent-Bricks"


def test_register_dcr_creates_mcp_connection_with_challenge_scope():
    workspace = _workspace()

    created = register_connection_via_dcr(
        workspace,
        fqn="main.agent_connections.github",
        url="https://mcp.example.test/mcp",
        transport="mcp",
        request=_request,
    )

    assert created.full_name == "main.agent_connections.github"
    kwargs = workspace.connections.create.call_args.kwargs
    assert kwargs["name"] == "github"
    assert kwargs["parent"] == "schemas/main.agent_connections"
    assert kwargs["connection_type"] == ConnectionType.HTTP
    assert kwargs["options"] == {
        "host": "https://mcp.example.test",
        "port": "443",
        "base_path": "/mcp",
        "oauth_credential_exchange_method": "header_and_body",
        "client_id": "registered-client",
        "client_secret": "registered-secret",
        "authorization_endpoint": "https://auth.example.test/authorize",
        "token_endpoint": "https://auth.example.test/token",
        "oauth_scope": "repo read",
    }


def test_register_http_connection_omits_mcp_marker_and_uses_metadata_scope():
    workspace = _workspace()

    def request(method: str, url: str, json_body=None) -> HttpJsonResponse:
        response = _request(method, url, json_body)
        if url == "https://mcp.example.test/mcp":
            return HttpJsonResponse(
                401,
                {"WWW-Authenticate": f'Bearer resource_metadata="{RESOURCE_URL}"'},
                None,
            )
        return response

    register_connection_via_dcr(
        workspace,
        fqn="main.agent_connections.salesforce",
        url="https://mcp.example.test/mcp",
        transport="http",
        request=request,
    )

    options = workspace.connections.create.call_args.kwargs["options"]
    assert options["oauth_scope"] == "metadata-scope"
    assert "is_mcp_connection" not in options


def test_registration_uses_well_known_fallback_in_order():
    workspace = _workspace()
    requested: list[str] = []

    def request(method: str, url: str, json_body=None) -> HttpJsonResponse:
        requested.append(url)
        if url == "https://mcp.example.test/mcp":
            return HttpJsonResponse(401, {}, None)
        if url == RESOURCE_URL:
            return HttpJsonResponse(404, {}, None)
        if url == "https://mcp.example.test/.well-known/oauth-protected-resource":
            return HttpJsonResponse(200, {}, {"authorization_servers": [AUTH_SERVER]})
        return _request(method, url, json_body)

    register_connection_via_dcr(
        workspace,
        fqn="main.agent_connections.github",
        url="https://mcp.example.test/mcp",
        transport="mcp",
        request=request,
    )

    assert requested[:3] == [
        "https://mcp.example.test/mcp",
        RESOURCE_URL,
        "https://mcp.example.test/.well-known/oauth-protected-resource",
    ]


@pytest.mark.parametrize(
    "url",
    ["", "not-a-url", "http://mcp.example.test/mcp", "https://user@mcp.example.test/mcp"],
)
def test_registration_requires_safe_https_provider_url(url):
    with pytest.raises(AgentCliError, match="HTTPS"):
        register_connection_via_dcr(
            _workspace(),
            fqn="main.agent_connections.github",
            url=url,
            transport="mcp",
            request=_request,
        )


def test_existing_connection_is_rejected_before_oauth_requests():
    workspace = _workspace(get_error=RuntimeError("unused"))
    workspace.connections.get.side_effect = None
    workspace.connections.get.return_value = ConnectionInfo(name="github")
    request = Mock()

    with pytest.raises(AgentCliError, match="already exists"):
        register_connection_via_dcr(
            workspace,
            fqn="main.agent_connections.github",
            url="https://mcp.example.test/mcp",
            transport="mcp",
            request=request,
        )

    request.assert_not_called()


def test_connection_lookup_permission_error_is_not_treated_as_absent():
    workspace = _workspace(get_error=PermissionDenied("denied"))

    with pytest.raises(AgentCliError, match="read UC Connection"):
        register_connection_via_dcr(
            workspace,
            fqn="main.agent_connections.github",
            url="https://mcp.example.test/mcp",
            transport="mcp",
            request=_request,
        )

    workspace.connections.create.assert_not_called()


def test_registration_failure_redacts_secret_response(caplog):
    sentinel = "SENTINEL-CLIENT-SECRET"

    def request(method: str, url: str, json_body=None) -> HttpJsonResponse:
        response = _request(method, url, json_body)
        if method == "POST" and url == REGISTRATION_URL:
            return HttpJsonResponse(201, {}, {"client_secret": sentinel})
        return response

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(AgentCliError, match="client id") as raised,
    ):
        register_connection_via_dcr(
            _workspace(),
            fqn="main.agent_connections.github",
            url="https://mcp.example.test/mcp",
            transport="mcp",
            request=request,
        )

    assert sentinel not in str(raised.value)
    assert sentinel not in caplog.text
