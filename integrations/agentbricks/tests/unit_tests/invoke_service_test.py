"""Local unit coverage for the framework-agnostic endpoint invocation service."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke.auth import WorkspaceOAuthAuthenticator
from databricks_agentbricks.services.invoke.transport import EndpointResponse, HttpSession
from databricks_agentbricks.services.invoke_service import InvokeRequest, InvokeService


def _response(status_code: int = 200, body=None) -> EndpointResponse:
    return EndpointResponse(
        url="http://example.test/endpoint",
        status_code=status_code,
        headers={},
        body=body,
        elapsed_seconds=0.01,
    )


def _service() -> tuple[InvokeService, Mock, Mock, Mock]:
    apps = Mock(spec=AppsClient)
    authenticator = Mock(spec=WorkspaceOAuthAuthenticator)
    authenticator.authorization_header.return_value = "Bearer test-token"
    transport = Mock(spec=HttpSession)
    transport.send.return_value = _response()
    return (
        InvokeService(apps_client=apps, authenticator=authenticator, transport=transport),
        apps,
        authenticator,
        transport,
    )


def test_explicit_url_skips_app_lookup_auth_and_generated_routing_key() -> None:
    service, apps, authenticator, transport = _service()

    service.invoke(InvokeRequest(url="http://localhost:8000/", path="/health"))

    request = transport.send.call_args.args[0]
    assert request.url == "http://localhost:8000/health"
    assert request.headers == {}
    apps.get_app_url.assert_not_called()
    authenticator.authorization_header.assert_not_called()


def test_explicit_url_honors_auth_and_routing_key() -> None:
    service, _, authenticator, transport = _service()

    service.invoke(
        InvokeRequest(
            url="http://localhost:8000",
            path="/health",
            auth=True,
            routing_key="session-42",
        )
    )

    request = transport.send.call_args.args[0]
    assert request.headers == {
        "Authorization": "Bearer test-token",
        "X-Routing-Key": "session-42",
    }
    authenticator.authorization_header.assert_called_once_with()


def test_app_and_url_are_mutually_exclusive() -> None:
    service, apps, authenticator, transport = _service()

    with pytest.raises(AgentCliError, match="APP and --url are mutually exclusive"):
        service.invoke(
            InvokeRequest(
                app="demo",
                url="http://localhost:8000",
                path="/invoke",
            )
        )

    apps.get_app_url.assert_not_called()
    authenticator.authorization_header.assert_not_called()
    transport.send.assert_not_called()


def test_missing_target_fails_before_http() -> None:
    service, apps, authenticator, transport = _service()

    with pytest.raises(AgentCliError, match="Provide a Databricks App name or --url"):
        service.invoke(InvokeRequest(path="/invoke"))

    apps.get_app_url.assert_not_called()
    authenticator.authorization_header.assert_not_called()
    transport.send.assert_not_called()


def test_unresolved_app_url_fails_before_http() -> None:
    service, apps, authenticator, transport = _service()
    apps.get_app_url.return_value = None

    with pytest.raises(AgentCliError, match="Could not resolve a URL for Databricks App"):
        service.invoke(InvokeRequest(app="demo", path="/invoke"))

    apps.get_app_url.assert_called_once_with("agent-bricks-demo")
    authenticator.authorization_header.assert_not_called()
    transport.send.assert_not_called()


def test_valid_request_propagates_http_fields() -> None:
    service, _, _, transport = _service()

    service.invoke(
        InvokeRequest(
            url="http://localhost:8000",
            method="patch",
            path="/invoke",
            query=("mode=fast", "page=2"),
            json_value='{"question":"hello"}',
            timeout=12.5,
        )
    )

    request = transport.send.call_args.args[0]
    assert request.method == "PATCH"
    assert request.body == {"question": "hello"}
    assert request.url == "http://localhost:8000/invoke?mode=fast&page=2"
    assert request.timeout == 12.5


@patch("databricks_agentbricks.services.invoke_service.uuid4")
def test_app_name_is_prefixed_and_uses_oauth_and_generated_routing_key(uuid4: Mock) -> None:
    service, apps, authenticator, transport = _service()
    apps.get_app_url.return_value = "https://agent.example/"
    uuid4.return_value = "generated-routing-key"

    service.invoke(InvokeRequest(app="demo", path="/invoke"))

    request = transport.send.call_args.args[0]
    assert request.url == "https://agent.example/invoke"
    assert request.headers == {
        "Authorization": "Bearer test-token",
        "X-Routing-Key": "generated-routing-key",
    }
    apps.get_app_url.assert_called_once_with("agent-bricks-demo")
    authenticator.authorization_header.assert_called_once_with()


@pytest.mark.parametrize(
    ("url", "app", "auth", "should_auth"),
    [
        ("http://localhost:8000", None, True, True),
        (None, "demo", False, False),
    ],
)
def test_explicit_auth_mode_overrides_target_default(
    url: str | None, app: str | None, auth: bool, should_auth: bool
) -> None:
    service, apps, authenticator, transport = _service()
    apps.get_app_url.return_value = "https://agent.example"

    service.invoke(InvokeRequest(url=url, app=app, path="/invoke", auth=auth, routing_key="fixed"))

    request = transport.send.call_args.args[0]
    assert ("Authorization" in request.headers) is should_auth
    if should_auth:
        assert request.headers["Authorization"] == "Bearer test-token"
        authenticator.authorization_header.assert_called_once_with()
    else:
        authenticator.authorization_header.assert_not_called()
    assert request.headers["X-Routing-Key"] == "fixed"


@pytest.mark.parametrize(
    ("invoke_request", "message"),
    [
        (
            InvokeRequest(url="http://localhost:8000", path="/invoke", json_value="{"),
            "Invalid JSON request body",
        ),
        (
            InvokeRequest(url="http://localhost:8000", path="https://other.example/invoke"),
            "--path must be relative",
        ),
    ],
)
def test_invalid_json_or_path_fails_before_http(
    invoke_request: InvokeRequest, message: str
) -> None:
    service, _, _, transport = _service()

    with pytest.raises(AgentCliError, match=message):
        service.invoke(invoke_request)

    transport.send.assert_not_called()


def test_non_2xx_response_raises_with_response_body_hint() -> None:
    service, _, _, transport = _service()
    transport.send.return_value = _response(503, {"error": "unavailable"})

    with pytest.raises(AgentCliError) as exc_info:
        service.invoke(InvokeRequest(url="http://localhost:8000", path="/invoke"))

    assert str(exc_info.value) == "Endpoint returned HTTP 503."
    assert exc_info.value.hint == '{"error": "unavailable"}'


def test_sse_callback_is_forwarded_to_transport() -> None:
    service, _, _, transport = _service()
    event = {"event": "message", "data": {"answer": "done"}}

    def send(_request, *, on_event=None):
        assert on_event is not None
        on_event(event)
        return _response()

    transport.send.side_effect = send
    received: list[dict] = []
    callback = received.append

    response = service.invoke(
        InvokeRequest(url="http://localhost:8000", path="/stream", sse=True),
        on_event=callback,
    )

    assert received == [event]
    assert response.status_code == 200
    assert transport.send.call_args.kwargs["on_event"] is callback
