"""Framework-agnostic workflow behind ``agentbricks endpoint invoke``."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Callable
from uuid import uuid4

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.deployment.names import _prefixed_name
from databricks_agentbricks.services.invoke.auth import WorkspaceOAuthAuthenticator
from databricks_agentbricks.services.invoke.request import build_request
from databricks_agentbricks.services.invoke.transport import EndpointResponse, HttpSession

_ROUTING_KEY_HEADER = "X-Routing-Key"


@dataclass(frozen=True)
class InvokeRequest:
    """The framework-independent inputs for one ``endpoint invoke`` operation.

    ``json_value`` intentionally remains text at this boundary. ``build_request`` validates
    and decodes it immediately before creating the wire request, which lets callers distinguish an
    omitted body from an explicit JSON ``null``.
    """

    path: str
    app: str | None = None
    url: str | None = None
    method: str = "POST"
    query: tuple[str, ...] = ()
    json_value: str | None = None
    sse: bool = False
    routing_key: str | None = None
    timeout: float = 300.0
    auth: bool | None = None


@dataclass(frozen=True)
class ResolvedEndpoint:
    """The URL selected for an invocation and whether it came from a Databricks App."""

    url: str
    is_app: bool


class InvokeService:
    """Resolve, authenticate, send, and validate one endpoint invocation.

    The service owns command policy and returns the raw response facts.  Click and output
    formatting stay in the CLI/presentation layer.  Each collaborator is an object so callers can
    compose a real workspace implementation or small fakes without injecting a bare function.
    """

    def __init__(
        self,
        *,
        apps_client: AppsClient,
        authenticator: WorkspaceOAuthAuthenticator,
        transport: HttpSession,
    ) -> None:
        self._apps_client = apps_client
        self._authenticator = authenticator
        self._transport = transport

    def invoke(
        self,
        request: InvokeRequest,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> EndpointResponse:
        """Execute an invocation and raise ``AgentCliError`` for transport/status failures."""
        endpoint = self._resolve_endpoint(request.app, request.url)
        authenticate = endpoint.is_app if request.auth is None else request.auth
        routing_key = request.routing_key or (str(uuid4()) if endpoint.is_app else None)
        wire_request = build_request(
            base_url=endpoint.url,
            method=request.method,
            path=request.path,
            query=request.query,
            json_value=request.json_value,
            timeout=request.timeout,
            sse=request.sse,
        )
        wire_request = replace(
            wire_request,
            headers={
                **wire_request.headers,
                **self._platform_headers(authenticate=authenticate, routing_key=routing_key),
            },
        )
        response = self._transport.send(wire_request, on_event=on_event)
        if not 200 <= response.status_code < 300:
            raise AgentCliError(
                f"Endpoint returned HTTP {response.status_code}.",
                hint=json.dumps(response.body, default=str)[:1000]
                if response.body is not None
                else None,
            )
        return response

    def _resolve_endpoint(self, app: str | None, url: str | None) -> ResolvedEndpoint:
        if app and url:
            raise AgentCliError("APP and --url are mutually exclusive.")
        if url:
            return ResolvedEndpoint(url=url.rstrip("/"), is_app=False)
        if not app:
            raise AgentCliError(
                "Provide a Databricks App name or --url.",
                hint="Use --url http://localhost:8000 when running the agent locally.",
            )
        app_name = _prefixed_name(app)
        resolved_url = self._apps_client.get_app_url(app_name)
        if not resolved_url:
            raise AgentCliError(f"Could not resolve a URL for Databricks App {app_name!r}.")
        return ResolvedEndpoint(url=resolved_url.rstrip("/"), is_app=True)

    def _platform_headers(
        self,
        *,
        authenticate: bool,
        routing_key: str | None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        if authenticate:
            headers["Authorization"] = self._authenticator.authorization_header()
        if routing_key:
            headers[_ROUTING_KEY_HEADER] = routing_key
        return headers


__all__ = [
    "InvokeRequest",
    "InvokeService",
    "ResolvedEndpoint",
]
