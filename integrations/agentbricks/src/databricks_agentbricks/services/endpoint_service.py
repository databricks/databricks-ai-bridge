"""Framework-agnostic workflows behind ``agentbricks endpoint``."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from importlib.util import find_spec
from typing import Any, Callable
from uuid import uuid4

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.deployment.names import _prefixed_name
from databricks_agentbricks.services.invoke.auth import WorkspaceOAuthAuthenticator
from databricks_agentbricks.services.invoke.request import build_request
from databricks_agentbricks.services.invoke.transport import (
    EndpointRequest,
    EndpointResponse,
    HttpSession,
)
from databricks_agentbricks.services.loadtest.plan import LoadTestPlan
from databricks_agentbricks.services.loadtest.runner import (
    LoadtestResult,
    run_loadtest,
)

_ROUTING_KEY_HEADER = "X-Routing-Key"


@dataclass(frozen=True)
class EndpointTarget:
    """Shared HTTP target settings for single invocations and load tests."""

    path: str
    app: str | None = None
    url: str | None = None
    method: str = "POST"
    query: tuple[str, ...] = ()
    sse: bool = False
    routing_key: str | None = None
    timeout: float = 300.0
    auth: bool | None = None


@dataclass(frozen=True)
class InvokeRequest(EndpointTarget):
    """One invocation; body text distinguishes an omitted body from JSON ``null``."""

    json_value: str | None = None


@dataclass(frozen=True)
class ResolvedEndpoint:
    """The URL selected for an invocation and whether it came from a Databricks App."""

    url: str
    is_app: bool


class EndpointService:
    """Resolve and authenticate HTTP targets, then invoke or load test them.

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

    def _prepare(self, target: EndpointTarget, json_value: str | None = None) -> EndpointRequest:
        # Validate local request inputs before an Apps lookup or OAuth interaction.
        wire_request = self._local_request(target, json_value)
        endpoint = self._resolve_endpoint(target.app, target.url)
        authenticate = endpoint.is_app if target.auth is None else target.auth
        routing_key = target.routing_key or (str(uuid4()) if endpoint.is_app else None)
        return replace(
            wire_request,
            url=build_request(
                base_url=endpoint.url,
                method=target.method,
                path=target.path,
                query=target.query,
                json_value=None,
                timeout=target.timeout,
                sse=target.sse,
            ).url,
            headers={
                **wire_request.headers,
                **self._platform_headers(authenticate=authenticate, routing_key=routing_key),
            },
        )

    def invoke(
        self,
        request: InvokeRequest,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> EndpointResponse:
        """Execute an invocation and raise ``AgentCliError`` for transport/status failures."""
        response = self._transport.send(
            self._prepare(request, request.json_value), on_event=on_event
        )
        if not 200 <= response.status_code < 300:
            raise AgentCliError(
                f"Endpoint returned HTTP {response.status_code}.",
                hint=json.dumps(response.body, default=str)[:1000]
                if response.body is not None
                else None,
            )
        return response

    def loadtest(self, plan: LoadTestPlan) -> LoadtestResult:
        """Validate the local plan before resolving and authenticating its target."""
        self._local_request(plan.target)
        plan.validate()
        if find_spec("locust") is None:
            raise AgentCliError(
                "Locust is required for endpoint loadtest.",
                hint="Install the loadtest extra: pip install 'databricks-agentbricks[loadtest]'",
            )
        request = self._prepare(plan.target)
        if plan.bodies is not None:
            request = replace(
                request,
                body_set=True,
                headers={**request.headers, "Content-Type": "application/json"},
            )
        return run_loadtest(
            request,
            plan=plan,
            per_user_routing=bool(plan.target.app and not plan.target.routing_key),
        )

    @staticmethod
    def _local_request(target: EndpointTarget, json_value: str | None = None) -> EndpointRequest:
        if not math.isfinite(target.timeout) or target.timeout <= 0:
            raise AgentCliError("--timeout must be a finite positive number.")
        if target.app and target.url:
            raise AgentCliError("APP and --url are mutually exclusive.")
        if not target.app and not target.url:
            raise AgentCliError(
                "Provide a Databricks App name or --url.",
                hint="Use --url http://localhost:8000 when running the agent locally.",
            )
        return build_request(
            base_url=target.url or "https://endpoint.invalid",
            method=target.method,
            path=target.path,
            query=target.query,
            json_value=json_value,
            timeout=target.timeout,
            sse=target.sse,
        )

    def _resolve_endpoint(self, app: str | None, url: str | None) -> ResolvedEndpoint:
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
    "EndpointTarget",
    "InvokeRequest",
    "EndpointService",
    "ResolvedEndpoint",
]
