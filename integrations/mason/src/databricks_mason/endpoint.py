"""CLI entry point for invoking arbitrary HTTP endpoints."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Optional
from uuid import uuid4

import click

from databricks_mason._api_client import _workspace_client
from databricks_mason.deploy import _app_url, _prefixed_name
from databricks_mason.endpoint_output import SsePrinter, render_response
from databricks_mason.endpoint_request import build_request
from databricks_mason.endpoint_transport import HttpSession
from databricks_mason.errors import AgentCliError

_ROUTING_COOKIE = "__Host-databricks-app-router"


def _resolve_endpoint(
    app: str | None,
    url: str | None,
    profile: Optional[str],
) -> tuple[str, bool]:
    if app and url:
        raise AgentCliError("APP and --url are mutually exclusive.")
    if url:
        return url.rstrip("/"), False
    if not app:
        raise AgentCliError(
            "Provide a Databricks App name or --url.",
            hint="Use --url http://localhost:8000 when running the agent locally.",
        )
    app_name = _prefixed_name(app)
    resolved_url = _app_url(app_name, profile)
    if not resolved_url:
        raise AgentCliError(f"Could not resolve a URL for Databricks App {app_name!r}.")
    return resolved_url.rstrip("/"), True


def _authorization_header(profile: Optional[str]) -> str:
    try:
        client = _workspace_client(profile)
        if client.config.auth_type == "pat":
            raise AgentCliError(
                "Databricks Apps API routes require OAuth; the selected profile uses a PAT.",
                hint="Authenticate the same workspace with `databricks auth login`.",
            )
        authorization = client.config.authenticate().get("Authorization")
    except AgentCliError:
        raise
    except Exception as exc:  # noqa: BLE001 - render auth failures without a traceback
        raise AgentCliError(f"Could not initialize endpoint authentication: {exc}.") from exc
    if not authorization:
        raise AgentCliError("Could not resolve an OAuth access token for the endpoint request.")
    return authorization


def _platform_headers(
    *,
    authenticate: bool,
    profile: Optional[str],
    session_id: str | None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if authenticate:
        headers["Authorization"] = _authorization_header(profile)
    if session_id:
        headers["Cookie"] = f"{_ROUTING_COOKIE}={session_id}"
    return headers


@click.group()
def endpoint() -> None:
    """Invoke arbitrary HTTP endpoints."""


@click.command("invoke")
@click.argument("app", required=False, metavar="[APP]")
@click.option("--url", default=None, help="Base URL for localhost or an arbitrary HTTP server.")
@click.option("--method", default="POST", show_default=True)
@click.option("--path", required=True, help="Request path, such as /api/invocations.")
@click.option("--query", "query", multiple=True, help="Query parameter as 'name=value'.")
@click.option("--json", "json_value", default=None, help="Complete JSON request body.")
@click.option("--sse", is_flag=True, help="Consume the response as Server-Sent Events.")
@click.option(
    "--session-id",
    default=None,
    help="Application session id (default: generated for a Databricks App).",
)
@click.option("--timeout", type=click.FloatRange(min=0.1), default=300.0, show_default=True)
@click.option("--auth/--no-auth", default=None, help="Inject Databricks OAuth authentication.")
@click.pass_obj
def invoke(
    obj,
    app,
    url,
    method,
    path,
    query,
    json_value,
    sse,
    session_id,
    timeout,
    auth,
) -> None:
    """Send one HTTP request to a Databricks App or arbitrary URL."""
    base_url, is_app = _resolve_endpoint(app, url, obj.profile)
    authenticate = is_app if auth is None else auth
    session_id = session_id or (str(uuid4()) if is_app else None)
    request = build_request(
        base_url=base_url,
        method=method,
        path=path,
        query=query,
        json_value=json_value,
        timeout=timeout,
        sse=sse,
    )
    request = replace(
        request,
        headers={
            **request.headers,
            **_platform_headers(
                authenticate=authenticate,
                profile=obj.profile,
                session_id=session_id,
            ),
        },
    )
    printer = SsePrinter(enabled=sse and obj.output == "text")
    response = HttpSession().send(request, on_event=printer)
    if not 200 <= response.status_code < 300:
        raise AgentCliError(
            f"Endpoint returned HTTP {response.status_code}.",
            hint=json.dumps(response.body, default=str)[:1000]
            if response.body is not None
            else None,
        )
    render_response(response, output=obj.output, streamed=sse)


endpoint.add_command(invoke)
