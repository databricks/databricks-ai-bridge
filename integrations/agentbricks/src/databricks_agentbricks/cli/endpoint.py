"""Thin Click entry point for invoking arbitrary HTTP endpoints."""

from __future__ import annotations

import click

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.presentation.endpoint import SsePrinter, render_response
from databricks_agentbricks.services.invoke.invoke_service import (
    InvokeRequest,
    InvokeService,
    WorkspaceOAuthAuthenticator,
)
from databricks_agentbricks.services.invoke.transport import HttpSession


def build_invoke_service(obj) -> InvokeService:
    """Compose the endpoint workflow for one CLI invocation."""
    return InvokeService(
        apps_client=AppsClient(obj.profile),
        authenticator=WorkspaceOAuthAuthenticator(obj.profile),
        transport=HttpSession(),
    )


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
    "--routing-key",
    default=None,
    help=(
        "Sticky-routing key; set it to your stable session id to keep a session on one app replica. "
        "Sent verbatim as the X-Routing-Key header (default: generated for a Databricks App). "
        "Routing only: while session_id is a natural routing key, it is recommended to pass the session "
        "id in the --json body for session continuity."
    ),
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
    routing_key,
    timeout,
    auth,
) -> None:
    """Send one HTTP request to a Databricks App or arbitrary URL."""
    service = build_invoke_service(obj)
    request = InvokeRequest(
        app=app,
        url=url,
        method=method,
        path=path,
        query=query,
        json_value=json_value,
        sse=sse,
        routing_key=routing_key,
        timeout=timeout,
        auth=auth,
    )
    printer = SsePrinter(enabled=sse and obj.output == "text")
    response = service.invoke(request, on_event=printer)
    render_response(response, output=obj.output, streamed=sse)


endpoint.add_command(invoke)
