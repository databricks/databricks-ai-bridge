"""Thin Click entry point for invoking arbitrary HTTP endpoints."""

from __future__ import annotations

import click

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation.endpoint import SsePrinter, render_response
from databricks_agentbricks.services.invoke.auth import WorkspaceOAuthAuthenticator
from databricks_agentbricks.services.invoke.loadtest import run_loadtest
from databricks_agentbricks.services.invoke.transport import HttpSession
from databricks_agentbricks.services.invoke_service import InvokeRequest, InvokeService


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
@click.option(
    "--loadtest", is_flag=True, help="Repeat the request using Locust and report statistics."
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Concurrent users with --loadtest.",
)
@click.option(
    "--duration",
    type=click.IntRange(min=1),
    default=60,
    show_default=True,
    help="Load test duration in seconds with --loadtest.",
)
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
    loadtest,
    concurrency,
    duration,
) -> None:
    """Send an HTTP request, or load test a Databricks App or arbitrary URL."""
    if not loadtest and any(
        click.get_current_context().get_parameter_source(name)
        is not click.core.ParameterSource.DEFAULT
        for name in ("concurrency", "duration")
    ):
        raise AgentCliError("--concurrency and --duration require --loadtest.")
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
    if loadtest:
        returncode = run_loadtest(
            service.prepare(request),
            concurrency=concurrency,
            duration=duration,
            output=obj.output,
            per_user_routing=bool(app and not routing_key),
        )
        if returncode:
            raise click.exceptions.Exit(returncode)
        return
    printer = SsePrinter(enabled=sse and obj.output == "text")
    response = service.invoke(request, on_event=printer)
    render_response(response, output=obj.output, streamed=sse)


endpoint.add_command(invoke)
