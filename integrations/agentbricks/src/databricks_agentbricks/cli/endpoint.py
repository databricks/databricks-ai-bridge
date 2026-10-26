"""Thin Click entry points for invoking and load testing HTTP endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import click

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.presentation.endpoint import SsePrinter, render_response
from databricks_agentbricks.presentation.loadtest import render_loadtest_summary
from databricks_agentbricks.services.endpoint_service import (
    EndpointService,
    EndpointTarget,
    InvokeRequest,
)
from databricks_agentbricks.services.invoke.auth import WorkspaceOAuthAuthenticator
from databricks_agentbricks.services.invoke.transport import HttpSession
from databricks_agentbricks.services.loadtest.plan import LoadTestPlan, load_bodies


def build_endpoint_service(obj) -> EndpointService:
    """Compose the endpoint workflow for one CLI invocation."""
    return EndpointService(
        apps_client=AppsClient(obj.profile),
        authenticator=WorkspaceOAuthAuthenticator(obj.profile),
        transport=HttpSession(),
    )


def request_options(command):
    """Share the HTTP request options between single requests and load tests."""
    for option in reversed(
        (
            click.argument("app", required=False, metavar="[APP]"),
            click.option(
                "--url", default=None, help="Base URL for localhost or an arbitrary HTTP server."
            ),
            click.option("--method", default="POST", show_default=True),
            click.option("--path", required=True, help="Request path, such as /api/invocations."),
            click.option(
                "--query", "query", multiple=True, help="Query parameter as 'name=value'."
            ),
            click.option("--sse", is_flag=True, help="Consume the response as Server-Sent Events."),
            click.option(
                "--routing-key",
                default=None,
                help=(
                    "Sticky-routing key; set it to your stable session id to keep a session on one app replica. "
                    "Sent verbatim as the X-Routing-Key header (default: generated for a Databricks App). "
                    "Routing only: for session continuity, also include the session "
                    "id in the request body."
                ),
            ),
            click.option(
                "--timeout", type=click.FloatRange(min=0.1), default=300.0, show_default=True
            ),
            click.option(
                "--auth/--no-auth", default=None, help="Inject Databricks OAuth authentication."
            ),
        )
    ):
        command = option(command)
    return command


@click.group()
def endpoint() -> None:
    """Invoke or load test HTTP endpoints."""


@click.command("invoke")
@request_options
@click.option("--json", "json_value", default=None, help="Complete JSON request body.")
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
    service = build_endpoint_service(obj)
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


@click.command("loadtest")
@request_options
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Concurrent users.",
)
@click.option(
    "--duration",
    type=click.IntRange(min=1),
    default=60,
    show_default=True,
    help="Stop dispatching after this many seconds or --max-requests, whichever comes first; drain in-flight requests.",
)
@click.option(
    "--max-requests",
    type=click.IntRange(min=1),
    default=None,
    help="Stop dispatching after this many requests or --duration, whichever comes first; excludes warm-up and drains in-flight requests.",
)
@click.option(
    "--requests",
    "requests_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON array of request bodies, cycled in order. One body repeats it; omit for bodyless requests.",
)
@click.option(
    "--report-dir",
    type=click.Path(path_type=Path),
    help="Write all HTML, CSV, and JSON reports together in a new or empty directory.",
)
@click.option(
    "--save-responses",
    "responses_file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save all successful inputs, outputs, and explicit trace IDs to a new private JSON file; may contain sensitive data.",
)
@click.option(
    "--warmup/--no-warmup",
    default=True,
    show_default=True,
    help="Send one excluded real request before load; excludes cold-start latency. Disable to measure cold start with the load.",
)
@click.pass_obj
def loadtest(
    obj,
    app,
    url,
    method,
    path,
    query,
    sse,
    routing_key,
    timeout,
    auth,
    concurrency,
    duration,
    max_requests,
    requests_file,
    report_dir,
    responses_file,
    warmup,
) -> None:
    """Load test a Databricks App or arbitrary URL with Locust."""
    target = EndpointTarget(
        app=app,
        url=url,
        method=method,
        path=path,
        query=query,
        sse=sse,
        routing_key=routing_key,
        timeout=timeout,
        auth=auth,
    )
    plan = LoadTestPlan(
        target=target,
        bodies=load_bodies(requests_file),
        concurrency=concurrency,
        duration=duration,
        max_requests=max_requests,
        report_dir=report_dir,
        responses_file=responses_file,
        warmup=warmup,
    )
    result = build_endpoint_service(obj).loadtest(plan)
    render_loadtest_summary(result.summary, report_dir=report_dir, output=obj.output)
    if obj.output == "json":
        click.echo(json.dumps(result.summary))
    if result.returncode:
        raise click.exceptions.Exit(result.returncode)


endpoint.add_command(invoke)
endpoint.add_command(loadtest)
