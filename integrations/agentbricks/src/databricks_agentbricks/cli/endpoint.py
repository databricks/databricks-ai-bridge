"""Thin Click entry points for invoking and load testing HTTP endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import click

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation.endpoint import SsePrinter, render_response
from databricks_agentbricks.presentation.loadtest import render_loadtest_summary
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
            click.option("--json", "json_value", default=None, help="Complete JSON request body."),
            click.option("--sse", is_flag=True, help="Consume the response as Server-Sent Events."),
            click.option(
                "--routing-key",
                default=None,
                help=(
                    "Sticky-routing key; set it to your stable session id to keep a session on one app replica. "
                    "Sent verbatim as the X-Routing-Key header (default: generated for a Databricks App). "
                    "Routing only: while session_id is a natural routing key, it is recommended to pass the session "
                    "id in the --json body for session continuity."
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
    help="Stop dispatching after this many requests or --duration, whichever comes first; excludes preflight and drains in-flight requests.",
)
@click.option(
    "--requests",
    "requests_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON array of request bodies, cycled in order; replaces --json.",
)
@click.option(
    "--report-dir",
    type=click.Path(path_type=Path),
    help="Write all HTML, CSV, and JSON reports together in a new or empty directory.",
)
@click.option(
    "--follow-events",
    is_flag=True,
    help=(
        "Measure POST, GET events, and GET status as one invocation. Requires POST bodies with "
        "id, background=true, stream=true and a 202 response containing the same id, events_url, and status_url "
        "(DurableAgentServer or a compatible custom implementation)."
    ),
)
@click.option(
    "--save-responses",
    "responses_file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save sampled inputs, outputs, and explicit trace IDs to a new JSON file for later evaluation; may contain sensitive data.",
)
@click.option(
    "--sample-requests",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Sample at most this many dispatched requests for --save-responses.",
)
@click.pass_obj
def loadtest(
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
    concurrency,
    duration,
    max_requests,
    requests_file,
    report_dir,
    follow_events,
    responses_file,
    sample_requests,
) -> None:
    """Load test a Databricks App or arbitrary URL with Locust."""
    if responses_file is None and (
        click.get_current_context().get_parameter_source("sample_requests")
        is not click.core.ParameterSource.DEFAULT
    ):
        raise AgentCliError("--sample-requests requires --save-responses.")
    cases = None
    if requests_file is not None:
        if json_value is not None:
            raise AgentCliError("--requests and --json are mutually exclusive.")
        try:
            cases = json.loads(requests_file.read_text())
        except (OSError, ValueError) as exc:
            raise AgentCliError(
                "--requests must contain a valid JSON array of request bodies."
            ) from exc
        if not isinstance(cases, list) or not cases:
            raise AgentCliError("--requests must contain a nonempty JSON array of request bodies.")
    service = build_invoke_service(obj)
    request = InvokeRequest(
        app=app,
        url=url,
        method=method,
        path=path,
        query=query,
        json_value=json.dumps(cases[0]) if cases is not None else json_value,
        sse=sse,
        routing_key=routing_key,
        timeout=timeout,
        auth=auth,
    )
    result = run_loadtest(
        service.prepare(request),
        concurrency=concurrency,
        duration=duration,
        output=obj.output,
        per_user_routing=bool(app and not routing_key),
        max_requests=max_requests,
        cases=cases,
        report_dir=report_dir,
        follow_events=follow_events,
        responses_file=responses_file,
        sample_requests=sample_requests,
    )
    render_loadtest_summary(result.summary, report_dir=report_dir)
    if result.returncode:
        raise click.exceptions.Exit(result.returncode)


endpoint.add_command(invoke)
endpoint.add_command(loadtest)
