"""Thin Click entry point for invoking arbitrary HTTP endpoints."""

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
from databricks_agentbricks.services.invoke.loadtest_mlflow import LoadTestMLflowService
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
@click.option(
    "--max-requests",
    type=click.IntRange(min=1),
    default=None,
    help="Stop after this many load requests (excludes preflight) with --loadtest.",
)
@click.option(
    "--cases",
    "cases_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON array of request bodies, cycled in order with --loadtest; replaces --json.",
)
@click.option(
    "--report-dir",
    type=click.Path(path_type=Path),
    help="Save HTML, CSV, and JSON reports in a new or empty directory with --loadtest.",
)
@click.option(
    "--follow-events",
    is_flag=True,
    help="Load test background POST, GET events, and GET status as one invocation.",
)
@click.option(
    "--experiment-id", help="Existing MLflow experiment for load-test traces and evaluation."
)
@click.option(
    "--scorer",
    "scorer_names",
    multiple=True,
    help="Registered MLflow scorer name; repeat to evaluate captured load-test answers.",
)
@click.option(
    "--traces", is_flag=True, help="Read explicitly correlated MLflow traces after the load test."
)
@click.option(
    "--mlflow-max-requests",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Maximum dispatched requests sampled for MLflow traces/evaluation.",
)
@click.option(
    "--expectations",
    "expectations_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON array of MLflow expectation objects, one per load-test case.",
)
@click.option(
    "--warehouse", "warehouse_id", help="SQL warehouse ID for reading UC-backed MLflow traces."
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
    max_requests,
    cases_file,
    report_dir,
    follow_events,
    experiment_id,
    scorer_names,
    traces,
    mlflow_max_requests,
    expectations_file,
    warehouse_id,
) -> None:
    """Send an HTTP request, or load test a Databricks App or arbitrary URL."""
    if not loadtest and any(
        click.get_current_context().get_parameter_source(name)
        is not click.core.ParameterSource.DEFAULT
        for name in (
            "concurrency",
            "duration",
            "max_requests",
            "cases_file",
            "report_dir",
            "follow_events",
            "experiment_id",
            "scorer_names",
            "traces",
            "mlflow_max_requests",
            "expectations_file",
            "warehouse_id",
        )
    ):
        raise AgentCliError("Load-test options require --loadtest.")
    mlflow_enabled = bool(scorer_names or traces)
    if mlflow_enabled and not experiment_id:
        raise AgentCliError("--scorer and --traces require --experiment-id.")
    if not mlflow_enabled and (
        experiment_id
        or expectations_file
        or warehouse_id
        or click.get_current_context().get_parameter_source("mlflow_max_requests")
        is not click.core.ParameterSource.DEFAULT
    ):
        raise AgentCliError("MLflow options require --scorer or --traces.")
    if expectations_file and not scorer_names:
        raise AgentCliError("--expectations requires --scorer.")
    cases = None
    if cases_file is not None:
        if json_value is not None:
            raise AgentCliError("--cases and --json are mutually exclusive.")
        try:
            cases = json.loads(cases_file.read_text())
        except (OSError, ValueError) as exc:
            raise AgentCliError(
                "--cases must contain a valid JSON array of request bodies."
            ) from exc
        if not isinstance(cases, list) or not cases:
            raise AgentCliError("--cases must contain a nonempty JSON array of request bodies.")
    expectations = None
    if expectations_file is not None:
        try:
            expectations = json.loads(expectations_file.read_text())
        except (OSError, ValueError) as exc:
            raise AgentCliError("--expectations must contain a JSON array of objects.") from exc
        if (
            not isinstance(expectations, list)
            or len(expectations) != (len(cases) if cases is not None else 1)
            or any(not isinstance(item, dict) for item in expectations)
        ):
            raise AgentCliError("--expectations requires one object per load-test case.")
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
    if loadtest:
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
            mlflow_service=(
                LoadTestMLflowService(
                    obj.profile,
                    experiment_id,
                    scorer_names=scorer_names,
                    traces=traces,
                    max_requests=mlflow_max_requests,
                    expectations=expectations,
                    warehouse_id=warehouse_id,
                )
                if mlflow_enabled
                else None
            ),
        )
        render_loadtest_summary(result.summary, report_dir=report_dir)
        if result.returncode:
            raise click.exceptions.Exit(result.returncode)
        return
    printer = SsePrinter(enabled=sse and obj.output == "text")
    response = service.invoke(request, on_event=printer)
    render_response(response, output=obj.output, streamed=sse)


endpoint.add_command(invoke)
