"""Implementation of the ``mason endpoint loadtest`` command."""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import replace
from uuid import uuid4

import click

from databricks_mason.endpoint_output import loadtest_result, render_loadtest
from databricks_mason.endpoint_presets import PRESET_NAMES, build_preset_body, get_preset
from databricks_mason.endpoint_request import ROUTING_COOKIE, build_request, resolve_target
from databricks_mason.endpoint_transport import EndpointResponse, HttpSession
from databricks_mason.errors import AgentCliError


@click.command("loadtest")
@click.argument("target", required=False, metavar="[APP]")
@click.option("--url", default=None, help="Base URL instead of a Databricks App name.")
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Project directory used when APP is omitted.",
)
@click.option("--preset", type=click.Choice(PRESET_NAMES), default=None)
@click.option("--method", default="POST", show_default=True)
@click.option("--path", default=None, help="Request path; required without a preset.")
@click.option("--header", "header", multiple=True, help="HTTP header as 'Name: value'.")
@click.option("--query", "query", multiple=True, help="Query parameter as 'name=value'.")
@click.option("--json", "json_value", default=None, help="Complete JSON request body.")
@click.option(
    "--json-file",
    default=None,
    type=click.Path(dir_okay=False, allow_dash=True),
    help="Read the complete JSON body from a file, or '-' for stdin.",
)
@click.option("--message", default=None, help="User message shorthand for a Mason preset.")
@click.option("--stream", is_flag=True, help="Request and fully consume streaming responses.")
@click.option("--background", is_flag=True, help="Submit background invocations without polling.")
@click.option("--timeout", type=click.FloatRange(min=0.1), default=300.0, show_default=True)
@click.option("--expect-status", type=int, multiple=True, help="Expected HTTP status (repeatable).")
@click.option(
    "--requests", "request_count", type=click.IntRange(min=1), default=10, show_default=True
)
@click.option("--concurrency", type=click.IntRange(min=1, max=100), default=1, show_default=True)
@click.option(
    "--routing-keys",
    type=click.IntRange(min=1, max=1000),
    default=None,
    help="Number of sticky Databricks Apps sessions (default: concurrency).",
)
@click.option("--auth/--no-auth", default=None, help="Inject Databricks OAuth authentication.")
@click.pass_obj
def loadtest(
    obj,
    target,
    url,
    source,
    preset,
    method,
    path,
    header,
    query,
    json_value,
    json_file,
    message,
    stream,
    background,
    timeout,
    expect_status,
    request_count,
    concurrency,
    routing_keys,
    auth,
) -> None:
    """Repeat one generic or preset HTTP request and summarize latency."""
    selected_preset = get_preset(preset)
    base_url, is_app, _ = resolve_target(
        target=target,
        url=url,
        source=source,
        profile=obj.profile,
    )
    authenticate = is_app if auth is None else auth
    session_count = routing_keys or concurrency
    keys = [str(uuid4()) for _ in range(session_count)] if is_app else [None]
    template = build_request(
        base_url=base_url,
        profile=obj.profile,
        authenticate=authenticate,
        preset=selected_preset,
        method=method,
        path=path,
        header=header,
        query=query,
        json_value=json_value,
        json_file=json_file,
        message=message,
        stream=stream,
        background=background,
        request_id=None,
        timeout=timeout,
        routing_key=None,
        force_request_id=False,
    )

    def run(sequence: int) -> EndpointResponse:
        body = template.body
        if selected_preset is not None:
            body = build_preset_body(
                selected_preset,
                template.body,
                message=None,
                stream=stream,
                background=background,
                request_id=str(uuid4()),
                force_request_id=selected_preset.client_generated_id,
            )
        headers = dict(template.headers)
        routing_key = keys[sequence % len(keys)]
        if routing_key:
            cookie = f"{ROUTING_COOKIE}={routing_key}"
            headers["Cookie"] = f"{headers['Cookie']}; {cookie}" if "Cookie" in headers else cookie
        return HttpSession().send(replace(template, headers=headers, body=body))

    responses: list[EndpointResponse] = []
    failures: list[str] = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run, sequence) for sequence in range(request_count)]
        for future in concurrent.futures.as_completed(futures):
            try:
                responses.append(future.result())
            except Exception as exc:  # noqa: BLE001 - aggregate transport failures in the report
                failures.append(str(exc))
    result = loadtest_result(
        started=started,
        responses=responses,
        failures=failures,
        expected=expect_status,
    )
    render_loadtest(result, output=obj.output)
    if result["failed"]:
        raise AgentCliError(f"{result['failed']} load-test request(s) failed.")
