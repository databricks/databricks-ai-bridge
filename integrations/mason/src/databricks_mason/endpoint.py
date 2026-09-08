"""CLI entry point for invoking deployed HTTP endpoints."""

from __future__ import annotations

import json
import time
from typing import Mapping
from uuid import UUID, uuid4

import click

from databricks_mason.endpoint_loadtest import loadtest
from databricks_mason.endpoint_output import StreamPrinter, render_response, success
from databricks_mason.endpoint_presets import (
    PRESET_NAMES,
    EndpointPreset,
    get_preset,
    polling_path,
    terminal_status,
)
from databricks_mason.endpoint_request import build_request, request_url, resolve_target
from databricks_mason.endpoint_transport import EndpointRequest, EndpointResponse, HttpSession
from databricks_mason.errors import AgentCliError


def _wait_for_completion(
    session: HttpSession,
    response: EndpointResponse,
    *,
    preset: EndpointPreset,
    base_url: str,
    headers: dict[str, str],
    timeout: float,
    poll_interval: float,
) -> EndpointResponse:
    if not isinstance(response.body, Mapping):
        raise AgentCliError("The background response did not contain a JSON object to poll.")
    path = polling_path(preset, response.body)
    if path is None:
        raise AgentCliError(
            "The background response did not contain an invocation id or status URL."
        )
    deadline = time.monotonic() + timeout
    current = response
    while True:
        if isinstance(current.body, Mapping) and terminal_status(current.body.get("status")):
            return current
        if not 200 <= current.status_code < 300:
            return current
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AgentCliError(f"Timed out waiting {timeout:g}s for the invocation to complete.")
        time.sleep(min(poll_interval, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AgentCliError(f"Timed out waiting {timeout:g}s for the invocation to complete.")
        current = session.send(
            EndpointRequest(
                url=request_url(base_url, path, {}),
                method="GET",
                headers=headers,
                body=None,
                timeout=min(remaining, 60.0),
            )
        )


@click.group()
def endpoint() -> None:
    """Invoke and load-test arbitrary HTTP endpoints."""


@click.command("invoke")
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
@click.option("--stream", is_flag=True, help="Request and consume an SSE response.")
@click.option("--background", is_flag=True, help="Submit a background invocation.")
@click.option("--wait", is_flag=True, help="Poll a preset background invocation to completion.")
@click.option("--id", "request_id", default=None, help="Durable invocation id (default: UUID).")
@click.option("--timeout", type=click.FloatRange(min=0.1), default=300.0, show_default=True)
@click.option("--poll-interval", type=click.FloatRange(min=0.1), default=1.0, show_default=True)
@click.option("--expect-status", type=int, multiple=True, help="Expected HTTP status (repeatable).")
@click.option("--routing-key", default=None, help="Stable Databricks Apps routing cookie value.")
@click.option("--auth/--no-auth", default=None, help="Inject Databricks OAuth authentication.")
@click.pass_obj
def invoke(
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
    wait,
    request_id,
    timeout,
    poll_interval,
    expect_status,
    routing_key,
    auth,
) -> None:
    """Send one HTTP request to a Databricks App or arbitrary URL."""
    selected_preset = get_preset(preset)
    if request_id is not None and (
        selected_preset is None or not selected_preset.client_generated_id
    ):
        raise AgentCliError("--id requires --preset mason-durable.")
    if request_id is not None:
        try:
            UUID(request_id)
        except ValueError as exc:
            raise AgentCliError("--id must be a valid UUID.") from exc
    if wait and (not background or selected_preset is None):
        raise AgentCliError("--wait requires --background and a Mason preset.")
    base_url, is_app, _ = resolve_target(
        target=target,
        url=url,
        source=source,
        profile=obj.profile,
    )
    authenticate = is_app if auth is None else auth
    routing_key = routing_key or (str(uuid4()) if is_app else None)
    request = build_request(
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
        request_id=request_id,
        timeout=timeout,
        routing_key=routing_key,
    )
    session = HttpSession()
    printer = StreamPrinter(enabled=stream and obj.output == "text")
    response = session.send(request, on_event=printer)
    printer.finish()
    if not success(response.status_code, expect_status):
        raise AgentCliError(
            f"Endpoint returned HTTP {response.status_code}.",
            hint=json.dumps(response.body, default=str)[:1000]
            if response.body is not None
            else None,
        )
    if wait:
        if selected_preset is None:
            raise AgentCliError("--wait requires a Mason preset.")
        response = _wait_for_completion(
            session,
            response,
            preset=selected_preset,
            base_url=base_url,
            headers=request.headers,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        if not success(response.status_code, expect_status):
            raise AgentCliError(f"Polling returned HTTP {response.status_code}.")
    render_response(response, output=obj.output, streamed=stream and not wait)


endpoint.add_command(invoke)
endpoint.add_command(loadtest)
