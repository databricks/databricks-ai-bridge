"""Output rendering and copy-pasteable examples for endpoint commands."""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

import click

from databricks_agentbricks.presentation import render

if TYPE_CHECKING:
    from databricks_agentbricks.services.invoke.transport import EndpointResponse


def render_response(response: EndpointResponse, *, output: str, streamed: bool) -> None:
    """Render an endpoint response in JSON or human-readable form."""
    if output == "json":
        render.emit_json(_response_payload(response))
        return
    if streamed:
        return
    if isinstance(response.body, (dict, list)):
        render.emit_json(response.body)
    elif response.body is not None:
        click.echo(str(response.body))


def _response_payload(response: EndpointResponse) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": response.url,
        "status_code": response.status_code,
        "elapsed_seconds": round(response.elapsed_seconds, 6),
    }
    if response.events:
        payload["events"] = list(response.events)
    else:
        payload["body"] = response.body
    return payload


class SsePrinter:
    """Print SSE data values as they arrive."""

    def __init__(self, *, enabled: bool):
        self.enabled = enabled

    def __call__(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        data = event.get("data")
        if isinstance(data, str):
            click.echo(data)
        else:
            click.echo(json.dumps(data, default=str))


def agent_invoke_command(target: str, *, uses_runtime_api: bool) -> str:
    """Build a single-line example with fresh runtime identifiers on every run."""
    path = "/api/invocations" if uses_runtime_api else "/invocations"
    body = '{"input":[{"role":"user","content":"hi"}]}'
    if uses_runtime_api:
        # Double quotes allow command substitution while preserving the JSON's literal quotes.
        body = '{"id":"$(uuidgen)","session_id":"$(uuidgen)",' + body[1:]
        json_arg = '"' + body.replace('"', r"\"") + '"'
    else:
        json_arg = shlex.quote(body)
    return f"agentbricks endpoint invoke {target} --path {path} --json {json_arg}"


def agent_invoke_step(target: str, *, uses_runtime_api: bool) -> tuple[str, str]:
    """The ``(command, description)`` next step shown last in the dev and deploy success panels."""
    return (
        agent_invoke_command(target, uses_runtime_api=uses_runtime_api),
        "Invoke with Agent Bricks",
    )
