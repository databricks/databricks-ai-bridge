"""Output rendering for endpoint commands."""

from __future__ import annotations

import json
from typing import Any

import click

from databricks_mason import render
from databricks_mason.endpoint_transport import EndpointResponse


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
