"""Output rendering and load-test summaries for endpoint commands."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from typing import Any

import click

from databricks_mason import render
from databricks_mason.endpoint_transport import EndpointResponse


def success(status_code: int, expected: tuple[int, ...]) -> bool:
    """Return whether a status matches explicit or default success criteria."""
    return status_code in expected if expected else 200 <= status_code < 300


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


class StreamPrinter:
    """Print text deltas while retaining non-delta SSE events."""

    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self._printed_delta = False

    def __call__(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        data = event.get("data")
        if isinstance(data, dict) and data.get("type") == "delta" and data.get("content"):
            click.echo(str(data["content"]), nl=False)
            self._printed_delta = True
            return
        if data == "[DONE]":
            if self._printed_delta:
                click.echo()
                self._printed_delta = False
            return
        if self._printed_delta:
            click.echo()
            self._printed_delta = False
        click.echo(data if isinstance(data, str) else json.dumps(data, default=str))

    def finish(self) -> None:
        if self.enabled and self._printed_delta:
            click.echo()


def loadtest_result(
    *,
    started: float,
    responses: list[EndpointResponse],
    failures: list[str],
    expected: tuple[int, ...],
) -> dict[str, Any]:
    """Aggregate load-test responses into stable summary metrics."""
    elapsed = time.perf_counter() - started
    successful = [response for response in responses if success(response.status_code, expected)]
    latencies = [response.elapsed_seconds for response in responses]
    status_counts = Counter(str(response.status_code) for response in responses)
    return {
        "requests": len(responses) + len(failures),
        "successful": len(successful),
        "failed": len(responses) - len(successful) + len(failures),
        "elapsed_seconds": round(elapsed, 6),
        "requests_per_second": round((len(responses) + len(failures)) / elapsed, 3)
        if elapsed
        else None,
        "latency_seconds": {
            "mean": round(statistics.fmean(latencies), 6) if latencies else None,
            "p50": round(_percentile(latencies, 0.50) or 0, 6) if latencies else None,
            "p95": round(_percentile(latencies, 0.95) or 0, 6) if latencies else None,
            "p99": round(_percentile(latencies, 0.99) or 0, 6) if latencies else None,
            "max": round(max(latencies), 6) if latencies else None,
        },
        "status_codes": dict(sorted(status_counts.items())),
        "errors": failures[:10],
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return sorted(values)[index]


def render_loadtest(result: dict[str, Any], *, output: str) -> None:
    """Render a load-test result in JSON or a compact detail panel."""
    if output == "json":
        render.emit_json(result)
        return
    latency = result["latency_seconds"]
    render.detail(
        "Endpoint Load Test",
        f"{result['requests']} requests",
        {
            "Successful": result["successful"],
            "Failed": result["failed"],
            "Throughput": f"{result['requests_per_second']} req/s",
            "Mean latency": f"{latency['mean']}s",
            "P50 latency": f"{latency['p50']}s",
            "P95 latency": f"{latency['p95']}s",
            "P99 latency": f"{latency['p99']}s",
            "Status codes": result["status_codes"],
        },
    )
    for error in result["errors"]:
        click.echo(f"Error sample: {error}", err=True)
