"""Make one HTTP request, consuming its complete response before returning."""

import json
import time
from uuid import uuid4

from databricks_agentbricks.services.loadtest.streaming import (
    StreamError,
    StreamMetrics,
    explicit_trace_id,
    output_tokens,
)


def materialize_body(template: str | None, user_id: str) -> str | None:
    """Give each invocation a fresh request ID while retaining the virtual user's identity."""
    if template is None:
        return None
    request_id = str(uuid4())
    return (
        template.replace("{{request_id}}", request_id)
        .replace("{{uuid}}", request_id)
        .replace("{{user_id}}", user_id)
    )


def perform_request(session, request, body, started, *, capture_output=False):
    """Return timing, explicit usage, and optional content for one ordinary HTTP call."""
    remaining = request["timeout"] - (time.perf_counter() - started)
    if remaining <= 0:
        raise TimeoutError("Request deadline exceeded")
    with session.request(
        request["method"],
        request["url"],
        data=body,
        timeout=remaining,
        allow_redirects=False,
        stream=True,
    ) as response:
        if not 200 <= response.status_code < 300:
            # Include the complete error response in latency without retaining its contents.
            _ = response.content
            raise StreamError(f"HTTP {response.status_code}")
        metrics = {}
        if capture_output:
            identifier = explicit_trace_id({"trace_id": response.headers.get("X-MLflow-Trace-ID")})
            if identifier:
                metrics["trace_id"] = identifier
        if response.headers.get("Content-Type", "").split(";", 1)[0].strip() == "text/event-stream":
            stream = StreamMetrics(capture_output=capture_output)
            stream.read(response, started, timeout=request["timeout"])
            metrics.update(ttft_ms=stream.ttft_ms, output_tokens=stream.output_tokens)
            if capture_output:
                metrics["outputs"] = stream.answer
                if stream.trace_id:
                    metrics["trace_id"] = stream.trace_id
            return metrics
        content = response.content
        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            if capture_output:
                metrics["outputs"] = content.decode("utf-8", errors="replace")
            return metrics
        metrics["output_tokens"] = output_tokens(payload)
        if capture_output:
            metrics["outputs"] = (
                payload.get("output", payload) if isinstance(payload, dict) else payload
            )
            if identifier := explicit_trace_id(payload):
                metrics["trace_id"] = identifier
        return metrics
