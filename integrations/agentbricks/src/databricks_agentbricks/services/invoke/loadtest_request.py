"""One measured invocation, optionally followed through its events and final state."""

import json
import time
from urllib.parse import urljoin, urlsplit

from databricks_agentbricks.services.invoke.loadtest_streaming import (
    StreamError,
    StreamMetrics,
    explicit_trace_id,
    output_tokens,
)


def _follow_url(base, value):
    if not isinstance(value, str) or not value:
        raise StreamError("Invocation response is missing a follow-up URL")
    try:
        url = urljoin(base, value)
        target, origin = urlsplit(url), urlsplit(base)

        def authority(parts):
            return (
                parts.scheme,
                parts.hostname,
                parts.port if parts.port is not None else (443 if parts.scheme == "https" else 80),
            )

        if (
            target.scheme not in {"http", "https"}
            or not target.hostname
            or target.username is not None
            or target.password is not None
            or target.fragment
            or authority(target) != authority(origin)
        ):
            raise ValueError
    except ValueError:
        raise StreamError("Invocation follow-up URL must use the same origin") from None
    return url


def _json(response):
    try:
        value = json.loads(response.content)
    except (ValueError, TypeError):
        raise StreamError("Invocation returned invalid JSON") from None
    if not isinstance(value, dict):
        raise StreamError("Invocation returned an invalid response object")
    return value


def perform_request(session, request, body, started, *, follow_events=False, capture_output=False):
    """Return metrics and optional sample content, without instrumenting individual HTTP calls."""
    metrics = {}

    def send(method, url, data=None):
        remaining = request["timeout"] - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("Request deadline exceeded")
        response = session.request(
            method, url, data=data, timeout=remaining, allow_redirects=False, stream=True
        )
        if not 200 <= response.status_code < 300:
            # Read the complete failed response so its latency includes the body.
            try:
                _ = response.content
            finally:
                response.close()
            raise StreamError(f"HTTP {response.status_code}")
        if capture_output:
            identifier = explicit_trace_id({"trace_id": response.headers.get("X-MLflow-Trace-ID")})
            if identifier:
                metrics["trace_id"] = identifier
        return response

    def consume(response):
        if response.headers.get("Content-Type", "").split(";", 1)[0].strip() == "text/event-stream":
            stream = StreamMetrics(capture_output=capture_output)
            stream.read(response, started, timeout=request["timeout"])
            metrics.update(ttft_ms=stream.ttft_ms, output_tokens=stream.output_tokens)
            if capture_output:
                metrics["outputs"] = stream.answer
                if stream.trace_id:
                    metrics["trace_id"] = stream.trace_id
            return stream
        content = response.content
        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            if capture_output:
                metrics["outputs"] = content.decode("utf-8", errors="replace")
            return None
        metrics["output_tokens"] = output_tokens(payload)
        if capture_output:
            metrics["outputs"] = (
                payload.get("output", payload) if isinstance(payload, dict) else payload
            )
            if identifier := explicit_trace_id(payload):
                metrics["trace_id"] = identifier
        return None

    with send(request["method"], request["url"], body) as response:
        if not follow_events:
            consume(response)
            return metrics
        if response.status_code != 202:
            raise StreamError("Background invocation must return HTTP 202")
        accepted = _json(response)
        identifier = json.loads(body)["id"]
        if accepted.get("id") != identifier:
            raise StreamError("Invocation response ID did not match the request")
        events_url = _follow_url(request["url"], accepted.get("events_url"))
        status_url = _follow_url(request["url"], accepted.get("status_url"))
        if capture_output and (trace := explicit_trace_id(accepted)):
            metrics["trace_id"] = trace

    with send("GET", events_url) as response:
        stream = consume(response)
        if stream is None or stream.terminal_event != "run.completed":
            raise StreamError("Invocation stream did not report run.completed")
    with send("GET", status_url) as response:
        final = _json(response)
        if (
            final.get("id") != identifier
            or final.get("status") != "completed"
            or "output" not in final
        ):
            raise StreamError("Invocation final state did not confirm successful completion")
        if (tokens := output_tokens(final["output"])) is not None:
            metrics["output_tokens"] = tokens
        if capture_output:
            metrics["outputs"] = final["output"]
            if trace := explicit_trace_id(final):
                metrics["trace_id"] = trace
    return metrics
