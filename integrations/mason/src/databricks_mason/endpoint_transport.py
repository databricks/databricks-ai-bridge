"""HTTP transport and response decoding for Mason endpoint commands."""

from __future__ import annotations

import http.cookiejar
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from databricks_mason.errors import AgentCliError


@dataclass(frozen=True)
class EndpointRequest:
    """One fully materialized HTTP request."""

    url: str
    method: str
    headers: dict[str, str]
    body: Any
    timeout: float
    sse: bool = False
    body_set: bool = False


@dataclass(frozen=True)
class EndpointResponse:
    """HTTP response data used by endpoint rendering."""

    url: str
    status_code: int
    headers: dict[str, str]
    body: Any
    elapsed_seconds: float
    events: tuple[dict[str, Any], ...] = ()


class HttpSession:
    """Small stdlib HTTP client retaining cookies across polling requests."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )

    def send(
        self,
        request: EndpointRequest,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> EndpointResponse:
        data = json.dumps(request.body).encode() if request.body_set else None
        http_request = urllib.request.Request(
            request.url,
            data=data,
            headers=request.headers,
            method=request.method,
        )
        started = time.perf_counter()
        try:
            response = self._opener.open(http_request, timeout=request.timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", exc)
            raise AgentCliError(f"Could not reach endpoint {request.url}: {reason}.") from exc
        try:
            headers = dict(response.headers.items())
            content_type = response.headers.get_content_type()
            if request.sse and content_type == "text/event-stream":
                events = tuple(iter_sse(response, on_event=on_event))
                body: Any = None
            else:
                events = ()
                body = decode_body(response.read(), content_type)
            status_code = response.status
            if not isinstance(status_code, int):
                raise AgentCliError(f"Endpoint {request.url} returned no HTTP status code.")
            return EndpointResponse(
                url=request.url,
                status_code=status_code,
                headers=headers,
                body=body,
                elapsed_seconds=time.perf_counter() - started,
                events=events,
            )
        finally:
            response.close()


def iter_sse(
    response,
    *,
    on_event: Callable[[dict[str, Any]], None] | None,
) -> Iterable[dict[str, Any]]:
    """Parse an SSE byte stream and optionally emit events as they arrive."""
    event: dict[str, Any] = {}
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines or event:
                completed = _complete_event(event, data_lines)
                if on_event is not None:
                    on_event(completed)
                yield completed
            event = {}
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
        elif field in {"event", "id", "retry"}:
            event[field] = value
    if data_lines or event:
        completed = _complete_event(event, data_lines)
        if on_event is not None:
            on_event(completed)
        yield completed


def _complete_event(event: dict[str, Any], data_lines: list[str]) -> dict[str, Any]:
    value = "\n".join(data_lines)
    try:
        event["data"] = json.loads(value)
    except json.JSONDecodeError:
        event["data"] = value
    return event


def decode_body(data: bytes, content_type: str) -> Any:
    """Decode a response as JSON when possible, otherwise return text."""
    text = data.decode("utf-8", errors="replace")
    if not text:
        return None
    if content_type == "application/json" or text[:1] in {"{", "[", '"'}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text
