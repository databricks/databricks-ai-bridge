"""Generic HTTP invocation and lightweight load testing for deployed endpoints."""

from __future__ import annotations

import concurrent.futures
import http.cookiejar
import json
import math
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Optional
from uuid import uuid4

import click

from databricks_mason import render
from databricks_mason._api_client import _workspace_client
from databricks_mason.agent_project import AgentProject
from databricks_mason.deploy import _app_url, _prefixed_name
from databricks_mason.endpoint_presets import (
    PRESET_NAMES,
    EndpointPreset,
    build_preset_body,
    get_preset,
    polling_path,
    terminal_status,
)
from databricks_mason.errors import AgentCliError

_ROUTING_COOKIE = "__Host-databricks-app-router"


@dataclass(frozen=True)
class EndpointRequest:
    """One fully materialized HTTP request."""

    url: str
    method: str
    headers: dict[str, str]
    body: Any
    timeout: float
    stream: bool = False


@dataclass(frozen=True)
class EndpointResponse:
    """HTTP response data used by invoke and load-test rendering."""

    url: str
    status_code: int
    headers: dict[str, str]
    body: Any
    elapsed_seconds: float
    events: tuple[dict[str, Any], ...] = ()


class _HttpSession:
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
        data = None if request.body is None else json.dumps(request.body).encode()
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
            if request.stream and content_type == "text/event-stream":
                events = tuple(_iter_sse(response, on_event=on_event))
                body: Any = None
            else:
                events = ()
                body = _decode_body(response.read(), content_type)
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


def _iter_sse(
    response,
    *,
    on_event: Callable[[dict[str, Any]], None] | None,
) -> Iterable[dict[str, Any]]:
    event: dict[str, Any] = {}
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines or event:
                value = "\n".join(data_lines)
                try:
                    event["data"] = json.loads(value)
                except json.JSONDecodeError:
                    event["data"] = value
                completed = event
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
        value = "\n".join(data_lines)
        try:
            event["data"] = json.loads(value)
        except json.JSONDecodeError:
            event["data"] = value
        if on_event is not None:
            on_event(event)
        yield event


def _decode_body(data: bytes, content_type: str) -> Any:
    text = data.decode("utf-8", errors="replace")
    if not text:
        return None
    if content_type == "application/json" or text[:1] in {"{", "[", '"'}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def _parse_pairs(values: tuple[str, ...], *, separator: str, description: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, found, item = value.partition(separator)
        key = key.strip()
        if not found or not key or "\n" in key or "\n" in item:
            raise AgentCliError(f"Invalid {description} {value!r}.")
        parsed[key] = item.strip()
    return parsed


def _load_json(json_value: str | None, json_file: str | None) -> Any:
    if json_value is not None and json_file is not None:
        raise AgentCliError("--json and --json-file are mutually exclusive.")
    if json_file is not None:
        try:
            text = (
                sys.stdin.read()
                if json_file == "-"
                else pathlib.Path(json_file).read_text(encoding="utf-8")
            )
        except OSError as exc:
            raise AgentCliError(f"Could not read JSON file {json_file!r}: {exc}.") from exc
    else:
        text = json_value
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentCliError(f"Invalid JSON request body: {exc}.") from exc


def _resolve_target(
    *,
    target: str | None,
    url: str | None,
    source: str,
    profile: Optional[str],
) -> tuple[str, bool, str | None]:
    if target and url:
        raise AgentCliError("APP and --url are mutually exclusive.")
    if url:
        return url.rstrip("/"), False, None
    app_name = target
    if app_name is None:
        project = AgentProject.load(source)
        if not project.deployment_name:
            raise AgentCliError(
                "No APP was given and agent.toml has no recorded deployment name.",
                hint="Pass an App name, or run `mason deploy <name>` once from this project.",
            )
        app_name = _prefixed_name(str(project.deployment_name))
    app_url = _app_url(app_name, profile)
    if not app_url:
        raise AgentCliError(f"Could not resolve a URL for Databricks App {app_name!r}.")
    return app_url.rstrip("/"), True, app_name


def _auth_headers(profile: Optional[str]) -> dict[str, str]:
    try:
        client = _workspace_client(profile)
        if client.config.auth_type == "pat":
            raise AgentCliError(
                "Databricks Apps API routes require OAuth; the selected profile uses a PAT.",
                hint="Authenticate the same workspace with `databricks auth login`.",
            )
        authorization = client.config.authenticate().get("Authorization")
    except AgentCliError:
        raise
    except Exception as exc:  # noqa: BLE001 - render auth failures without a traceback
        raise AgentCliError(f"Could not initialize endpoint authentication: {exc}.") from exc
    if not authorization:
        raise AgentCliError("Could not resolve an OAuth access token for the endpoint request.")
    return {"Authorization": authorization}


def _request_url(base_url: str, path: str, query: Mapping[str, str]) -> str:
    if not path:
        raise AgentCliError("--path is required when no --preset is selected.")
    joined = urllib.parse.urljoin(f"{base_url}/", path)
    if not query:
        return joined
    parsed = urllib.parse.urlsplit(joined)
    merged_query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    merged_query.update(query)
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(merged_query)))


def _headers(
    values: tuple[str, ...],
    *,
    authenticate: bool,
    profile: Optional[str],
    routing_key: str | None,
) -> dict[str, str]:
    headers = _parse_pairs(values, separator=":", description="header")
    if authenticate:
        for key, value in _auth_headers(profile).items():
            headers.setdefault(key, value)
    headers.setdefault("Content-Type", "application/json")
    if routing_key:
        cookie = f"{_ROUTING_COOKIE}={routing_key}"
        headers["Cookie"] = f"{headers['Cookie']}; {cookie}" if "Cookie" in headers else cookie
    return headers


def _build_request(
    *,
    base_url: str,
    profile: Optional[str],
    authenticate: bool,
    preset: EndpointPreset | None,
    method: str,
    path: str | None,
    header: tuple[str, ...],
    query: tuple[str, ...],
    json_value: str | None,
    json_file: str | None,
    message: str | None,
    stream: bool,
    background: bool,
    request_id: str | None,
    timeout: float,
    routing_key: str | None,
    force_request_id: bool = False,
) -> EndpointRequest:
    body = _load_json(json_value, json_file)
    if preset is None:
        if message is not None:
            raise AgentCliError("--message requires --preset mason or --preset mason-durable.")
        if background:
            raise AgentCliError(
                "--background requires a preset; generic mode sends the JSON as-is."
            )
    else:
        body = build_preset_body(
            preset,
            body,
            message=message,
            stream=stream,
            background=background,
            request_id=request_id,
            force_request_id=force_request_id,
        )
    request_path = path or (preset.path if preset else "")
    request_query = _parse_pairs(query, separator="=", description="query parameter")
    return EndpointRequest(
        url=_request_url(base_url, request_path, request_query),
        method=method.upper(),
        headers=_headers(
            header,
            authenticate=authenticate,
            profile=profile,
            routing_key=routing_key,
        ),
        body=body,
        timeout=timeout,
        stream=stream,
    )


def _success(status_code: int, expected: tuple[int, ...]) -> bool:
    return status_code in expected if expected else 200 <= status_code < 300


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


def _render_response(response: EndpointResponse, *, output: str, streamed: bool) -> None:
    if output == "json":
        render.emit_json(_response_payload(response))
        return
    if streamed:
        return
    if isinstance(response.body, (dict, list)):
        render.emit_json(response.body)
    elif response.body is not None:
        click.echo(str(response.body))


class _StreamPrinter:
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


def _wait_for_completion(
    session: _HttpSession,
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
                url=_request_url(base_url, path, {}),
                method="GET",
                headers=headers,
                body=None,
                timeout=min(remaining, 60.0),
            )
        )


@click.group()
def endpoint() -> None:
    """Invoke and load-test arbitrary HTTP endpoints."""


@endpoint.command("invoke")
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
    if wait and (not background or selected_preset is None):
        raise AgentCliError("--wait requires --background and a Mason preset.")
    base_url, is_app, _ = _resolve_target(
        target=target,
        url=url,
        source=source,
        profile=obj.profile,
    )
    authenticate = is_app if auth is None else auth
    routing_key = routing_key or (str(uuid4()) if is_app else None)
    request = _build_request(
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
    session = _HttpSession()
    printer = _StreamPrinter(enabled=stream and obj.output == "text")
    response = session.send(request, on_event=printer)
    printer.finish()
    if not _success(response.status_code, expect_status):
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
        if not _success(response.status_code, expect_status):
            raise AgentCliError(f"Polling returned HTTP {response.status_code}.")
    _render_response(response, output=obj.output, streamed=stream and not wait)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return sorted(values)[index]


def _loadtest_result(
    *,
    started: float,
    responses: list[EndpointResponse],
    failures: list[str],
    expected: tuple[int, ...],
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    successful = [response for response in responses if _success(response.status_code, expected)]
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


def _render_loadtest(result: dict[str, Any], *, output: str) -> None:
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


@endpoint.command("loadtest")
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
    base_url, is_app, _ = _resolve_target(
        target=target,
        url=url,
        source=source,
        profile=obj.profile,
    )
    authenticate = is_app if auth is None else auth
    session_count = routing_keys or concurrency
    keys = [str(uuid4()) for _ in range(session_count)] if is_app else [None]
    template = _build_request(
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
            cookie = f"{_ROUTING_COOKIE}={routing_key}"
            headers["Cookie"] = f"{headers['Cookie']}; {cookie}" if "Cookie" in headers else cookie
        return _HttpSession().send(replace(template, headers=headers, body=body))

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
    result = _loadtest_result(
        started=started,
        responses=responses,
        failures=failures,
        expected=expect_status,
    )
    _render_loadtest(result, output=obj.output)
    if result["failed"]:
        raise AgentCliError(f"{result['failed']} load-test request(s) failed.")
