"""Generic HTTP request construction for Mason endpoint commands."""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Mapping

from databricks_mason.endpoint_transport import EndpointRequest
from databricks_mason.errors import AgentCliError


def parse_pairs(values: tuple[str, ...], *, separator: str, description: str) -> dict[str, str]:
    """Parse repeatable CLI key/value options."""
    parsed: dict[str, str] = {}
    for value in values:
        key, found, item = value.partition(separator)
        key = key.strip()
        if not found or not key or "\n" in key or "\n" in item:
            raise AgentCliError(f"Invalid {description} {value!r}.")
        parsed[key] = item.strip()
    return parsed


def load_json(value: str | None) -> Any:
    """Validate and decode an optional JSON request body."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise AgentCliError(f"Invalid JSON request body: {exc}.") from exc


def request_url(base_url: str, path: str, query: Mapping[str, str]) -> str:
    """Join a request path to a base URL and merge query parameters."""
    parsed_path = urllib.parse.urlsplit(path)
    if parsed_path.scheme or parsed_path.netloc:
        raise AgentCliError("--path must be relative to the selected endpoint URL.")
    joined = urllib.parse.urljoin(f"{base_url}/", path)
    if not query:
        return joined
    parsed = urllib.parse.urlsplit(joined)
    merged_query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    merged_query.update(query)
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(merged_query)))


def build_request(
    *,
    base_url: str,
    method: str,
    path: str,
    query: tuple[str, ...],
    json_value: str | None,
    timeout: float,
    sse: bool,
) -> EndpointRequest:
    """Materialize one generic HTTP request."""
    body = load_json(json_value)
    request_query = parse_pairs(query, separator="=", description="query parameter")
    headers = {"Content-Type": "application/json"} if json_value is not None else {}
    return EndpointRequest(
        url=request_url(base_url, path, request_query),
        method=method.upper(),
        headers=headers,
        body=body,
        timeout=timeout,
        sse=sse,
        body_set=json_value is not None,
    )
