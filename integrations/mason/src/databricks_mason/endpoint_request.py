"""Target resolution and request construction for Mason endpoint commands."""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.parse
from typing import Any, Mapping, Optional

from databricks_mason._api_client import _workspace_client
from databricks_mason.agent_project import AgentProject
from databricks_mason.deploy import _app_url, _prefixed_name
from databricks_mason.endpoint_presets import EndpointPreset, build_preset_body
from databricks_mason.endpoint_transport import EndpointRequest
from databricks_mason.errors import AgentCliError

ROUTING_COOKIE = "__Host-databricks-app-router"


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


def load_json(json_value: str | None, json_file: str | None) -> Any:
    """Load a JSON body from an option, file, or stdin."""
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


def resolve_target(
    *,
    target: str | None,
    url: str | None,
    source: str,
    profile: Optional[str],
) -> tuple[str, bool, str | None]:
    """Resolve an App name, recorded deployment, or explicit URL."""
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


def auth_headers(profile: Optional[str]) -> dict[str, str]:
    """Resolve the OAuth header required by Databricks Apps API routes."""
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


def request_url(base_url: str, path: str, query: Mapping[str, str]) -> str:
    """Join a request path to a base URL and merge query parameters."""
    if not path:
        raise AgentCliError("--path is required when no --preset is selected.")
    joined = urllib.parse.urljoin(f"{base_url}/", path)
    if not query:
        return joined
    parsed = urllib.parse.urlsplit(joined)
    merged_query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    merged_query.update(query)
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(merged_query)))


def request_headers(
    values: tuple[str, ...],
    *,
    authenticate: bool,
    profile: Optional[str],
    routing_key: str | None,
) -> dict[str, str]:
    """Build request headers, including optional OAuth and sticky routing."""
    headers = parse_pairs(values, separator=":", description="header")
    if authenticate:
        for key, value in auth_headers(profile).items():
            headers.setdefault(key, value)
    headers.setdefault("Content-Type", "application/json")
    if routing_key:
        cookie = f"{ROUTING_COOKIE}={routing_key}"
        headers["Cookie"] = f"{headers['Cookie']}; {cookie}" if "Cookie" in headers else cookie
    return headers


def build_request(
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
    """Materialize one generic or preset endpoint request."""
    body = load_json(json_value, json_file)
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
    request_query = parse_pairs(query, separator="=", description="query parameter")
    return EndpointRequest(
        url=request_url(base_url, request_path, request_query),
        method=method.upper(),
        headers=request_headers(
            header,
            authenticate=authenticate,
            profile=profile,
            routing_key=routing_key,
        ),
        body=body,
        timeout=timeout,
        stream=stream,
    )
