"""OAuth discovery, DCR, and UC HTTP Connection creation for Agent Bricks."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import ConnectionInfo, ConnectionType

from databricks_agentbricks.errors import AgentCliError

_HTTP_TIMEOUT_SECONDS = 10
_WWW_AUTH_PARAMETER = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"')


@dataclass(frozen=True)
class HttpJsonResponse:
    """Small injectable HTTP result used by the registration workflow."""

    status_code: int
    headers: Mapping[str, str]
    payload: Mapping[str, Any] | None


HttpRequest = Callable[[str, str, Mapping[str, Any] | None], HttpJsonResponse]


def _https_url(value: str, description: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise AgentCliError(f"{description} must be a safe HTTPS URL.")
    return value


def _fqn_parts(fqn: str) -> tuple[str, str]:
    parts = fqn.split(".") if isinstance(fqn, str) else []
    if len(parts) != 3 or any(not part or any(char.isspace() for char in part) for part in parts):
        raise AgentCliError(
            f"Invalid UC Connection name {fqn!r}.",
            hint="Use a three-part name: catalog.schema.connection.",
        )
    return f"schemas/{'.'.join(parts[:2])}", parts[2]


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.lower()
    return next((value for key, value in headers.items() if key.lower() == expected), None)


def _auth_parameter(header: str | None, name: str) -> str | None:
    if not header:
        return None
    return next(
        (value for key, value in _WWW_AUTH_PARAMETER.findall(header) if key == name),
        None,
    )


def _protected_resource_candidates(provider_url: str, challenge: str | None) -> list[str]:
    parsed = urlsplit(provider_url)
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    path = parsed.path.lstrip("/")
    candidates: list[str] = []
    from_header = _auth_parameter(challenge, "resource_metadata")
    if from_header:
        candidates.append(_https_url(from_header, "OAuth resource metadata URL"))
    if path:
        candidates.append(f"{origin}/.well-known/oauth-protected-resource/{path}")
    candidates.append(f"{origin}/.well-known/oauth-protected-resource")
    return list(dict.fromkeys(candidates))


def _authorization_metadata_candidates(authorization_server: str) -> list[str]:
    _https_url(authorization_server, "OAuth authorization server URL")
    base = authorization_server.rstrip("/") + "/"
    parsed = urlsplit(authorization_server)
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    path = parsed.path.rstrip("/")
    candidates = [
        urljoin(base, ".well-known/oauth-authorization-server"),
        urljoin(base, ".well-known/openid-configuration"),
    ]
    if path:
        candidates.extend(
            [
                f"{origin}/.well-known/oauth-authorization-server{path}",
                f"{origin}/.well-known/openid-configuration{path}",
            ]
        )
    return list(dict.fromkeys(candidates))


def _first_json(request: HttpRequest, urls: list[str], description: str) -> Mapping[str, Any]:
    for url in urls:
        response = request("GET", _https_url(url, f"{description} URL"), None)
        if 200 <= response.status_code < 300 and response.payload is not None:
            return response.payload
    raise AgentCliError(f"Could not discover {description}.")


def _authorization_server(resource_metadata: Mapping[str, Any]) -> str:
    servers = resource_metadata.get("authorization_servers")
    if isinstance(servers, list) and servers and isinstance(servers[0], str):
        return servers[0]
    for key in ("authorization_server", "issuer"):
        value = resource_metadata.get(key)
        if isinstance(value, str) and value:
            return value
    raise AgentCliError("OAuth resource metadata has no authorization server.")


def _scope(resource_metadata: Mapping[str, Any], challenge: str | None) -> str:
    challenged = _auth_parameter(challenge, "scope")
    if challenged is not None:
        return challenged
    supported = resource_metadata.get("scopes_supported")
    if isinstance(supported, list) and all(isinstance(value, str) for value in supported):
        return " ".join(supported)
    if isinstance(supported, str):
        return supported
    return ""


def _required_endpoint(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise AgentCliError(f"OAuth authorization metadata has no {key.replace('_', ' ')}.")
    return _https_url(value, f"OAuth {key.replace('_', ' ')}")


def _default_request(
    method: str, url: str, json_body: Mapping[str, Any] | None
) -> HttpJsonResponse:
    data = None if json_body is None else json.dumps(json_body).encode("utf-8")
    headers = {"Accept": "application/json", "User-Agent": "Databricks-Agent-Bricks"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS)
        status = response.status
        response_headers = dict(response.headers.items())
        body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        body = exc.read() if 200 <= status < 300 else b""
    except (OSError, TimeoutError) as exc:
        host = urlsplit(url).hostname or "OAuth endpoint"
        raise AgentCliError(f"Could not reach OAuth endpoint at {host}.") from exc
    payload: Mapping[str, Any] | None = None
    if body:
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AgentCliError("OAuth endpoint returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise AgentCliError("OAuth endpoint returned a non-object JSON response.")
        payload = parsed
    return HttpJsonResponse(status, response_headers, payload)


def register_connection_via_dcr(
    workspace_client: Any,
    *,
    fqn: str,
    url: str,
    transport: Literal["mcp", "http"],
    request: HttpRequest | None = None,
) -> ConnectionInfo:
    """Register an OAuth client and create one schema-level UC HTTP Connection."""

    provider_url = _https_url(url, "Provider URL")
    parent, name = _fqn_parts(fqn)
    if transport not in ("mcp", "http"):
        raise AgentCliError("Connection transport must be 'mcp' or 'http'.")
    try:
        workspace_client.connections.get(fqn)
    except NotFound:
        pass
    except Exception as exc:
        raise AgentCliError(f"Could not read UC Connection {fqn!r}.") from exc
    else:
        raise AgentCliError(f"UC Connection {fqn!r} already exists.")

    send = request or _default_request
    challenge_response = send("GET", provider_url, None)
    if challenge_response.status_code != 401:
        raise AgentCliError("OAuth discovery expected a 401 challenge from the provider URL.")
    challenge = _header(challenge_response.headers, "WWW-Authenticate")
    resource_metadata = _first_json(
        send,
        _protected_resource_candidates(provider_url, challenge),
        "OAuth protected resource metadata",
    )
    authorization_server = _authorization_server(resource_metadata)
    authorization_metadata = _first_json(
        send,
        _authorization_metadata_candidates(authorization_server),
        "OAuth authorization server metadata",
    )
    registration_endpoint = _required_endpoint(authorization_metadata, "registration_endpoint")
    authorization_endpoint = _required_endpoint(authorization_metadata, "authorization_endpoint")
    token_endpoint = _required_endpoint(authorization_metadata, "token_endpoint")
    workspace_host = str(getattr(workspace_client.config, "host", "") or "").rstrip("/")
    if not workspace_host:
        raise AgentCliError("Databricks workspace host is not configured.")
    registration = send(
        "POST",
        registration_endpoint,
        {
            "client_name": "Databricks Agent Bricks",
            "redirect_uris": [f"{workspace_host}/login/oauth/http.html"],
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
    )
    if not 200 <= registration.status_code < 300 or registration.payload is None:
        raise AgentCliError("Dynamic client registration failed.")
    client_id = registration.payload.get("client_id")
    if not isinstance(client_id, str) or not client_id:
        raise AgentCliError("Dynamic client registration returned no client id.")

    parsed_provider = urlsplit(provider_url)
    options = {
        "host": urlunsplit((parsed_provider.scheme, parsed_provider.netloc, "", "", "")),
        "port": str(parsed_provider.port or 443),
        "base_path": parsed_provider.path or "/",
        "oauth_credential_exchange_method": "header_and_body",
        "client_id": client_id,
        "authorization_endpoint": authorization_endpoint,
        "token_endpoint": token_endpoint,
        "oauth_scope": _scope(resource_metadata, challenge),
    }
    client_secret = registration.payload.get("client_secret")
    if isinstance(client_secret, str):
        options["client_secret"] = client_secret
    try:
        return workspace_client.connections.create(
            name=name,
            parent=parent,
            connection_type=ConnectionType.HTTP,
            options=options,
        )
    except Exception as exc:
        raise AgentCliError(f"Could not create UC Connection {fqn!r}.") from exc
