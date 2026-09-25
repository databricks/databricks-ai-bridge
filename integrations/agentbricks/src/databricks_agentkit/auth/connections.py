"""Async clients for invoking third-party APIs through governed UC Connections."""

from __future__ import annotations

import asyncio
import json as json_module
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast
from urllib.parse import quote, unquote, urlsplit

import httpx

from databricks_agentbricks.agent_project import AgentProject, ConnectionSpec
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit.runtime.workspace import workspace_client

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


HttpMethod: TypeAlias = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
QueryParams: TypeAlias = Mapping[str, str | int | float | bool | list[str]]

_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
_MCP_METHODS = frozenset({"GET", "POST", "DELETE"})
_METHOD = re.compile(r"^[A-Za-z]+$")
_FORBIDDEN_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "host"})


class ConnectionError(RuntimeError):
    """A credential-free governed-connection failure safe to return to agent code."""


class ConnectionTimeoutError(ConnectionError):
    """The Databricks connection proxy did not respond within the requested timeout."""


class ConnectionHTTPError(ConnectionError):
    """A non-success response returned by the external provider."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Connection request returned HTTP status {status_code}")
        self.status_code = status_code


@dataclass(frozen=True)
class ConnectionResponse:
    """Credential-free response returned by the Databricks connection proxy."""

    status_code: int
    headers: Mapping[str, str]
    content: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))
        object.__setattr__(self, "content", bytes(self.content))

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> JsonValue:
        return cast(JsonValue, json_module.loads(self.content))

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise ConnectionHTTPError(self.status_code)


class ConnectionRegistry:
    """Resolve public aliases from the active Agent Bricks project without resolving credentials."""

    def client(self, alias: str) -> ConnectionClient:
        if not isinstance(alias, str) or not alias:
            raise ConnectionError("Connection alias must be a non-empty string")
        try:
            project = AgentProject.load()
        except AgentCliError:
            raise ConnectionError(
                "Could not load connections from the active Agent Bricks project"
            ) from None
        for spec in project.connections:
            if spec.name == alias:
                return ConnectionClient(spec)
        raise ConnectionError("Connection alias is not declared in the active project")


class ConnectionClient:
    """Async client that routes requests through one governed UC Connection."""

    def __init__(self, spec: ConnectionSpec) -> None:
        self._spec = spec

    @property
    def alias(self) -> str:
        return self._spec.name

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: QueryParams | None = None,
        json: JsonValue = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ConnectionResponse:
        normalized_method = _validate_method(method, self._spec.transport)
        normalized_headers = _validate_headers(headers)
        _validate_timeout(timeout)
        proxy_path = _proxy_path(self._spec, normalized_method, path)
        selected_workspace = self._workspace_client()

        operation = asyncio.to_thread(
            _perform_request,
            selected_workspace,
            method=normalized_method,
            path=proxy_path,
            params=params,
            json_body=json,
            headers=normalized_headers,
            timeout=timeout,
        )
        try:
            raw_response = (
                await asyncio.wait_for(operation, timeout=timeout)
                if timeout is not None
                else await operation
            )
        except (asyncio.TimeoutError, httpx.TimeoutException):
            raise ConnectionTimeoutError("Connection proxy request timed out") from None
        except Exception:
            raise ConnectionError("Connection proxy request failed") from None

        return ConnectionResponse(
            status_code=raw_response.status_code,
            headers=raw_response.headers,
            content=raw_response.content,
        )

    def _workspace_client(self) -> WorkspaceClient:
        if self._spec.principal == "app":
            return workspace_client()

        from databricks_agentkit.auth.context import _current_request_auth

        request_auth = _current_request_auth()
        if request_auth is None:
            raise ConnectionError(
                "User connection requires an active request authentication context"
            )
        try:
            return request_auth.client_for("user")
        except Exception:
            raise ConnectionError("Request-user authentication failed") from None


def _validate_method(method: str, transport: str) -> str:
    if not isinstance(method, str) or not _METHOD.fullmatch(method):
        raise ConnectionError("Unsupported connection method")
    normalized = method.upper()
    allowed = _MCP_METHODS if transport == "mcp" else _HTTP_METHODS
    if normalized not in allowed:
        raise ConnectionError("Unsupported connection method")
    return normalized


def _validate_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    if headers is None:
        return {}
    result: dict[str, str] = {}
    try:
        items = headers.items()
    except AttributeError:
        raise ConnectionError("Connection headers must be a string mapping") from None
    for name, value in items:
        if not isinstance(name, str) or not isinstance(value, str):
            raise ConnectionError("Connection headers must be a string mapping")
        normalized = name.lower()
        if normalized in _FORBIDDEN_HEADERS or normalized.startswith("x-databricks-"):
            raise ConnectionError("Security-sensitive connection header is not allowed")
        if any(ord(character) < 32 or ord(character) == 127 for character in name + value):
            raise ConnectionError("Invalid connection header")
        result[name] = value
    return result


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ConnectionError("Connection timeout must be a positive finite number")
    if timeout <= 0 or not math.isfinite(timeout):
        raise ConnectionError("Connection timeout must be a positive finite number")


def _proxy_path(spec: ConnectionSpec, method: str, path: str) -> str:
    if not isinstance(path, str):
        raise ConnectionError("Connection path must be a string")
    base = f"/api/2.0/unity-catalog/connections/{quote(spec.uc_connection, safe='')}/proxy"
    if spec.transport == "mcp":
        if method not in _MCP_METHODS or path not in ("", "/"):
            raise ConnectionError("MCP connection path must be empty or '/'")
        return base

    relative_path = _validate_http_path(path)
    return f"{base}/{relative_path}" if relative_path else base


def _validate_http_path(path: str) -> str:
    if path.startswith("//") or "\\" in path:
        raise ConnectionError("Connection path must be relative")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise ConnectionError("Invalid connection path")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ConnectionError("Connection path must not contain an origin, query, or fragment")

    decoded = parsed.path
    for _ in range(5):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    if "\\" in decoded or any(part == ".." for part in decoded.split("/")):
        raise ConnectionError("Connection path must not contain parent traversal")
    return parsed.path.lstrip("/")


def _perform_request(
    workspace: WorkspaceClient,
    *,
    method: str,
    path: str,
    params: QueryParams | None,
    json_body: JsonValue,
    headers: Mapping[str, str],
    timeout: float | None,
) -> httpx.Response:
    """Perform one proxy call with fresh WorkspaceClient authentication headers."""
    config = workspace.config
    host = str(config.host or "").rstrip("/")
    if not host:
        raise RuntimeError("Workspace host is unavailable")
    custom_headers = dict(getattr(config, "_custom_headers", {}) or {})
    auth_headers = dict(config.authenticate())
    merged_headers = {**dict(headers), **custom_headers, **auth_headers}
    effective_timeout = timeout or config.http_timeout_seconds or 60
    return httpx.request(
        method,
        f"{host}{path}",
        params=params,
        json=json_body,
        headers=merged_headers,
        timeout=effective_timeout,
        follow_redirects=False,
    )


__all__ = [
    "ConnectionClient",
    "ConnectionError",
    "ConnectionHTTPError",
    "ConnectionResponse",
    "ConnectionTimeoutError",
    "HttpMethod",
    "JsonValue",
    "QueryParams",
]
