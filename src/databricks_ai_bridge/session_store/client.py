"""Async client for the experimental Databricks Session Store REST API."""

from __future__ import annotations

import os
from typing import Any, cast
from urllib.parse import quote

import httpx
from databricks.sdk import WorkspaceClient

from databricks_ai_bridge.utils.annotations import experimental

DEFAULT_API_PATH = "/api/2.0/agent-conversation"
DEFAULT_PAGE_SIZE = 1000


class SessionStoreError(RuntimeError):
    """Raised when the Databricks Session Store rejects a request."""

    def __init__(self, status_code: int, detail: object) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Session Store returned {status_code}: {detail}")


@experimental
class DatabricksSessionStoreClient:
    """Authenticated async client for the Databricks Session Store.

    The default endpoint is ``/api/2.0/agent-conversation`` on the configured
    workspace. ``base_url`` and ``traffic_id`` are explicit escape hatches for
    prototype deployments such as LiteSwap.

    Args:
        workspace_client: Databricks client used for the workspace URL and
            authentication. A default client is created when omitted.
        base_url: API prefix including the workspace host. Defaults to
            ``{workspace_host}/api/2.0/agent-conversation``.
        traffic_id: Optional ``x-databricks-traffic-id`` value.
        timeout: Request timeout in seconds when this class creates the HTTP
            client.
        http_client: Optional async HTTP client. Injected clients are not closed
            by :meth:`aclose`.
    """

    def __init__(
        self,
        *,
        workspace_client: WorkspaceClient | None = None,
        base_url: str | None = None,
        traffic_id: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._workspace_client = workspace_client or WorkspaceClient()
        configured_base_url = base_url or os.getenv("DATABRICKS_SESSION_STORE_BASE_URL")
        self._base_url = (
            configured_base_url
            or f"{self._workspace_client.config.host.rstrip('/')}{DEFAULT_API_PATH}"
        ).rstrip("/")
        self._traffic_id = traffic_id or os.getenv("DATABRICKS_SESSION_STORE_TRAFFIC_ID")
        self._http_client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_http_client = http_client is None

    async def __aenter__(self) -> DatabricksSessionStoreClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the internally created HTTP client, if any."""
        if self._owns_http_client:
            await self._http_client.aclose()

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        display_name: str = "",
        metadata: dict[str, str] | None = None,
        user_id: str | None = None,
        parent_session: str | None = None,
    ) -> dict[str, Any]:
        """Create a session, optionally with a caller-provided ID."""
        body: dict[str, Any] = {
            "display_name": display_name,
            "metadata": dict(metadata or {}),
        }
        if user_id is not None:
            body["user_id"] = user_id
        if parent_session is not None:
            body["parent_session"] = parent_session
        params = {"session_id": session_id} if session_id is not None else None
        response = await self._request("POST", "/sessions", params=params, json=body)
        self._raise_for_status(response)
        return self._json_object(response)

    async def ensure_session(
        self,
        session_id: str,
        *,
        display_name: str = "",
        metadata: dict[str, str] | None = None,
        user_id: str | None = None,
        parent_session: str | None = None,
    ) -> dict[str, Any]:
        """Create a session, or return the existing session on a conflict."""
        body: dict[str, Any] = {
            "display_name": display_name,
            "metadata": dict(metadata or {}),
        }
        if user_id is not None:
            body["user_id"] = user_id
        if parent_session is not None:
            body["parent_session"] = parent_session
        response = await self._request(
            "POST", "/sessions", params={"session_id": session_id}, json=body
        )
        if response.status_code == 409:
            return await self.get_session(session_id)
        self._raise_for_status(response)
        return self._json_object(response)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Return one session by ID."""
        response = await self._request("GET", self._session_path(session_id))
        self._raise_for_status(response)
        return self._json_object(response)

    async def session_exists(self, session_id: str) -> bool:
        """Return whether a session exists."""
        response = await self._request("GET", self._session_path(session_id))
        if response.status_code == 404:
            return False
        self._raise_for_status(response)
        return True

    async def list_sessions(
        self,
        *,
        user_id: str | None = None,
        parent_session: str | None = None,
        root_session: str | None = None,
        filter_expression: str | None = None,
        order_by: str | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> list[dict[str, Any]]:
        """Return all sessions matching the supplied filters."""
        result: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, str | int] = {"page_size": page_size}
            if user_id is not None:
                params["user_id"] = user_id
            if parent_session is not None:
                params["parent_session"] = parent_session
            if root_session is not None:
                params["root_session"] = root_session
            if filter_expression is not None:
                params["filter"] = filter_expression
            if order_by is not None:
                params["order_by"] = order_by
            if page_token is not None:
                params["page_token"] = page_token
            response = await self._request("GET", "/sessions", params=params)
            self._raise_for_status(response)
            body = self._json_object(response)
            result.extend(self._json_object_list(body.get("sessions"), "sessions"))
            page_token = self._next_page_token(body)
            if page_token is None:
                return result

    async def append_events(
        self,
        session_id: str,
        events: list[dict[str, Any]],
        *,
        idempotency_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Append events and return the service-owned event resources."""
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        response = await self._request(
            "POST",
            f"{self._session_path(session_id)}/events:append",
            json={"events": events},
            headers=headers,
        )
        self._raise_for_status(response)
        return self._json_object_list(self._json_object(response).get("events"), "events")

    async def list_events(
        self,
        session_id: str,
        *,
        filter_expression: str | None = None,
        order_by: str = "sequence asc",
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> list[dict[str, Any]]:
        """Return every event in a session in the requested order."""
        result: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, str | int] = {
                "page_size": page_size,
                "order_by": order_by,
            }
            if filter_expression is not None:
                params["filter"] = filter_expression
            if page_token is not None:
                params["page_token"] = page_token
            response = await self._request(
                "GET", f"{self._session_path(session_id)}/events", params=params
            )
            if response.status_code == 404:
                return []
            self._raise_for_status(response)
            body = self._json_object(response)
            result.extend(self._json_object_list(body.get("events"), "events"))
            page_token = self._next_page_token(body)
            if page_token is None:
                return result

    async def pop_event(self, session_id: str) -> dict[str, Any] | None:
        """Remove and return the most recently appended event."""
        response = await self._request(
            "POST", f"{self._session_path(session_id)}/events:pop", json={}
        )
        self._raise_for_status(response)
        event = self._json_object(response).get("event")
        if event is None:
            return None
        if not isinstance(event, dict):
            raise TypeError("Session Store response field 'event' must be an object")
        return event

    async def clear_events(self, session_id: str) -> None:
        """Remove every event from a session without deleting the session."""
        response = await self._request(
            "POST", f"{self._session_path(session_id)}/events:clear", json={}
        )
        self._raise_for_status(response)

    async def delete_session(self, session_id: str, *, force: bool = False) -> None:
        """Delete a session, optionally cascading to its descendants."""
        response = await self._request(
            "DELETE",
            self._session_path(session_id),
            params={"force": str(force).lower()},
        )
        if response.status_code == 404:
            return
        self._raise_for_status(response)

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = dict(self._workspace_client.config.authenticate())
        headers["Content-Type"] = "application/json"
        if self._traffic_id is not None:
            headers["x-databricks-traffic-id"] = self._traffic_id
        headers.update(kwargs.pop("headers", None) or {})
        return await self._http_client.request(
            method, f"{self._base_url}{path}", headers=headers, **kwargs
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            detail: object = response.json()
        except ValueError:
            detail = response.text
        raise SessionStoreError(response.status_code, detail)

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        value = response.json()
        if not isinstance(value, dict):
            raise TypeError("Session Store response must be a JSON object")
        return cast(dict[str, Any], value)

    @staticmethod
    def _json_object_list(value: object, field_name: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise TypeError(
                f"Session Store response field {field_name!r} must be a list of objects"
            )
        return cast(list[dict[str, Any]], value)

    @staticmethod
    def _next_page_token(body: dict[str, Any]) -> str | None:
        value = body.get("next_page_token")
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise TypeError("Session Store response field 'next_page_token' must be a string")
        return value

    @staticmethod
    def _session_path(session_id: str) -> str:
        return f"/sessions/{quote(session_id, safe='')}"
