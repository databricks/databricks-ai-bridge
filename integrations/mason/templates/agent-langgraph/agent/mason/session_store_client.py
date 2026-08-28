"""Minimal Databricks Session Store REST client (vendored).

Wraps the managed Session Store API (``/api/agents/v1/session-stores``) — the SDK-agnostic durable
store for agent session history — with just the calls the LangGraph saver needs: resolve a store,
get-or-create a session, and append / list / clear its ordered items. Auth/host come from a
``databricks.sdk.WorkspaceClient`` (default credentials).

This is a trimmed stand-in for the first-party ``databricks_agent_client.SessionStoreClient``; when
that package is published, import it instead and delete this file. Item ``data`` is opaque JSON, so
callers store whatever shape they like (the saver stores checkpoint fragments).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from databricks.sdk import WorkspaceClient

from agent.mason.workspace import workspace_client as default_workspace_client

_API_ROOT = "/api/agents/v1"


@dataclass(frozen=True)
class Session:
    """A durable interaction within a Session Store, addressed by (store, session_id)."""

    session_store_name: str
    session_id: str
    actor_id: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionItem:
    """One opaque, ordered history entry. ``data`` is arbitrary JSON."""

    item_id: str
    data: Any


class SessionStoreClient:
    """Thin REST client over the managed Session Store API."""

    def __init__(self, workspace_client: Optional[WorkspaceClient] = None) -> None:
        self._api = (workspace_client or default_workspace_client()).api_client
        self._store_name: Optional[str] = None

    def set_session_store(self, session_store_name: str) -> "SessionStoreClient":
        """Bind this client to a store; returns self so it doubles as the store handle."""
        if not session_store_name:
            raise ValueError("session_store_name is required")
        self._store_name = session_store_name
        return self

    def get_session(self, *, session_id: str) -> Session:
        resp = self._api.do("GET", f"{self._sessions_path()}/{session_id}")
        return self._session(resp)

    def create_session(
        self,
        *,
        actor_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Session:
        body: dict[str, Any] = {"actor_id": actor_id}
        if metadata:
            body["metadata"] = metadata
        query = {"session_id": session_id} if session_id else None
        resp = self._api.do("POST", self._sessions_path(), query=query, body=body)
        return self._session(resp)

    def append_items(self, session: Session, *, items: Sequence[Any]) -> None:
        if not items:
            raise ValueError("at least one item is required")
        self._api.do(
            "POST",
            f"{self._items_path(session)}:append",
            body={"items": [{"data": item} for item in items]},
        )

    def list_items(self, session: Session, *, order_by: Optional[str] = None) -> Iterator[SessionItem]:
        """Yield the session's items (paginating transparently)."""
        page_token: Optional[str] = None
        while True:
            query = {k: v for k, v in {"order_by": order_by, "page_token": page_token}.items() if v}
            resp = self._api.do("GET", self._items_path(session), query=query or None)
            for item in resp.get("session_items", []):
                if "data" in item:
                    yield SessionItem(item_id=item.get("item_id", ""), data=item["data"])
            page_token = resp.get("next_page_token")
            if not page_token:
                return

    def clear_items(self, session: Session) -> None:
        self._api.do("POST", f"{self._items_path(session)}:clear", body={})

    # ----- internals ----------------------------------------------------------

    def _sessions_path(self) -> str:
        if not self._store_name:
            raise ValueError("call set_session_store() before using the client")
        return f"{_API_ROOT}/session-stores/{self._store_name}/sessions"

    def _items_path(self, session: Session) -> str:
        return f"{self._sessions_path()}/{session.session_id}/items"

    def _session(self, resp: dict[str, Any]) -> Session:
        return Session(
            session_store_name=resp.get("session_store_name", self._store_name or ""),
            session_id=resp["session_id"],
            actor_id=resp.get("actor_id", ""),
            metadata=dict(resp.get("metadata", {})),
        )
