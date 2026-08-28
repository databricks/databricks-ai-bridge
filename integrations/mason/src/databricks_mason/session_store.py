"""Typed session-store resources and operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from databricks_mason._pagination import validate_page_size
from databricks_mason.timefmt import parse_timestamp

if TYPE_CHECKING:
    from databricks_mason.client import MasonClient


@dataclass(frozen=True)
class SessionItem:
    item_id: str
    data: Any
    create_time: Optional[datetime] = None


@dataclass(frozen=True)
class SessionItemPage:
    items: List[SessionItem] = field(default_factory=list)
    next_page_token: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class Session:
    session_store_name: str
    session_id: str
    actor_id: str
    parent_session_id: Optional[str] = None
    root_session_id: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    last_activity_time: Optional[datetime] = None
    _client: SessionStoreClient = field(repr=False, compare=False)

    def update(self, *, metadata: dict[str, str]) -> Session:
        return self._client.update_session(self, metadata=metadata)

    def delete(self) -> None:
        self._client.delete_session(self)

    def fork(
        self,
        *,
        actor_id: str,
        up_to_item_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Session:
        return self._client.fork_session(
            self,
            actor_id=actor_id,
            up_to_item_id=up_to_item_id,
            session_id=session_id,
            metadata=metadata,
        )

    def list_items(
        self,
        *,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> SessionItemPage:
        return self._client.list_items(
            self,
            page_size=page_size,
            page_token=page_token,
            order_by=order_by,
        )

    def append_items(self, items: Sequence[Any]) -> List[SessionItem]:
        return self._client.append_items(self, items=items)

    def pop_item(self) -> Optional[SessionItem]:
        return self._client.pop_item(self)

    def clear_items(self) -> None:
        self._client.clear_items(self)


@dataclass(frozen=True, kw_only=True)
class SessionStore:
    session_store_name: str
    session_store_id: Optional[str] = None
    creator_user_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    description: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)
    _client: SessionStoreClient = field(repr=False, compare=False)

    def update(
        self,
        *,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> SessionStore:
        return self._client.update(self, description=description, metadata=metadata)

    def delete(self) -> None:
        self._client.delete(self)

    def create_session(
        self,
        *,
        actor_id: str,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Session:
        return self._client.create_session(
            self,
            actor_id=actor_id,
            session_id=session_id,
            parent_session_id=parent_session_id,
            metadata=metadata,
        )

    def list_sessions(
        self,
        *,
        page_size: Optional[int] = None,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> List[Session]:
        return self._client.list_sessions(
            self,
            page_size=page_size,
            filter=filter,
            order_by=order_by,
        )

    def get_session(self, *, session_id: str) -> Session:
        return self._client.get_session(self, session_id=session_id)


class SessionStoreClient:
    def __init__(self, api: MasonClient):
        self._api = api

    def bind(self, session_store_name: str) -> SessionStore:
        """Create a bound local handle without making an API request."""
        if not session_store_name:
            raise ValueError("session_store_name is required")
        return SessionStore(session_store_name=session_store_name, _client=self)

    def create(
        self,
        *,
        session_store_name: str,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> SessionStore:
        response = self._api.create_session_store(
            session_store_name,
            description,
            metadata,
        )
        return self._store_from_response(response)

    def list(self, *, page_size: Optional[int] = None) -> List[SessionStore]:
        validate_page_size(page_size)
        stores = []
        page_token = None
        while True:
            response = self._api.list_session_stores(
                page_size=page_size,
                page_token=page_token,
            )
            stores.extend(
                self._store_from_response(store) for store in response.get("session_stores", [])
            )
            page_token = response.get("next_page_token")
            if not page_token:
                return stores

    def get(self, *, session_store_name: str) -> SessionStore:
        return self._store_from_response(self._api.get_session_store(session_store_name))

    def update(
        self,
        store: SessionStore,
        *,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> SessionStore:
        if description is None and metadata is None:
            raise ValueError("at least one of description and metadata is required")
        response = self._api.update_session_store(
            store.session_store_name,
            description=description,
            metadata=metadata,
        )
        return self._store_from_response(response)

    def delete(self, store: SessionStore) -> None:
        self._api.delete_session_store(store.session_store_name)

    def create_session(
        self,
        store: Optional[SessionStore] = None,
        *,
        session_store_name: Optional[str] = None,
        actor_id: str,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Session:
        name = self._resolve_session_store_name(store, session_store_name)
        response = self._api.create_session(
            name,
            actor_id,
            session_id=session_id,
            parent_session_id=parent_session_id,
            metadata=metadata,
        )
        return self._session_from_response(response, name)

    def list_sessions(
        self,
        store: Optional[SessionStore] = None,
        *,
        session_store_name: Optional[str] = None,
        page_size: Optional[int] = None,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> List[Session]:
        validate_page_size(page_size)
        name = self._resolve_session_store_name(store, session_store_name)
        if order_by is None:
            order_by = "create_time desc"
        sessions = []
        page_token = None
        while True:
            response = self._api.list_sessions(
                name,
                filter=filter,
                order_by=order_by,
                page_size=page_size,
                page_token=page_token,
            )
            sessions.extend(
                self._session_from_response(session, name)
                for session in response.get("sessions", [])
            )
            page_token = response.get("next_page_token")
            if not page_token:
                return sessions

    def get_session(
        self,
        store: Optional[SessionStore] = None,
        *,
        session_store_name: Optional[str] = None,
        session_id: str,
    ) -> Session:
        name = self._resolve_session_store_name(store, session_store_name)
        response = self._api.get_session(session_id, name)
        return self._session_from_response(response, name)

    def update_session(self, session: Session, *, metadata: dict[str, str]) -> Session:
        response = self._api.update_session(
            session.session_store_name,
            session.session_id,
            metadata,
        )
        return self._session_from_response(response, session.session_store_name)

    def delete_session(self, session: Session) -> None:
        self._api.delete_session(session.session_store_name, session.session_id)

    def fork_session(
        self,
        session: Session,
        *,
        actor_id: str,
        up_to_item_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Session:
        response = self._api.fork_session(
            session.session_store_name,
            session.session_id,
            actor_id,
            up_to_item_id=up_to_item_id,
            session_id=session_id,
            metadata=metadata,
        )
        return self._session_from_response(
            response.get("session", response),
            session.session_store_name,
        )

    def list_items(
        self,
        session: Session,
        *,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> SessionItemPage:
        validate_page_size(page_size)
        response = self._api.list_session_items(
            session.session_store_name,
            session.session_id,
            order_by=order_by,
            page_size=page_size,
            page_token=page_token,
        )
        return SessionItemPage(
            items=[self._item_from_response(item) for item in response.get("session_items", [])],
            next_page_token=response.get("next_page_token"),
        )

    def append_items(self, session: Session, *, items: Sequence[Any]) -> List[SessionItem]:
        if not items:
            raise ValueError("at least one item is required")
        response = self._api.append_session_items(
            session.session_store_name,
            session.session_id,
            list(items),
        )
        return [self._item_from_response(item) for item in response.get("session_items", [])]

    def pop_item(self, session: Session) -> Optional[SessionItem]:
        response = self._api.pop_session_item(session.session_store_name, session.session_id)
        item = response.get("item")
        return self._item_from_response(item) if item is not None else None

    def clear_items(self, session: Session) -> None:
        self._api.clear_session_items(session.session_store_name, session.session_id)

    @staticmethod
    def _resolve_session_store_name(
        store: Optional[SessionStore],
        session_store_name: Optional[str],
    ) -> str:
        if store is None:
            if not session_store_name:
                raise ValueError("session_store_name is required")
            return session_store_name
        if session_store_name is not None and session_store_name != store.session_store_name:
            raise ValueError("session_store_name conflicts with the bound session store")
        return store.session_store_name

    def _store_from_response(self, response: dict[str, Any]) -> SessionStore:
        return SessionStore(
            session_store_name=response["session_store_name"],
            session_store_id=response.get("session_store_id"),
            creator_user_id=response.get("creator_user_id"),
            create_time=parse_timestamp(response.get("create_time")),
            update_time=parse_timestamp(response.get("update_time")),
            description=response.get("description"),
            metadata=dict(response.get("metadata", {})),
            _client=self,
        )

    def _session_from_response(
        self,
        response: dict[str, Any],
        session_store_name: str,
    ) -> Session:
        return Session(
            session_store_name=response.get("session_store_name", session_store_name),
            session_id=response["session_id"],
            actor_id=response["actor_id"],
            parent_session_id=response.get("parent_session_id"),
            root_session_id=response.get("root_session_id"),
            metadata=dict(response.get("metadata", {})),
            create_time=parse_timestamp(response.get("create_time")),
            update_time=parse_timestamp(response.get("update_time")),
            last_activity_time=parse_timestamp(response.get("last_activity_time")),
            _client=self,
        )

    @staticmethod
    def _item_from_response(response: dict[str, Any]) -> SessionItem:
        return SessionItem(
            item_id=response["item_id"],
            data=response["data"],
            create_time=parse_timestamp(response.get("create_time")),
        )
