"""Typed session-store resources and operations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from databricks_mason._pagination import validate_page_size
from databricks_mason.timefmt import parse_timestamp

if TYPE_CHECKING:
    from databricks_mason._api_client import _MasonApiClient


@dataclass(frozen=True)
class SessionItem:
    item_id: str
    data: Any
    create_time: Optional[datetime] = None


@dataclass(frozen=True, kw_only=True)
class Session:
    store_name: str
    session_id: str
    actor_id: str
    parent_session_id: Optional[str] = None
    root_session_id: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    last_activity_time: Optional[datetime] = None
    _client: SessionStores = field(repr=False, compare=False)

    def update(self, *, metadata: dict[str, str]) -> Session:
        return self._client._update_session(self, metadata=metadata)

    def delete(self) -> None:
        self._client._delete_session(self)

    def fork(
        self,
        *,
        actor_id: str,
        up_to_item_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Session:
        return self._client._fork_session(
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
        order_by: Optional[str] = None,
    ) -> Iterator[SessionItem]:
        return self._client._list_items(
            self,
            page_size=page_size,
            order_by=order_by,
        )

    def append_items(self, items: Sequence[Any]) -> List[SessionItem]:
        return self._client._append_items(self, items=items)

    def pop_item(self) -> Optional[SessionItem]:
        return self._client._pop_item(self)

    def clear_items(self) -> None:
        self._client._clear_items(self)


@dataclass(frozen=True, kw_only=True)
class SessionStore:
    name: str
    session_store_id: Optional[str] = None
    creator_user_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    description: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)
    _client: SessionStores = field(repr=False, compare=False)

    def update(
        self,
        *,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> SessionStore:
        return self._client._update_store(
            self,
            description=description,
            metadata=metadata,
        )

    def delete(self) -> None:
        self._client._delete_store(self)

    def create_session(
        self,
        *,
        actor_id: str,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Session:
        return self._client._create_session(
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
    ) -> Iterator[Session]:
        return self._client._list_sessions(
            self,
            page_size=page_size,
            filter=filter,
            order_by=order_by,
        )

    def get_session(self, session_id: str) -> Session:
        return self._client._get_session(self, session_id=session_id)


class SessionStores:
    def __init__(self, api: _MasonApiClient):
        self._api = api

    def create(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> SessionStore:
        response = self._api.create_session_store(
            name,
            description,
            metadata,
        )
        return self._store_from_response(response)

    def list(self, *, page_size: Optional[int] = None) -> Iterator[SessionStore]:
        validate_page_size(page_size)
        page_token = None
        while True:
            response = self._api.list_session_stores(
                page_size=page_size,
                page_token=page_token,
            )
            for store in response.get("session_stores", []):
                yield self._store_from_response(store)
            page_token = response.get("next_page_token")
            if not page_token:
                return

    def get(self, name: str) -> SessionStore:
        return self._store_from_response(self._api.get_session_store(name))

    def _update_store(
        self,
        store: SessionStore,
        *,
        description: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> SessionStore:
        if description is None and metadata is None:
            raise ValueError("at least one of description and metadata is required")
        response = self._api.update_session_store(
            store.name,
            description=description,
            metadata=metadata,
        )
        return self._store_from_response(response)

    def _delete_store(self, store: SessionStore) -> None:
        self._api.delete_session_store(store.name)

    def _create_session(
        self,
        store: SessionStore,
        *,
        actor_id: str,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Session:
        response = self._api.create_session(
            store.name,
            actor_id,
            session_id=session_id,
            parent_session_id=parent_session_id,
            metadata=metadata,
        )
        return self._session_from_response(response, store.name)

    def _list_sessions(
        self,
        store: SessionStore,
        *,
        page_size: Optional[int] = None,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> Iterator[Session]:
        validate_page_size(page_size)
        if order_by is None:
            order_by = "create_time desc"
        page_token = None
        while True:
            response = self._api.list_sessions(
                store.name,
                filter=filter,
                order_by=order_by,
                page_size=page_size,
                page_token=page_token,
            )
            for session in response.get("sessions", []):
                yield self._session_from_response(session, store.name)
            page_token = response.get("next_page_token")
            if not page_token:
                return

    def _get_session(
        self,
        store: SessionStore,
        *,
        session_id: str,
    ) -> Session:
        response = self._api.get_session(session_id, store.name)
        return self._session_from_response(response, store.name)

    def _update_session(self, session: Session, *, metadata: dict[str, str]) -> Session:
        response = self._api.update_session(
            session.store_name,
            session.session_id,
            metadata,
        )
        return self._session_from_response(response, session.store_name)

    def _delete_session(self, session: Session) -> None:
        self._api.delete_session(session.store_name, session.session_id)

    def _fork_session(
        self,
        session: Session,
        *,
        actor_id: str,
        up_to_item_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Session:
        response = self._api.fork_session(
            session.store_name,
            session.session_id,
            actor_id,
            up_to_item_id=up_to_item_id,
            session_id=session_id,
            metadata=metadata,
        )
        return self._session_from_response(
            response.get("session", response),
            session.store_name,
        )

    def _list_items(
        self,
        session: Session,
        *,
        page_size: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> Iterator[SessionItem]:
        validate_page_size(page_size)
        page_token = None
        while True:
            response = self._api.list_session_items(
                session.store_name,
                session.session_id,
                order_by=order_by,
                page_size=page_size,
                page_token=page_token,
            )
            for item in response.get("session_items", []):
                yield self._item_from_response(item)
            page_token = response.get("next_page_token")
            if not page_token:
                return

    def _append_items(self, session: Session, *, items: Sequence[Any]) -> List[SessionItem]:
        if not items:
            raise ValueError("at least one item is required")
        response = self._api.append_session_items(
            session.store_name,
            session.session_id,
            list(items),
        )
        return [self._item_from_response(item) for item in response.get("session_items", [])]

    def _pop_item(self, session: Session) -> Optional[SessionItem]:
        response = self._api.pop_session_item(session.store_name, session.session_id)
        item = response.get("item")
        return self._item_from_response(item) if item is not None else None

    def _clear_items(self, session: Session) -> None:
        self._api.clear_session_items(session.store_name, session.session_id)

    def _store_from_response(self, response: dict[str, Any]) -> SessionStore:
        return SessionStore(
            name=response["session_store_name"],
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
            store_name=response.get("session_store_name", session_store_name),
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
