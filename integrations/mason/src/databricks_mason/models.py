"""Typed resource models for the Mason Python SDK.

These dataclasses wrap the raw ``agents/v1`` JSON returned by :class:`AgentApiClient`
and carry a back-reference to the SDK store client, so nested calls
(``store.add(...)``, ``session.append(...)``) stay scoped to the right resource.
The CLI works with raw dicts; only the SDK layer (:mod:`databricks_mason.sdk`) uses
these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from databricks_mason.sdk import MemoryStoreClient, SessionStoreClient


def _resource_id(name: str) -> str:
    return name.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ManagedMemoryEntry:
    name: str
    actor_id: str
    path: str
    session_id: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    source_type: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None

    @property
    def entry_id(self) -> str:
        return _resource_id(self.name)


@dataclass(frozen=True)
class ManagedMemoryStore:
    name: str
    display_name: str
    workspace_id: Optional[int] = None
    storage_backend: Optional[dict[str, Any]] = None
    owner_user_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    description: Optional[str] = None
    _client: "MemoryStoreClient" = field(repr=False, compare=False, default=None)

    @property
    def store_id(self) -> str:
        return _resource_id(self.name)

    def add(
        self,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        return self._client.add(
            self,
            actor_id=actor_id,
            path=path,
            content=content,
            session_id=session_id,
            description=description,
            source_type=source_type,
        )

    def append(
        self,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        return self._client.append(
            self,
            actor_id=actor_id,
            path=path,
            content=content,
            session_id=session_id,
            description=description,
            source_type=source_type,
        )

    def list(
        self,
        *,
        actor_id: str,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> list[ManagedMemoryEntry]:
        return self._client.list(
            self,
            actor_id=actor_id,
            path_prefix=path_prefix,
            session_id=session_id,
        )

    def get(self, *, entry_id: str) -> ManagedMemoryEntry:
        return self._client.get_entry(self, entry_id=entry_id)

    def search(
        self, *, actor_id: str, query: str, limit: Optional[int] = None
    ) -> list[ManagedMemoryEntry]:
        return self._client.search(self, actor_id=actor_id, query=query, limit=limit)

    def delete(self, *, entry_id: str) -> None:
        self._client.delete(self, entry_id=entry_id)


@dataclass(frozen=True)
class SessionItem:
    item_id: str
    data: Any
    create_time: Optional[datetime] = None


@dataclass(frozen=True)
class SessionItemPage:
    items: list[SessionItem] = field(default_factory=list)
    next_page_token: Optional[str] = None


@dataclass(frozen=True)
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
    _client: "SessionStoreClient" = field(repr=False, compare=False, default=None)

    def update(self, *, metadata: dict[str, str]) -> "Session":
        return self._client.update_session(self, metadata=metadata)

    def delete(self, *, force: bool = False) -> None:
        self._client.delete_session(self, force=force)

    def fork(
        self,
        *,
        actor_id: str,
        up_to_item_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> "Session":
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
            self, page_size=page_size, page_token=page_token, order_by=order_by
        )

    def append(self, items: list[Any]) -> list[SessionItem]:
        return self._client.append_items(self, items=items)

    def pop(self) -> Optional[SessionItem]:
        return self._client.pop_item(self)

    def clear(self) -> None:
        self._client.clear_items(self)


@dataclass(frozen=True)
class SessionStore:
    session_store_name: str
    session_store_id: Optional[str] = None
    creator_user_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    description: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)
    _client: "SessionStoreClient" = field(repr=False, compare=False, default=None)

    def update(
        self, *, description: Optional[str] = None, metadata: Optional[dict[str, str]] = None
    ) -> "SessionStore":
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
    ) -> list[Session]:
        return self._client.list_sessions(
            self, page_size=page_size, filter=filter, order_by=order_by
        )

    def get_session(self, *, session_id: str) -> Session:
        return self._client.get_session(self, session_id=session_id)
