"""Typed, importable Python SDK for the Databricks agent memory and session APIs.

This is a thin ergonomic layer over :class:`~databricks_mason.client.AgentApiClient`
(the same client the ``mason`` CLI uses, so it shares mason's profile auth and error
handling). Where ``AgentApiClient`` returns raw ``dict``s and single pages, this SDK
returns the typed handles in :mod:`databricks_mason.models`, auto-consumes pagination,
and adds convenience lookups (``get(display_name=..., create_if_not_exists=True)``,
``append``).

    from databricks_mason import DatabricksAgentClient

    client = DatabricksAgentClient(profile="my-profile")
    store = client.memory_store.get(display_name="coding_agent_memory", create_if_not_exists=True)
    store.add(actor_id="alice", path="/memories/prefs.md", content="Prefers concise answers.")
"""

from __future__ import annotations

from typing import Any, Optional

from databricks_mason.client import AgentApiClient
from databricks_mason.models import (
    ManagedMemoryEntry,
    ManagedMemoryStore,
    Session,
    SessionItem,
    SessionItemPage,
    SessionStore,
)
from databricks_mason.timefmt import parse_timestamp

_MAX_PAGE_SIZE = 100


def _validate_page_size(page_size: Optional[int]) -> None:
    """Reject page sizes the service rejects outright: it requires 1 <= page_size <= 100."""
    if page_size is not None and not 1 <= page_size <= _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {_MAX_PAGE_SIZE}")


class DatabricksAgentClient:
    """Entry point for the typed agent memory/session SDK.

    Args:
        profile: Databricks config profile used to build an :class:`AgentApiClient`
            when ``api_client`` is not supplied.
        api_client: An existing :class:`AgentApiClient` (or compatible stub) to use
            instead of constructing one; useful for testing and for sharing auth.
    """

    def __init__(self, profile: Optional[str] = None, *, api_client: Optional[Any] = None):
        api = api_client if api_client is not None else AgentApiClient(profile=profile)
        self.memory_store = MemoryStoreClient(api)
        self.session_store = SessionStoreClient(api)


class MemoryStoreClient:
    def __init__(self, api: Any):
        self._api = api

    def create(self, *, display_name: str, description: Optional[str] = None) -> ManagedMemoryStore:
        return self._store_from_response(self._api.create_memory_store(display_name, description))

    def get(
        self,
        *,
        store_id: Optional[str] = None,
        display_name: Optional[str] = None,
        create_if_not_exists: bool = False,
        description: Optional[str] = None,
    ) -> ManagedMemoryStore:
        if (store_id is None) == (display_name is None):
            raise ValueError("exactly one of store_id and display_name is required")
        if create_if_not_exists and store_id is not None:
            raise ValueError("create_if_not_exists requires display_name")
        if store_id is not None:
            return self._store_from_response(self._api.get_memory_store(store_id))

        for store in self._list_stores():
            if store.display_name == display_name:
                return store
        if create_if_not_exists:
            return self.create(display_name=display_name, description=description)
        raise KeyError(f"managed memory store not found: {display_name}")

    def _list_stores(self) -> list[ManagedMemoryStore]:
        stores = []
        page_token = None
        while True:
            response = self._api.list_memory_stores(page_token=page_token)
            stores.extend(
                self._store_from_response(store)
                for store in response.get("managed_memory_stores", [])
            )
            page_token = response.get("next_page_token")
            if not page_token:
                return stores

    def add(
        self,
        store: ManagedMemoryStore,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        return self._entry_from_response(
            self._api.create_memory_entry(
                store.store_id,
                actor_id,
                path,
                content=content,
                description=description,
                session_id=session_id,
                source_type=source_type,
            )
        )

    def list(
        self,
        store: ManagedMemoryStore,
        *,
        actor_id: str,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> list[ManagedMemoryEntry]:
        entries = []
        page_token = None
        while True:
            response = self._api.list_memory_entries(
                store.store_id,
                actor_id,
                path_prefix=path_prefix,
                session_id=session_id,
                page_token=page_token,
            )
            entries.extend(
                self._entry_from_response(entry)
                for entry in response.get("managed_memory_entries", [])
            )
            page_token = response.get("next_page_token")
            if not page_token:
                return entries

    def get_entry(self, store: ManagedMemoryStore, *, entry_id: str) -> ManagedMemoryEntry:
        return self._entry_from_response(self._api.get_memory_entry(store.store_id, entry_id))

    def search(
        self,
        store: ManagedMemoryStore,
        *,
        actor_id: str,
        query: str,
        limit: Optional[int] = None,
    ) -> list[ManagedMemoryEntry]:
        # Search is an unpaginated relevance-ranked top-N; ``limit`` (1-100) caps how many
        # results come back. Default to the max (100) when the caller omits it.
        effective_limit = 100 if limit is None else limit
        _validate_page_size(effective_limit)
        response = self._api.search_memory_entries(
            store.store_id, actor_id, query, limit=effective_limit
        )
        # Each result carries the entry under ``managed_memory_entry`` (plus a relevance score);
        # fall back to the deprecated top-level ``managed_memory_entries`` alias for older servers.
        results = response.get("results")
        if results is not None:
            return [self._entry_from_response(result["managed_memory_entry"]) for result in results]
        return [
            self._entry_from_response(entry) for entry in response.get("managed_memory_entries", [])
        ]

    def delete(self, store: ManagedMemoryStore, *, entry_id: str) -> None:
        self._api.delete_memory_entry(store.store_id, entry_id)

    def append(
        self,
        store: ManagedMemoryStore,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        # Non-atomic client-side read-modify-write: concurrent appends can overwrite each other.
        matches = self.list(store, actor_id=actor_id, session_id=session_id, path_prefix=path)
        exact_match = next(
            (
                entry
                for entry in matches
                if entry.actor_id == actor_id
                and entry.session_id == session_id
                and entry.path == path
            ),
            None,
        )
        if exact_match is None:
            return self.add(
                store,
                actor_id=actor_id,
                path=path,
                content=content,
                session_id=session_id,
                description=description,
                source_type=source_type,
            )

        current = self.get_entry(store, entry_id=exact_match.entry_id)
        updated_description = current.description if description is None else description
        return self._entry_from_response(
            self._api.update_memory_entry(
                store.store_id,
                current.entry_id,
                content=(current.content or "") + content,
                description=updated_description,
            )
        )

    def _store_from_response(self, response: dict[str, Any]) -> ManagedMemoryStore:
        return ManagedMemoryStore(
            name=response["name"],
            display_name=response["display_name"],
            workspace_id=response.get("workspace_id"),
            storage_backend=response.get("storage_backend"),
            owner_user_id=response.get("owner_user_id"),
            # Memory stores serialize timestamps as epoch-millis ``created_at``/``updated_at``;
            # tolerate the RFC 3339 ``create_time``/``update_time`` form too.
            create_time=parse_timestamp(response.get("created_at") or response.get("create_time")),
            update_time=parse_timestamp(response.get("updated_at") or response.get("update_time")),
            description=response.get("description"),
            _client=self,
        )

    @staticmethod
    def _entry_from_response(response: dict[str, Any]) -> ManagedMemoryEntry:
        return ManagedMemoryEntry(
            name=response["name"],
            actor_id=response["actor_id"],
            session_id=response.get("session_id"),
            path=response["path"],
            content=response.get("content"),
            description=response.get("description"),
            source_type=response.get("source_type"),
            create_time=parse_timestamp(response.get("create_time")),
            update_time=parse_timestamp(response.get("update_time")),
        )


class SessionStoreClient:
    def __init__(self, api: Any):
        self._api = api

    def set_session_store(self, session_store_name: str) -> SessionStore:
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
        return self._store_from_response(
            self._api.create_session_store(session_store_name, description, metadata)
        )

    def list(self, *, page_size: Optional[int] = None) -> list[SessionStore]:
        _validate_page_size(page_size)
        stores = []
        page_token = None
        while True:
            response = self._api.list_session_stores(page_size=page_size, page_token=page_token)
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
        return self._store_from_response(
            self._api.update_session_store(
                store.session_store_name, description=description, metadata=metadata
            )
        )

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
        return self._session_from_response(
            self._api.create_session(
                name,
                actor_id,
                session_id=session_id,
                parent_session_id=parent_session_id,
                metadata=metadata,
            ),
            name,
        )

    def list_sessions(
        self,
        store: Optional[SessionStore] = None,
        *,
        session_store_name: Optional[str] = None,
        page_size: Optional[int] = None,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> list[Session]:
        _validate_page_size(page_size)
        name = self._resolve_session_store_name(store, session_store_name)
        # This helper consumes every page, so it defaults to create_time ordering: continuation is
        # exactly-once under the immutable create_time, whereas the service's own last_activity_time
        # default can repeat or skip sessions whose activity changes between page fetches.
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
        return self._session_from_response(self._api.get_session(session_id, name), name)

    def update_session(self, session: Session, *, metadata: dict[str, str]) -> Session:
        return self._session_from_response(
            self._api.update_session(session.session_store_name, session.session_id, metadata),
            session.session_store_name,
        )

    def delete_session(self, session: Session, *, force: bool = False) -> None:
        self._api.delete_session(session.session_store_name, session.session_id, force=force)

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
            response.get("session", response), session.session_store_name
        )

    def list_items(
        self,
        session: Session,
        *,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> SessionItemPage:
        _validate_page_size(page_size)
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

    def append_items(self, session: Session, *, items: list[Any]) -> list[SessionItem]:
        if not items:
            raise ValueError("at least one item is required")
        response = self._api.append_session_items(
            session.session_store_name, session.session_id, list(items)
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
        store: Optional[SessionStore], session_store_name: Optional[str]
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

    def _session_from_response(self, response: dict[str, Any], session_store_name: str) -> Session:
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
