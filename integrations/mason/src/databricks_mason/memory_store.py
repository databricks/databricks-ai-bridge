"""Typed managed-memory resources and operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from databricks_mason._pagination import validate_page_size
from databricks_mason.errors import AgentCliError
from databricks_mason.timefmt import parse_timestamp

if TYPE_CHECKING:
    from databricks_mason.client import MasonClient


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
class ManagedMemoryEntrySearchResult:
    managed_memory_entry: ManagedMemoryEntry
    score: Optional[float] = None


@dataclass(frozen=True, kw_only=True)
class ManagedMemoryStore:
    name: str
    display_name: str
    workspace_id: Optional[int] = None
    storage_backend: Optional[dict[str, Any]] = None
    owner_user_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    description: Optional[str] = None
    _client: MemoryStoreClient = field(repr=False, compare=False)

    @property
    def store_id(self) -> str:
        return _resource_id(self.name)

    def update(
        self,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ManagedMemoryStore:
        return self._client.update(self, display_name=display_name, description=description)

    def delete(self) -> None:
        self._client.delete(self)

    def create_entry(
        self,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        return self._client.create_entry(
            self,
            actor_id=actor_id,
            path=path,
            content=content,
            session_id=session_id,
            description=description,
            source_type=source_type,
        )

    def list_entries(
        self,
        *,
        actor_id: str,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        page_size: Optional[int] = None,
        read_mask: Optional[str] = None,
    ) -> List[ManagedMemoryEntry]:
        return self._client.list_entries(
            self,
            actor_id=actor_id,
            path_prefix=path_prefix,
            session_id=session_id,
            page_size=page_size,
            read_mask=read_mask,
        )

    def get_entry(
        self,
        *,
        entry_id: str,
        read_mask: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        return self._client.get_entry(self, entry_id=entry_id, read_mask=read_mask)

    def search_entries(
        self,
        *,
        actor_id: str,
        query: str,
        page_size: Optional[int] = None,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        read_mask: Optional[str] = None,
    ) -> List[ManagedMemoryEntrySearchResult]:
        return self._client.search_entries(
            self,
            actor_id=actor_id,
            query=query,
            page_size=page_size,
            path_prefix=path_prefix,
            session_id=session_id,
            read_mask=read_mask,
        )

    def update_entry(
        self,
        *,
        entry_id: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        return self._client.update_entry(
            self,
            entry_id=entry_id,
            content=content,
            description=description,
        )

    def delete_entry(self, *, entry_id: str) -> None:
        self._client.delete_entry(self, entry_id=entry_id)

    def append_entry_content(
        self,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        """Append through a non-atomic client-side read-modify-write operation."""
        return self._client.append_entry_content(
            self,
            actor_id=actor_id,
            path=path,
            content=content,
            session_id=session_id,
            description=description,
            source_type=source_type,
        )


class MemoryStoreClient:
    def __init__(self, api: MasonClient):
        self._api = api

    def create(
        self,
        *,
        display_name: str,
        description: Optional[str] = None,
    ) -> ManagedMemoryStore:
        response = self._api.create_memory_store(display_name, description)
        return self._store_from_response(response)

    def list(self, *, page_size: Optional[int] = None) -> List[ManagedMemoryStore]:
        validate_page_size(page_size)
        stores = []
        page_token = None
        while True:
            response = self._api.list_memory_stores(
                page_size=page_size,
                page_token=page_token,
            )
            stores.extend(
                self._store_from_response(store)
                for store in response.get("managed_memory_stores", [])
            )
            page_token = response.get("next_page_token")
            if not page_token:
                return stores

    def get(
        self,
        *,
        store_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> ManagedMemoryStore:
        if (store_id is None) == (display_name is None):
            raise ValueError("exactly one of store_id and display_name is required")
        if store_id is not None:
            return self._store_from_response(self._api.get_memory_store(store_id))

        assert display_name is not None
        for store in self.list():
            if store.display_name == display_name:
                return store
        raise KeyError(f"managed memory store not found: {display_name}")

    def get_or_create(
        self,
        *,
        display_name: str,
        description: Optional[str] = None,
    ) -> ManagedMemoryStore:
        try:
            return self.get(display_name=display_name)
        except KeyError:
            try:
                return self.create(display_name=display_name, description=description)
            except AgentCliError as error:
                if error.error_code != "ALREADY_EXISTS":
                    raise
                return self.get(display_name=display_name)

    def update(
        self,
        store: ManagedMemoryStore,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ManagedMemoryStore:
        if display_name is None and description is None:
            raise ValueError("at least one of display_name and description is required")
        response = self._api.update_memory_store(
            store.store_id,
            display_name=display_name,
            description=description,
        )
        return self._store_from_response(response)

    def delete(self, store: ManagedMemoryStore) -> None:
        self._api.delete_memory_store(store.store_id)

    def create_entry(
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
        response = self._api.create_memory_entry(
            store.store_id,
            actor_id,
            path,
            content=content,
            description=description,
            session_id=session_id,
            source_type=source_type,
        )
        return self._entry_from_response(response)

    def list_entries(
        self,
        store: ManagedMemoryStore,
        *,
        actor_id: str,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        page_size: Optional[int] = None,
        read_mask: Optional[str] = None,
    ) -> List[ManagedMemoryEntry]:
        validate_page_size(page_size)
        entries = []
        page_token = None
        while True:
            response = self._api.list_memory_entries(
                store.store_id,
                actor_id,
                path_prefix=path_prefix,
                session_id=session_id,
                page_size=page_size,
                page_token=page_token,
                read_mask=read_mask,
            )
            entries.extend(
                self._entry_from_response(entry)
                for entry in response.get("managed_memory_entries", [])
            )
            page_token = response.get("next_page_token")
            if not page_token:
                return entries

    def get_entry(
        self,
        store: ManagedMemoryStore,
        *,
        entry_id: str,
        read_mask: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        response = self._api.get_memory_entry(
            store.store_id,
            entry_id,
            read_mask=read_mask,
        )
        return self._entry_from_response(response)

    def search_entries(
        self,
        store: ManagedMemoryStore,
        *,
        actor_id: str,
        query: str,
        page_size: Optional[int] = None,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        read_mask: Optional[str] = None,
    ) -> List[ManagedMemoryEntrySearchResult]:
        effective_page_size = 100 if page_size is None else page_size
        validate_page_size(effective_page_size)
        response = self._api.search_memory_entries(
            store.store_id,
            actor_id,
            query,
            page_size=effective_page_size,
            path_prefix=path_prefix,
            session_id=session_id,
            read_mask=read_mask,
        )
        results = response.get("results")
        if results is not None:
            return [
                ManagedMemoryEntrySearchResult(
                    managed_memory_entry=self._entry_from_response(result["managed_memory_entry"]),
                    score=result.get("score"),
                )
                for result in results
            ]
        return [
            ManagedMemoryEntrySearchResult(managed_memory_entry=self._entry_from_response(entry))
            for entry in response.get("managed_memory_entries", [])
        ]

    def update_entry(
        self,
        store: ManagedMemoryStore,
        *,
        entry_id: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ManagedMemoryEntry:
        if content is None and description is None:
            raise ValueError("at least one of content and description is required")
        response = self._api.update_memory_entry(
            store.store_id,
            entry_id,
            content=content,
            description=description,
        )
        return self._entry_from_response(response)

    def delete_entry(self, store: ManagedMemoryStore, *, entry_id: str) -> None:
        self._api.delete_memory_entry(store.store_id, entry_id)

    def append_entry_content(
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
        matches = self.list_entries(
            store,
            actor_id=actor_id,
            session_id=session_id,
            path_prefix=path,
        )
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
            return self.create_entry(
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
        return self.update_entry(
            store,
            entry_id=current.entry_id,
            content=(current.content or "") + content,
            description=updated_description,
        )

    def _store_from_response(self, response: dict[str, Any]) -> ManagedMemoryStore:
        return ManagedMemoryStore(
            name=response["name"],
            display_name=response["display_name"],
            workspace_id=response.get("workspace_id"),
            storage_backend=response.get("storage_backend"),
            owner_user_id=response.get("owner_user_id"),
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
