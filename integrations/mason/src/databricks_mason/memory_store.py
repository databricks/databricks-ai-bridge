"""Typed managed-memory resources and operations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from databricks_mason._pagination import validate_limit, validate_page_size
from databricks_mason.errors import AgentCliError
from databricks_mason.timefmt import parse_timestamp

if TYPE_CHECKING:
    from databricks_mason._api_client import AgentApiClient


def _resource_id(name: str) -> str:
    return name.rsplit("/", 1)[-1]


@dataclass(frozen=True, kw_only=True)
class Memory:
    name: str
    actor_id: str
    path: str
    session_id: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    source_type: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    _store: MemoryStore = field(repr=False, compare=False)
    _client: MemoryStores = field(repr=False, compare=False)

    @property
    def id(self) -> str:
        return _resource_id(self.name)

    def update(
        self,
        *,
        content: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Memory:
        return self._client._update_memory(
            self._store,
            memory_id=self.id,
            content=content,
            description=description,
        )

    def delete(self) -> None:
        self._client._delete_memory(self._store, memory_id=self.id)


@dataclass(frozen=True)
class MemorySearchResult:
    memory: Memory
    score: Optional[float] = None


@dataclass(frozen=True, kw_only=True)
class MemoryStore:
    name: str
    display_name: str
    workspace_id: Optional[int] = None
    storage_backend: Optional[dict[str, Any]] = None
    owner_user_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    description: Optional[str] = None
    _client: MemoryStores = field(repr=False, compare=False)

    @property
    def id(self) -> str:
        return _resource_id(self.name)

    def update(
        self,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> MemoryStore:
        return self._client._update_store(
            self,
            display_name=display_name,
            description=description,
        )

    def delete(self) -> None:
        self._client._delete_store(self)

    def create_memory(
        self,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Memory:
        return self._client._create_memory(
            self,
            actor_id=actor_id,
            path=path,
            content=content,
            session_id=session_id,
            description=description,
            source_type=source_type,
        )

    def list_memories(
        self,
        *,
        actor_id: str,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        page_size: Optional[int] = None,
        read_mask: Optional[str] = None,
    ) -> Iterator[Memory]:
        return self._client._list_memories(
            self,
            actor_id=actor_id,
            path_prefix=path_prefix,
            session_id=session_id,
            page_size=page_size,
            read_mask=read_mask,
        )

    def get_memory(
        self,
        *,
        memory_id: str,
        read_mask: Optional[str] = None,
    ) -> Memory:
        return self._client._get_memory(self, memory_id=memory_id, read_mask=read_mask)

    def search_memories(
        self,
        *,
        actor_id: str,
        query: str,
        limit: Optional[int] = None,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        read_mask: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        return self._client._search_memories(
            self,
            actor_id=actor_id,
            query=query,
            limit=limit,
            path_prefix=path_prefix,
            session_id=session_id,
            read_mask=read_mask,
        )

    def append_memory(
        self,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Memory:
        """Append through a non-atomic client-side read-modify-write operation."""
        return self._client._append_memory(
            self,
            actor_id=actor_id,
            path=path,
            content=content,
            session_id=session_id,
            description=description,
            source_type=source_type,
        )


class MemoryStores:
    def __init__(self, api: AgentApiClient):
        self._api = api

    def create(
        self,
        display_name: str,
        *,
        description: Optional[str] = None,
    ) -> MemoryStore:
        response = self._api.create_memory_store(display_name, description)
        return self._store_from_response(response)

    def list(self, *, page_size: Optional[int] = None) -> Iterator[MemoryStore]:
        validate_page_size(page_size)
        page_token = None
        while True:
            response = self._api.list_memory_stores(
                page_size=page_size,
                page_token=page_token,
            )
            for store in response.get("managed_memory_stores", []):
                yield self._store_from_response(store)
            page_token = response.get("next_page_token")
            if not page_token:
                return

    def get(self, store_id: str) -> MemoryStore:
        return self._store_from_response(self._api.get_memory_store(store_id))

    def get_or_create(
        self,
        display_name: str,
        *,
        description: Optional[str] = None,
    ) -> MemoryStore:
        existing = self._find_by_display_name(display_name)
        if existing is not None:
            return existing
        try:
            return self.create(display_name, description=description)
        except AgentCliError as error:
            if error.error_code != "ALREADY_EXISTS":
                raise
            existing = self._find_by_display_name(display_name)
            if existing is None:
                raise
            return existing

    def _find_by_display_name(self, display_name: str) -> Optional[MemoryStore]:
        return next((store for store in self.list() if store.display_name == display_name), None)

    def _update_store(
        self,
        store: MemoryStore,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> MemoryStore:
        if display_name is None and description is None:
            raise ValueError("at least one of display_name and description is required")
        response = self._api.update_memory_store(
            store.id,
            display_name=display_name,
            description=description,
        )
        return self._store_from_response(response)

    def _delete_store(self, store: MemoryStore) -> None:
        self._api.delete_memory_store(store.id)

    def _create_memory(
        self,
        store: MemoryStore,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Memory:
        response = self._api.create_memory_entry(
            store.id,
            actor_id,
            path,
            content=content,
            description=description,
            session_id=session_id,
            source_type=source_type,
        )
        return self._memory_from_response(response, store)

    def _list_memories(
        self,
        store: MemoryStore,
        *,
        actor_id: str,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        page_size: Optional[int] = None,
        read_mask: Optional[str] = None,
    ) -> Iterator[Memory]:
        validate_page_size(page_size)
        page_token = None
        while True:
            response = self._api.list_memory_entries(
                store.id,
                actor_id,
                path_prefix=path_prefix,
                session_id=session_id,
                page_size=page_size,
                page_token=page_token,
                read_mask=read_mask,
            )
            for memory in response.get("managed_memory_entries", []):
                yield self._memory_from_response(memory, store)
            page_token = response.get("next_page_token")
            if not page_token:
                return

    def _get_memory(
        self,
        store: MemoryStore,
        *,
        memory_id: str,
        read_mask: Optional[str] = None,
    ) -> Memory:
        response = self._api.get_memory_entry(
            store.id,
            memory_id,
            read_mask=read_mask,
        )
        return self._memory_from_response(response, store)

    def _search_memories(
        self,
        store: MemoryStore,
        *,
        actor_id: str,
        query: str,
        limit: Optional[int] = None,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        read_mask: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        effective_limit = 100 if limit is None else limit
        validate_limit(effective_limit)
        response = self._api.search_memory_entries(
            store.id,
            actor_id,
            query,
            page_size=effective_limit,
            path_prefix=path_prefix,
            session_id=session_id,
            read_mask=read_mask,
        )
        results = response.get("results")
        if results is not None:
            return [
                MemorySearchResult(
                    memory=self._memory_from_response(result["managed_memory_entry"], store),
                    score=result.get("score"),
                )
                for result in results
            ]
        return [
            MemorySearchResult(memory=self._memory_from_response(memory, store))
            for memory in response.get("managed_memory_entries", [])
        ]

    def _update_memory(
        self,
        store: MemoryStore,
        *,
        memory_id: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Memory:
        if content is None and description is None:
            raise ValueError("at least one of content and description is required")
        response = self._api.update_memory_entry(
            store.id,
            memory_id,
            content=content,
            description=description,
        )
        return self._memory_from_response(response, store)

    def _delete_memory(self, store: MemoryStore, *, memory_id: str) -> None:
        self._api.delete_memory_entry(store.id, memory_id)

    def _append_memory(
        self,
        store: MemoryStore,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Memory:
        matches = self._list_memories(
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
            return self._create_memory(
                store,
                actor_id=actor_id,
                path=path,
                content=content,
                session_id=session_id,
                description=description,
                source_type=source_type,
            )

        current = self._get_memory(store, memory_id=exact_match.id)
        updated_description = current.description if description is None else description
        return self._update_memory(
            store,
            memory_id=current.id,
            content=(current.content or "") + content,
            description=updated_description,
        )

    def _store_from_response(self, response: dict[str, Any]) -> MemoryStore:
        return MemoryStore(
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

    def _memory_from_response(self, response: dict[str, Any], store: MemoryStore) -> Memory:
        return Memory(
            name=response["name"],
            actor_id=response["actor_id"],
            session_id=response.get("session_id"),
            path=response["path"],
            content=response.get("content"),
            description=response.get("description"),
            source_type=response.get("source_type"),
            create_time=parse_timestamp(response.get("create_time")),
            update_time=parse_timestamp(response.get("update_time")),
            _store=store,
            _client=self,
        )
