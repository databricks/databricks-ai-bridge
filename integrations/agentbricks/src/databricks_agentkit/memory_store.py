"""Typed managed-memory resources and operations."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit._pagination import validate_limit, validate_page_size
from databricks_agentkit._permissions import MAX_PERMISSION_PAGE_SIZE, StorePermission
from databricks_agentkit.timefmt import parse_timestamp

if TYPE_CHECKING:
    from databricks_agentkit._api_client import _AgentBricksApiClient

_WRITE_MODE_PREFIX = "MANAGED_MEMORY_ENTRY_WRITE_MODE_"
_WRITE_MODES = frozenset({"EXCLUSIVE", "OVERWRITE", "APPEND"})


def _resource_id(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _write_mode(value: str) -> str:
    """Normalize a short (``APPEND``) or full enum (``MANAGED_..._APPEND``) write mode."""
    mode = value.upper().removeprefix(_WRITE_MODE_PREFIX)
    # The server treated an unspecified mode as an exclusive create.
    if mode == "UNSPECIFIED":
        return "EXCLUSIVE"
    if mode not in _WRITE_MODES:
        raise ValueError(f"unsupported write_mode {value!r}; use EXCLUSIVE, OVERWRITE, or APPEND")
    return mode


def _has_lines(content: str, lines: str) -> bool:
    """Whether ``lines`` appears in ``content`` as whole lines, not as part of a longer line."""
    return f"\n{lines}\n" in f"\n{content}\n"


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

    def add(
        self,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
        write_mode: Optional[str] = None,
    ) -> Memory:
        """Create a memory, failing with ``ALREADY_EXISTS`` if one exists at the same
        ``(actor_id, session_id, path)``.

        ``write_mode`` is deprecated. Use :meth:`put` to create or update a memory at a path.
        """
        return self._client._add_memory(
            self,
            actor_id=actor_id,
            path=path,
            content=content,
            session_id=session_id,
            description=description,
            source_type=source_type,
            write_mode=write_mode,
        )

    def put(
        self,
        *,
        actor_id: str,
        path: str,
        content: str,
        description: Optional[str] = None,
        session_id: Optional[str] = None,
        append: bool = False,
    ) -> Memory:
        """Create the memory at ``(actor_id, session_id, path)``, or update it if one exists.

        An existing memory's content is replaced, or with ``append`` gains ``content`` as new lines
        unless those lines are already present, so a retried append is a no-op. An omitted
        ``description`` keeps the existing one.

        The API has no atomic upsert, so this creates and, on conflict, looks up and updates:
        concurrent writers to the same path can overwrite each other, and an append is a
        read-modify-write.
        """
        return self._client._put_memory(
            self,
            actor_id=actor_id,
            path=path,
            content=content,
            description=description,
            session_id=session_id,
            append=append,
        )

    def list(
        self,
        *,
        actor_id: str,
        path_prefix: Optional[str] = None,
        path: Optional[str] = None,
        session_id: Optional[str] = None,
        page_size: Optional[int] = None,
        read_mask: Optional[str] = None,
    ) -> Iterator[Memory]:
        """List an actor's memories.

        ``path`` matches one exact path in every session: the API ignores ``session_id`` alongside
        it, and it can't be combined with ``path_prefix``.
        """
        return self._client._list_memories(
            self,
            actor_id=actor_id,
            path_prefix=path_prefix,
            path=path,
            session_id=session_id,
            page_size=page_size,
            read_mask=read_mask,
        )

    def get(
        self,
        memory_id: str,
        *,
        read_mask: Optional[str] = None,
    ) -> Memory:
        return self._client._get_memory(self, memory_id=memory_id, read_mask=read_mask)

    def search(
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

    def grant_permission(
        self,
        principal_id: str,
        *,
        permission: str = "WRITE",
        principal_type: str = "SERVICE_PRINCIPAL",
    ) -> None:
        """Grant READ/WRITE as the store creator or a workspace admin.

        ``principal_id`` is an application (client) ID by default, or a user email with
        ``principal_type="USER"``. WRITE includes READ; granting READ never downgrades WRITE.
        """
        self._client._grant_permission(
            self, principal_id, permission=permission, principal_type=principal_type
        )

    def get_permission(self) -> StorePermission:
        """Return the authenticated caller's effective permission.

        PERMISSION_LEVEL_UNSPECIFIED means the caller has no access.
        """
        return self._client._get_permission(self)

    def list_permissions(self, *, page_size: Optional[int] = None) -> Iterator[StorePermission]:
        """Iterate direct grants as the creator or a workspace admin, following all pages.

        Excludes implicit ownership and inherited access. ``page_size`` may be 1–1000.
        """
        return self._client._list_permissions(self, page_size=page_size)

    def revoke_permission(
        self, principal_id: str, *, principal_type: str = "SERVICE_PRINCIPAL"
    ) -> None:
        """Revoke both READ and WRITE as the store creator or a workspace admin.

        ``principal_id`` is an application (client) ID by default, or a user email with
        ``principal_type="USER"``. Repeated revocations succeed; ownership cannot be revoked.
        """
        self._client._revoke_permission(self, principal_id, principal_type=principal_type)


class MemoryStores:
    def __init__(self, api: _AgentBricksApiClient):
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

    def _handle(self, store_id: str) -> MemoryStore:
        """A store addressed by id alone, for callers that skip the GET; only its id is populated."""
        store_id = store_id.removeprefix("memory-stores/")
        return MemoryStore(name=f"memory-stores/{store_id}", display_name=store_id, _client=self)

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

    def _grant_permission(
        self, store: MemoryStore, principal_id: str, *, permission: str, principal_type: str
    ) -> None:
        self._api.grant_memory_store_permission(
            store.id, principal_id, permission=permission, principal_type=principal_type
        )

    def _get_permission(self, store: MemoryStore) -> StorePermission:
        return StorePermission._from_response(self._api.get_memory_store_permission(store.id))

    def _list_permissions(
        self, store: MemoryStore, *, page_size: Optional[int]
    ) -> Iterator[StorePermission]:
        validate_page_size(page_size, maximum=MAX_PERMISSION_PAGE_SIZE)
        page_token = None
        while True:
            response = self._api.list_memory_store_permissions(
                store.id, page_size=page_size, page_token=page_token
            )
            for permission in response.get("memory_store_permissions", []):
                yield StorePermission._from_response(permission)
            page_token = response.get("next_page_token")
            if not page_token:
                return

    def _revoke_permission(
        self, store: MemoryStore, principal_id: str, *, principal_type: str
    ) -> None:
        self._api.revoke_memory_store_permission(
            store.id, principal_id, principal_type=principal_type
        )

    def _add_memory(
        self,
        store: MemoryStore,
        *,
        actor_id: str,
        path: str,
        content: str,
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        source_type: Optional[str] = None,
        write_mode: Optional[str] = None,
    ) -> Memory:
        if write_mode is None:
            return self._create_memory(
                store,
                actor_id=actor_id,
                path=path,
                content=content,
                session_id=session_id,
                description=description,
                source_type=source_type,
            )
        warnings.warn(
            "add(write_mode=...) is deprecated. Use put() to create or update a memory, or "
            "put(append=True) to append to it.",
            DeprecationWarning,
            stacklevel=3,
        )
        mode = _write_mode(write_mode)
        if mode == "EXCLUSIVE":
            return self._create_memory(
                store,
                actor_id=actor_id,
                path=path,
                content=content,
                session_id=session_id,
                description=description,
                source_type=source_type,
            )
        return self._put_memory(
            store,
            actor_id=actor_id,
            path=path,
            content=content,
            description=description,
            session_id=session_id,
            append=mode == "APPEND",
            source_type=source_type,
        )

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

    def _put_memory(
        self,
        store: MemoryStore,
        *,
        actor_id: str,
        path: str,
        content: str,
        description: Optional[str] = None,
        session_id: Optional[str] = None,
        append: bool = False,
        source_type: Optional[str] = None,
    ) -> Memory:
        # A conflicting entry can disappear before the lookup sees it, so create once more before
        # giving up.
        for _ in range(2):
            try:
                return self._create_memory(
                    store,
                    actor_id=actor_id,
                    path=path,
                    content=content,
                    session_id=session_id,
                    description=description,
                    source_type=source_type,
                )
            except AgentCliError as exc:
                if exc.error_code != "ALREADY_EXISTS":
                    raise
            existing = self._find_memory(store, actor_id=actor_id, path=path, session_id=session_id)
            if existing is not None:
                return self._overwrite_or_append(
                    store, existing, content=content, description=description, append=append
                )
        raise AgentCliError(
            f"Could not write memory {path!r} for actor {actor_id!r}: it already exists but was "
            "not found on lookup.",
            error_code="ABORTED",
            hint="Another writer is changing this memory concurrently; retry the write.",
        )

    def _find_memory(
        self,
        store: MemoryStore,
        *,
        actor_id: str,
        path: str,
        session_id: Optional[str],
    ) -> Optional[Memory]:
        # The exact-path filter matches every session, so pick the requested one here. Recheck the
        # path too: a server predating the filter ignores it and returns all of the actor's memories.
        for memory in self._list_memories(store, actor_id=actor_id, path=path):
            if memory.path == path and memory.session_id == session_id:
                return memory
        return None

    def _overwrite_or_append(
        self,
        store: MemoryStore,
        memory: Memory,
        *,
        content: str,
        description: Optional[str],
        append: bool,
    ) -> Memory:
        if append:
            current = memory.content or ""
            if not content or _has_lines(current, content):
                return memory
            # Matches the server's APPEND: newline-joined, and empty content takes the new text.
            content = f"{current}\n{content}" if current else content
        return self._update_memory(
            store, memory_id=memory.id, content=content, description=description
        )

    def _list_memories(
        self,
        store: MemoryStore,
        *,
        actor_id: str,
        path_prefix: Optional[str] = None,
        path: Optional[str] = None,
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
                path=path,
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
