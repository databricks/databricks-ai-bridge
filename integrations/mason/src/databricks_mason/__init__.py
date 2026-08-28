"""Databricks integration for Mason.

Mason ships a `mason` CLI and this Python SDK over the same agents/v1 preview APIs.
Construct a `MasonClient` and call one method per API operation:

    from databricks_mason import MasonClient

    client = MasonClient(profile="my-workspace")
    client.create_memory_store("my-store")
    for entry in client.list_memory_entries("my-store", actor_id="alice").get("entries", []):
        ...

Auth resolves through the Databricks SDK: pass a `.databrickscfg` profile or rely on
its default resolution. API errors surface as `AgentCliError`.
"""

from databricks_mason.client import MasonClient, memory_entry_path, memory_store_path
from databricks_mason.errors import AgentCliError
from databricks_mason.models import (
    MemoryEntry,
    MemoryEntryList,
    MemorySearchHit,
    MemorySearchResult,
    MemoryStore,
    MemoryStoreList,
    PoppedSessionItem,
    Session,
    SessionItem,
    SessionItemList,
    SessionList,
    SessionStore,
    SessionStoreList,
    StorageBackend,
)

__all__ = [
    "MasonClient",
    "AgentCliError",
    "memory_store_path",
    "memory_entry_path",
    "MemoryStore",
    "MemoryStoreList",
    "MemoryEntry",
    "MemoryEntryList",
    "MemorySearchResult",
    "MemorySearchHit",
    "StorageBackend",
    "SessionStore",
    "SessionStoreList",
    "Session",
    "SessionList",
    "SessionItem",
    "SessionItemList",
    "PoppedSessionItem",
]
