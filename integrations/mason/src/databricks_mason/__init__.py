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

The framework-neutral runtime helpers are resolved lazily so importing this package
does not pull in the tracing module's ``mlflow`` dependency.
"""

from typing import TYPE_CHECKING

from databricks_mason.client import MasonClient, memory_entry_path, memory_store_path
from databricks_mason.errors import AgentCliError
from databricks_mason.memory_store import (
    ManagedMemoryEntry,
    ManagedMemoryEntrySearchResult,
    ManagedMemoryStore,
)
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
from databricks_mason.session_store import (
    Session as BoundSession,
)
from databricks_mason.session_store import (
    SessionItem as BoundSessionItem,
)
from databricks_mason.session_store import (
    SessionItemPage,
)
from databricks_mason.session_store import (
    SessionStore as BoundSessionStore,
)

if TYPE_CHECKING:
    from databricks_mason.runtime import (
        configure_tracing,
        tag_session,
        workspace_client,
        workspace_headers,
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
    "ManagedMemoryEntry",
    "ManagedMemoryEntrySearchResult",
    "ManagedMemoryStore",
    "BoundSession",
    "BoundSessionItem",
    "SessionItemPage",
    "BoundSessionStore",
    "configure_tracing",
    "tag_session",
    "workspace_client",
    "workspace_headers",
]

_RUNTIME_REEXPORTS = frozenset(
    {"configure_tracing", "tag_session", "workspace_client", "workspace_headers"}
)


def __getattr__(name: str) -> object:
    if name in _RUNTIME_REEXPORTS:
        import importlib

        return getattr(importlib.import_module("databricks_mason.runtime"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
