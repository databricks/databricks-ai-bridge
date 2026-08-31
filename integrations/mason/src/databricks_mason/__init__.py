"""High-level Python client and framework-neutral runtime helpers for Mason."""

from typing import TYPE_CHECKING

from databricks_mason.client import MasonClient
from databricks_mason.memory_store import Memory, MemorySearchResult, MemoryStore
from databricks_mason.session_store import Session, SessionItem, SessionStore

if TYPE_CHECKING:
    from databricks_mason.runtime import (
        configure_tracing,
        tag_session,
        workspace_client,
        workspace_headers,
    )

__all__ = [
    "MasonClient",
    "Memory",
    "MemorySearchResult",
    "MemoryStore",
    "Session",
    "SessionItem",
    "SessionStore",
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
