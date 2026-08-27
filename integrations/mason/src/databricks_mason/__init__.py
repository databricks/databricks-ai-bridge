"""Databricks integration for Mason.

The framework-neutral runtime helpers (``configure_tracing``, ``tag_session``,
``workspace_client``, ``workspace_headers``) are resolved lazily so importing
``databricks_mason`` does not pull in the tracing module's ``mlflow`` dependency.
"""

from typing import TYPE_CHECKING

from databricks_mason.client import DatabricksAgentClient
from databricks_mason.memory_store import (
    ManagedMemoryEntry,
    ManagedMemoryEntrySearchResult,
    ManagedMemoryStore,
)
from databricks_mason.session_store import Session, SessionItem, SessionItemPage, SessionStore

if TYPE_CHECKING:
    from databricks_mason.runtime import (
        configure_tracing,
        tag_session,
        workspace_client,
        workspace_headers,
    )

__all__ = [
    "DatabricksAgentClient",
    "ManagedMemoryEntry",
    "ManagedMemoryEntrySearchResult",
    "ManagedMemoryStore",
    "Session",
    "SessionItem",
    "SessionItemPage",
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
