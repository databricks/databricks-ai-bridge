"""High-level Python client and framework-neutral runtime helpers for Mason."""

from typing import TYPE_CHECKING

from databricks_mason.client import AgentKitClient, MasonClient
from databricks_mason.memory_store import Memory, MemorySearchResult, MemoryStore
from databricks_mason.session_store import (
    ExtractedMemory,
    Session,
    SessionItem,
    SessionStore,
)

if TYPE_CHECKING:
    from databricks_mason.runtime import (
        AgentApp,
        DurableAgentServer,
        InvocationContext,
        configure_tracing,
        list_ai_gateway_model_services,
        start_trace,
        workspace_client,
        workspace_headers,
    )

__all__ = [
    "AgentKitClient",
    "MasonClient",
    "Memory",
    "MemorySearchResult",
    "MemoryStore",
    "ExtractedMemory",
    "Session",
    "SessionItem",
    "SessionStore",
    "DurableAgentServer",
    "AgentApp",
    "InvocationContext",
    "configure_tracing",
    "start_trace",
    "workspace_client",
    "workspace_headers",
    "list_ai_gateway_model_services",
]

_RUNTIME_REEXPORTS = frozenset(
    {
        "DurableAgentServer",
        "AgentApp",
        "InvocationContext",
        "configure_tracing",
        "start_trace",
        "workspace_client",
        "workspace_headers",
        "list_ai_gateway_model_services",
    }
)


def __getattr__(name: str) -> object:
    if name in _RUNTIME_REEXPORTS:
        import importlib

        return getattr(importlib.import_module("databricks_mason.runtime"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
