"""High-level Python client and framework-neutral runtime helpers for Agent Bricks."""

from typing import TYPE_CHECKING

from databricks_agentbricks.client import AgentBricksClient, AgentKitClient
from databricks_agentbricks.memory_store import Memory, MemorySearchResult, MemoryStore
from databricks_agentbricks.session_store import (
    ExtractedMemory,
    Session,
    SessionItem,
    SessionStore,
)

if TYPE_CHECKING:
    from databricks_agentbricks.runtime import (
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
    "AgentBricksClient",
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

        return getattr(importlib.import_module("databricks_agentbricks.runtime"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
