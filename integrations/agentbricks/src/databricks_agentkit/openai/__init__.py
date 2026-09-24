"""Lazy compatibility exports for :mod:`databricks_agentbricks.openai`."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_agentbricks.openai import (
        configure_tracing,
        genie_tools,
        mcp_servers,
        memory_tools,
        session_store,
        start_trace,
        workspace_client,
        workspace_headers,
    )

__all__ = [
    "genie_tools",
    "mcp_servers",
    "memory_tools",
    "session_store",
    "configure_tracing",
    "start_trace",
    "workspace_client",
    "workspace_headers",
]

_MODULE_BY_NAME = {
    "genie_tools": "databricks_agentbricks.openai",
    "mcp_servers": "databricks_agentbricks.openai",
    "memory_tools": "databricks_agentbricks.openai",
    "session_store": "databricks_agentbricks.openai",
    "configure_tracing": "databricks_agentbricks.openai",
    "start_trace": "databricks_agentbricks.openai",
    "workspace_client": "databricks_agentbricks.openai",
    "workspace_headers": "databricks_agentbricks.openai",
}

_SUBMODULES = frozenset({"genie", "mcp", "memory", "sessions"})


def __getattr__(name: str) -> object:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is not None:
        import importlib

        return getattr(importlib.import_module(module_name), name)
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
