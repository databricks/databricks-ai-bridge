"""Lazy compatibility exports for :mod:`databricks_mason.langgraph`."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_mason.langgraph import (
        configure_tracing,
        genie_tools,
        mcp_tools,
        memory_tools,
        start_trace,
        workspace_client,
        workspace_headers,
    )

__all__ = [
    "genie_tools",
    "mcp_tools",
    "memory_tools",
    "checkpointer",
    "thread_config",
    "configure_tracing",
    "start_trace",
    "workspace_client",
    "workspace_headers",
]

_MODULE_BY_NAME = {
    "genie_tools": "databricks_mason.langgraph",
    "mcp_tools": "databricks_mason.langgraph",
    "memory_tools": "databricks_mason.langgraph",
    "checkpointer": "databricks_mason.langgraph",
    "thread_config": "databricks_mason.langgraph",
    "configure_tracing": "databricks_mason.langgraph",
    "start_trace": "databricks_mason.langgraph",
    "workspace_client": "databricks_mason.langgraph",
    "workspace_headers": "databricks_mason.langgraph",
}

_SUBMODULES = frozenset({"genie", "mcp", "memory", "session_store"})


def __getattr__(name: str) -> object:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is not None:
        import importlib

        return getattr(importlib.import_module(module_name), name)
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
