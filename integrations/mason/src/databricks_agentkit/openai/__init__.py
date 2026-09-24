"""Lazy compatibility exports for :mod:`databricks_mason.openai`."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_mason.openai import (
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
    "genie_tools": "databricks_mason.openai",
    "mcp_servers": "databricks_mason.openai",
    "memory_tools": "databricks_mason.openai",
    "session_store": "databricks_mason.openai",
    "configure_tracing": "databricks_mason.openai",
    "start_trace": "databricks_mason.openai",
    "workspace_client": "databricks_mason.openai",
    "workspace_headers": "databricks_mason.openai",
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
