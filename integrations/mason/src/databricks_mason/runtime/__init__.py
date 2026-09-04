"""Framework-neutral runtime helpers for agents deployed with Mason.

The transport-neutral durable runtime remains internal for now. ``DurableAgentApp`` is re-exported
here and from :mod:`databricks_mason` as the supported application surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from databricks_mason.durable_server import (
        DurableAgentApp,
        DurableAgentContext,
    )
    from databricks_mason.runtime.tracing import configure_tracing, tag_session
    from databricks_mason.runtime.workspace import workspace_client, workspace_headers

_MODULE_BY_NAME = {
    "configure_tracing": "tracing",
    "tag_session": "tracing",
    "workspace_client": "workspace",
    "workspace_headers": "workspace",
}


def __getattr__(name: str) -> Any:
    if name in {"DurableAgentApp", "DurableAgentContext"}:
        from databricks_mason import durable_server

        return getattr(durable_server, name)
    if module_name := _MODULE_BY_NAME.get(name):
        from importlib import import_module

        return getattr(import_module(f"{__name__}.{module_name}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DurableAgentApp",
    "DurableAgentContext",
    "configure_tracing",
    "tag_session",
    "workspace_client",
    "workspace_headers",
]
