"""Framework-neutral helpers and the public durable application surface.

The durable execution engine remains an internal implementation detail for now. Applications use
:class:`DurableAgentApp`, which provides the HTTP server and delegates execution to that
engine. Imports resolve lazily so callers pay only for the features they use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from databricks_mason.runtime.app import (
        DurableAgentApp,
        DurableAgentContext,
    )
    from databricks_mason.runtime.tracing import configure_tracing, tag_session
    from databricks_mason.runtime.workspace import workspace_client, workspace_headers

_MODULE_BY_NAME = {
    "DurableAgentApp": "app",
    "DurableAgentContext": "app",
    "configure_tracing": "tracing",
    "tag_session": "tracing",
    "workspace_client": "workspace",
    "workspace_headers": "workspace",
}


def __getattr__(name: str) -> Any:
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
