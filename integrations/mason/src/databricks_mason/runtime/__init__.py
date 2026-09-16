"""Framework-neutral runtime helpers for an agent deployed on Databricks (via ``databricks-mason[runtime]``).

These have no agent-framework dependency — MLflow tracing setup and workspace-routed SDK client
construction — so they work regardless of which framework an agent is built with. Framework-specific
helpers (session-store checkpointer, MCP tools, memory tools) live in the per-framework adapter
package, e.g. :mod:`databricks_mason.langgraph`, which re-exports these for a single import point.

``__all__`` is the supported surface. ``tool_manifest`` and ``session_store_client`` are internal and
reachable by their submodule paths but not re-exported here.

The re-exports below are resolved lazily (PEP 562) so importing a neutral submodule such as
``databricks_mason.runtime.tool_manifest`` does not pull in the tracing module's ``mlflow`` dependency.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_mason.runtime.app import AgentApp
    from databricks_mason.runtime.tracing import configure_tracing, start_trace
    from databricks_mason.runtime.types import InvocationContext
    from databricks_mason.runtime.workspace import workspace_client, workspace_headers

__all__ = [
    "AgentApp",
    "InvocationContext",
    # MLflow tracing — call configure_tracing() once at startup (pass the framework's autolog, or use
    # a framework adapter that binds it). Wrap each invocation in start_trace() so a trace is recorded
    # (framework autolog only nests under an active trace); pass session_id= to group traces by session.
    "configure_tracing",
    "start_trace",
    # Workspace SDK client construction (account-host / run-local routing handled).
    "workspace_client",
    "workspace_headers",
]

_MODULE_BY_NAME = {
    "AgentApp": "app",
    "InvocationContext": "types",
    "configure_tracing": "tracing",
    "start_trace": "tracing",
    "workspace_client": "workspace",
    "workspace_headers": "workspace",
}


def __getattr__(name: str) -> object:
    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
