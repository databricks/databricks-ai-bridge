"""Lazy compatibility exports for :mod:`databricks_agentbricks.runtime`."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_agentbricks.runtime import (
        AgentApp,
        AuthError,
        DurableAgentServer,
        InvocationAuthPolicy,
        InvocationContext,
        RequestAuthContext,
        configure_tracing,
        list_ai_gateway_model_services,
        start_trace,
        workspace_client,
        workspace_headers,
    )

__all__ = [
    "DurableAgentServer",
    "AgentApp",
    "InvocationContext",
    "AuthError",
    "InvocationAuthPolicy",
    "RequestAuthContext",
    "configure_tracing",
    "start_trace",
    "workspace_client",
    "workspace_headers",
    "list_ai_gateway_model_services",
]

_MODULE_BY_NAME = {
    "DurableAgentServer": "databricks_agentbricks.runtime",
    "AgentApp": "databricks_agentbricks.runtime",
    "InvocationContext": "databricks_agentbricks.runtime",
    "AuthError": "databricks_agentbricks.runtime",
    "InvocationAuthPolicy": "databricks_agentbricks.runtime",
    "RequestAuthContext": "databricks_agentbricks.runtime",
    "configure_tracing": "databricks_agentbricks.runtime",
    "start_trace": "databricks_agentbricks.runtime",
    "workspace_client": "databricks_agentbricks.runtime",
    "workspace_headers": "databricks_agentbricks.runtime",
    "list_ai_gateway_model_services": "databricks_agentbricks.runtime",
}

_SUBMODULES = frozenset(
    {
        "app",
        "auth",
        "durability",
        "execution",
        "genie",
        "mcp_auth",
        "model_services",
        "runtime",
        "session_store_client",
        "store",
        "tool_manifest",
        "tracing",
        "types",
        "workspace",
    }
)


def __getattr__(name: str) -> object:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is not None:
        import importlib

        return getattr(importlib.import_module(module_name), name)
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
