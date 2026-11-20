"""Native Microsoft Agent Framework integration, installed with ``[maf]``."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_agentkit.maf.genie import genie_tools
    from databricks_agentkit.maf.mcp import mcp_tools
    from databricks_agentkit.maf.memory import memory_tools
    from databricks_agentkit.maf.sessions import read_history, session_store
    from databricks_agentkit.runtime import start_trace


def configure_tracing() -> None:
    """Enable invocation spans and OpenAI model-call tracing when MLflow is configured."""
    import mlflow

    from databricks_agentkit.runtime import configure_tracing as configure

    configure(autolog=mlflow.openai.autolog)


__all__ = [
    "configure_tracing",
    "genie_tools",
    "mcp_tools",
    "memory_tools",
    "read_history",
    "session_store",
    "start_trace",
]

_MODULE_BY_NAME = {
    "genie_tools": "databricks_agentkit.maf.genie",
    "mcp_tools": "databricks_agentkit.maf.mcp",
    "memory_tools": "databricks_agentkit.maf.memory",
    "read_history": "databricks_agentkit.maf.sessions",
    "session_store": "databricks_agentkit.maf.sessions",
    "start_trace": "databricks_agentkit.runtime",
}


def __getattr__(name: str) -> object:
    import importlib

    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
