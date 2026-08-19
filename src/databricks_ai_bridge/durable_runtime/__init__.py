"""Lakebase-backed durable execution for JSON request/response handlers.

Install the memory extra before importing this module::

    pip install databricks-ai-bridge[memory]
"""

try:
    import psycopg  # noqa: F401
    import sqlalchemy  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "DatabricksDurableRuntime requires databricks-ai-bridge[memory]. "
        "Install it with: pip install databricks-ai-bridge[memory]"
    ) from exc

from databricks_ai_bridge.durable_runtime.runtime import DatabricksDurableRuntime
from databricks_ai_bridge.durable_runtime.store import (
    DEFAULT_DURABILITY_SCHEMA,
    LakebaseDurabilityStore,
)
from databricks_ai_bridge.durable_runtime.types import (
    DurabilityStore,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionNotFoundError,
    DurableExecutionStatus,
    DurableExecutor,
    DurableRequestConflictError,
    JsonObject,
)

__all__ = [
    "DEFAULT_DURABILITY_SCHEMA",
    "DatabricksDurableRuntime",
    "DurabilityStore",
    "DurableExecution",
    "DurableExecutionContext",
    "DurableExecutionFailedError",
    "DurableExecutionNotFoundError",
    "DurableExecutionStatus",
    "DurableExecutor",
    "DurableRequestConflictError",
    "JsonObject",
    "LakebaseDurabilityStore",
]
