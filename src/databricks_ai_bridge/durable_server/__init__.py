"""Generic JSON server for durable agent execution."""

try:
    import fastapi  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "DatabricksDurableServer requires databricks-ai-bridge[agent-server]. "
        "Install it with: pip install databricks-ai-bridge[agent-server]"
    ) from exc

from databricks_ai_bridge.durable_server.server import (
    DatabricksDurableServer,
    DurableRequestContext,
    DurableRequestHandler,
)

__all__ = [
    "DatabricksDurableServer",
    "DurableRequestContext",
    "DurableRequestHandler",
]
