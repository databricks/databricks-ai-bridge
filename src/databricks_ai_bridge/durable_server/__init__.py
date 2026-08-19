"""FastAPI server for Lakebase-backed durable request execution.

Install the agent-server extra before importing this module::

    pip install databricks-ai-bridge[agent-server]
"""

try:
    import fastapi  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "DatabricksDurableServer requires databricks-ai-bridge[agent-server]. "
        "Install it with: pip install databricks-ai-bridge[agent-server]"
    ) from exc

from databricks_ai_bridge.durable_server.server import (
    DatabricksDurableServer,
    DurableRequestPreparer,
    DurableStatusResponse,
    PreparedDurableRequest,
)

__all__ = [
    "DatabricksDurableServer",
    "DurableRequestPreparer",
    "DurableStatusResponse",
    "PreparedDurableRequest",
]
