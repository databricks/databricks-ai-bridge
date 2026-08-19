"""MLflow AgentServer with Lakebase-backed durable execution.

Install the agent-server extra before importing this module::

    pip install databricks-ai-bridge[agent-server]
"""

try:
    import fastapi  # noqa: F401
    import mlflow.genai.agent_server  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "DatabricksDurableAgentServer requires databricks-ai-bridge[agent-server]. "
        "Install it with: pip install databricks-ai-bridge[agent-server]"
    ) from exc

from databricks_ai_bridge.durable_agent_server.server import (
    DatabricksDurableAgentServer,
    DurableRequestPreparer,
    PreparedDurableRequest,
    get_durable_execution_context,
)

__all__ = [
    "DatabricksDurableAgentServer",
    "DurableRequestPreparer",
    "PreparedDurableRequest",
    "get_durable_execution_context",
]
