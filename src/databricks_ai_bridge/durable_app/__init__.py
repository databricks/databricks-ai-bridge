"""AgentCore-style durable entrypoint hosted by Databricks AI Bridge."""

try:
    import fastapi  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "DatabricksDurableApp requires databricks-ai-bridge[agent-server]. "
        "Install it with: pip install databricks-ai-bridge[agent-server]"
    ) from exc

from databricks_ai_bridge.durable_app.app import (
    RUN_ID_HEADER,
    SESSION_ID_HEADER,
    DatabricksDurableApp,
    DurableAgentContext,
    DurableAgentEntrypoint,
)

__all__ = [
    "DatabricksDurableApp",
    "DurableAgentContext",
    "DurableAgentEntrypoint",
    "RUN_ID_HEADER",
    "SESSION_ID_HEADER",
]
