"""Long-running agent server with database persistence and background mode.

Requires the ``[agent-server]`` extra::

    pip install databricks-ai-bridge[agent-server]
"""

try:
    import fastapi  # noqa: F401
    import sqlalchemy  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Long-running server requires databricks-ai-bridge[agent-server]. "
        "Please install with: pip install databricks-ai-bridge[agent-server]"
    ) from e

from databricks_ai_bridge.long_running.db import (
    dispose_db,
    init_db,
    is_db_configured,
    session_scope,
)
from databricks_ai_bridge.long_running.models import Base, Message, Response
from databricks_ai_bridge.long_running.repository import (
    ResponseInfo,
    append_message,
    create_response,
    get_messages,
    get_response,
    update_response_status,
    update_response_trace_id,
)
from databricks_ai_bridge.long_running.server import AdvancedAgentServer
from databricks_ai_bridge.long_running.settings import LongRunningSettings

__all__ = [
    "Base",
    "AdvancedAgentServer",
    "LongRunningSettings",
    "Message",
    "Response",
    "ResponseInfo",
    "append_message",
    "create_response",
    "dispose_db",
    "get_messages",
    "get_response",
    "init_db",
    "is_db_configured",
    "session_scope",
    "update_response_status",
    "update_response_trace_id",
]
