"""Long-running agent server with database persistence and background mode.

Requires the ``[server]`` extra::

    pip install databricks-ai-bridge[server]
"""

from databricks_ai_bridge.long_running.db import dispose_db, init_db, is_db_configured
from databricks_ai_bridge.long_running.models import Base, Message, Response
from databricks_ai_bridge.long_running.repository import (
    append_message,
    create_response,
    get_messages,
    get_response,
    update_response_status,
    update_response_trace_id,
)
from databricks_ai_bridge.long_running.server import LongRunningAgentServer
from databricks_ai_bridge.long_running.settings import LongRunningSettings

__all__ = [
    "Base",
    "LongRunningAgentServer",
    "LongRunningSettings",
    "Message",
    "Response",
    "append_message",
    "create_response",
    "dispose_db",
    "get_messages",
    "get_response",
    "init_db",
    "is_db_configured",
    "update_response_status",
    "update_response_trace_id",
]
