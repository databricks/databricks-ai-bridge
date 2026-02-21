import os

from agents.tracing import set_trace_processors

from databricks_openai.agents.mcp_server import McpServer
from databricks_openai.agents.session import AsyncDatabricksSession

# Disable the OpenAI Agents SDK's built-in tracer by default since databricks_openai
# uses MLflow for tracing. Without this, users hit multiprocessing errors when the
# agents tracer runs alongside MLflow tracing.
# Set ENABLE_OPENAI_AGENTS_TRACING=true to keep the agents tracer.
if os.environ.get("ENABLE_OPENAI_AGENTS_TRACING", "").lower() != "true":
    set_trace_processors([])

__all__ = ["AsyncDatabricksSession", "McpServer"]
