"""
**Re-exported Unity Catalog Utilities**

This module re-exports selected utilities from the Unity Catalog open source package.

Available aliases:

- :class:`databricks_openai.UCFunctionToolkit`
- :class:`databricks_openai.DatabricksFunctionClient`
- :func:`databricks_openai.set_uc_function_client`

Refer to the Unity Catalog `documentation <https://docs.unitycatalog.io/ai/integrations/openai/#using-unity-catalog-ai-with-the-openai-sdk>`_ for more information.
"""

import os

from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

from databricks_openai.mcp_server_toolkit import McpServerToolkit, ToolInfo
from databricks_openai.utils.clients import AsyncDatabricksOpenAI, DatabricksOpenAI
from databricks_openai.uc_volume_tool import UCVolumeTool
from databricks_openai.vector_search_retriever_tool import VectorSearchRetrieverTool

# Disable the OpenAI Agents SDK's built-in tracer by default since databricks_openai
# uses MLflow for tracing. Without this, users hit multiprocessing errors when the
# agents tracer runs alongside MLflow tracing.
# Set ENABLE_OPENAI_AGENTS_TRACING=true to keep the agents tracer.
try:
    from agents.tracing import set_trace_processors

    if os.environ.get("ENABLE_OPENAI_AGENTS_TRACING", "").lower() != "true":
        set_trace_processors([])
except ImportError:
    pass

# Expose all integrations to users under databricks-openai
__all__ = [
    "UCVolumeTool",
    "VectorSearchRetrieverTool",
    "UCFunctionToolkit",
    "DatabricksFunctionClient",
    "set_uc_function_client",
    "DatabricksOpenAI",
    "AsyncDatabricksOpenAI",
    "McpServerToolkit",
    "ToolInfo",
]
