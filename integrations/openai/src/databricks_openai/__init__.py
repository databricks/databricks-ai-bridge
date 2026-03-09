"""
**Re-exported Unity Catalog Utilities**

This module re-exports selected utilities from the Unity Catalog open source package.

Available aliases:

- :class:`databricks_openai.UCFunctionToolkit`
- :class:`databricks_openai.DatabricksFunctionClient`
- :func:`databricks_openai.set_uc_function_client`

Refer to the Unity Catalog `documentation <https://docs.unitycatalog.io/ai/integrations/openai/#using-unity-catalog-ai-with-the-openai-sdk>`_ for more information.
"""

from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

from databricks_openai.mcp_server_toolkit import McpServerToolkit, ToolInfo
from databricks_openai.utils.clients import AsyncDatabricksOpenAI, DatabricksOpenAI
from databricks_openai.vector_search_retriever_tool import VectorSearchRetrieverTool

# Expose all integrations to users under databricks-openai
__all__ = [
    "VectorSearchRetrieverTool",
    "UCFunctionToolkit",
    "DatabricksFunctionClient",
    "set_uc_function_client",
    "DatabricksOpenAI",
    "AsyncDatabricksOpenAI",
    "McpServerToolkit",
    "ToolInfo",
]
