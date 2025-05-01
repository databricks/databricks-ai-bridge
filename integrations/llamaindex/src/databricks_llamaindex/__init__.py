"""
**Re-exported Unity Catalog Utilities**

This module re-exports selected utilities from the Unity Catalog open source package.

Available aliases:

- :class:`databricks_llamaindex.UCFunctionToolkit`
- :class:`databricks_llamaindex.UnityCatalogTool`
- :class:`databricks_llamaindex.DatabricksFunctionClient`
- :func:`databricks_llamaindex.set_uc_function_client`

Refer to the Unity Catalog `documentation <https://docs.unitycatalog.io/ai/integrations/llamaindex/#using-unity-catalog-ai-with-llamaindex>`_ for more information.
"""

from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.llama_index.toolkit import UCFunctionToolkit

from databricks_llamaindex.vector_search_retriever_tool import VectorSearchRetrieverTool

# Expose all integrations to users under databricks-langchain
__all__ = [
    "VectorSearchRetrieverTool",
    "set_uc_function_client",
    "DatabricksFunctionClient",
    "UCFunctionToolkit",
]
