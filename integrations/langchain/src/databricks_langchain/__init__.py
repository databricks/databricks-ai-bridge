"""
**Re-exported Unity Catalog Utilities**

This module re-exports selected utilities from the Unity Catalog open source package.

Available aliases:

- :class:`databricks_langchain.UCFunctionToolkit`
- :class:`databricks_langchain.UnityCatalogTool`
- :class:`databricks_langchain.DatabricksFunctionClient`
- :func:`databricks_langchain.set_uc_function_client`

Refer to the Unity Catalog `documentation <https://docs.unitycatalog.io/ai/integrations/langchain/#using-unity-catalog-ai-with-langchain>`_ for more information.
"""

from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit, UnityCatalogTool

from databricks_langchain.chat_models import ChatDatabricks
from databricks_langchain.checkpoint import CheckpointSaver
from databricks_langchain.embeddings import DatabricksEmbeddings
from databricks_langchain.genie import GenieAgent
from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool
from databricks_langchain.vectorstores import DatabricksVectorSearch

# Expose all integrations to users under databricks-langchain
__all__ = [
    "ChatDatabricks",
    "CheckpointSaver",
    "DatabricksEmbeddings",
    "DatabricksVectorSearch",
    "GenieAgent",
    "VectorSearchRetrieverTool",
    "UCFunctionToolkit",
    "UnityCatalogTool",
    "DatabricksFunctionClient",
    "set_uc_function_client",
]
