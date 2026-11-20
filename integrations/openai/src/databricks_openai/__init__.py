"""**Re-exported Unity Catalog Utilities**

This module re-exports selected utilities from the Unity Catalog open source package.

Available aliases:

- :class:`databricks_openai.UCFunctionToolkit`
- :class:`databricks_openai.DatabricksFunctionClient`
- :func:`databricks_openai.set_uc_function_client`

Refer to the Unity Catalog `documentation <https://docs.unitycatalog.io/ai/integrations/openai/#using-unity-catalog-ai-with-the-openai-sdk>`_ for more information.

Public exports load their implementation on first access.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unitycatalog.ai.core.base import set_uc_function_client
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient
    from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

    from databricks_openai.mcp_server_toolkit import McpServerToolkit, ToolInfo
    from databricks_openai.sandbox import (
        DEFAULT_DATABRICKS_WORKSPACE_ROOT,
        DatabricksSandboxClient,
        DatabricksSandboxClientOptions,
        DatabricksSandboxSession,
        DatabricksSandboxSessionState,
    )
    from databricks_openai.utils.clients import AsyncDatabricksOpenAI, DatabricksOpenAI
    from databricks_openai.vector_search_retriever_tool import VectorSearchRetrieverTool

__all__ = [
    "DEFAULT_DATABRICKS_WORKSPACE_ROOT",
    "VectorSearchRetrieverTool",
    "UCFunctionToolkit",
    "DatabricksFunctionClient",
    "DatabricksSandboxClient",
    "DatabricksSandboxClientOptions",
    "DatabricksSandboxSession",
    "DatabricksSandboxSessionState",
    "set_uc_function_client",
    "DatabricksOpenAI",
    "AsyncDatabricksOpenAI",
    "McpServerToolkit",
    "ToolInfo",
]

_MODULE_BY_NAME = {
    "DEFAULT_DATABRICKS_WORKSPACE_ROOT": "databricks_openai.sandbox",
    "VectorSearchRetrieverTool": "databricks_openai.vector_search_retriever_tool",
    "UCFunctionToolkit": "unitycatalog.ai.openai.toolkit",
    "DatabricksFunctionClient": "unitycatalog.ai.core.databricks",
    "DatabricksSandboxClient": "databricks_openai.sandbox",
    "DatabricksSandboxClientOptions": "databricks_openai.sandbox",
    "DatabricksSandboxSession": "databricks_openai.sandbox",
    "DatabricksSandboxSessionState": "databricks_openai.sandbox",
    "set_uc_function_client": "unitycatalog.ai.core.base",
    "DatabricksOpenAI": "databricks_openai.utils.clients",
    "AsyncDatabricksOpenAI": "databricks_openai.utils.clients",
    "McpServerToolkit": "databricks_openai.mcp_server_toolkit",
    "ToolInfo": "databricks_openai.mcp_server_toolkit",
}


def __getattr__(name: str) -> object:
    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
