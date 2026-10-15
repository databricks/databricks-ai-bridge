"""
**Re-exported Unity Catalog Utilities**

This module re-exports selected utilities from the Unity Catalog open source package.

Available aliases:

- :class:`databricks_openai.UCFunctionToolkit`
- :class:`databricks_openai.DatabricksFunctionClient`
- :func:`databricks_openai.set_uc_function_client`
- :class:`databricks_openai.DatabricksOpenAI`
- :class:`databricks_openai.AsyncDatabricksOpenAI`
- :class:`databricks_openai.McpServerToolkit`
- :class:`databricks_openai.ToolInfo`
- :class:`databricks_openai.VectorSearchRetrieverTool`

Refer to the Unity Catalog `documentation <https://docs.unitycatalog.io/ai/integrations/openai/#using-unity-catalog-ai-with-the-openai-sdk>`_ for more information.
"""

from importlib import import_module
from typing import Any

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

_LAZY_EXPORTS = {
    "DatabricksOpenAI": ("databricks_openai.utils.clients", "DatabricksOpenAI"),
    "AsyncDatabricksOpenAI": ("databricks_openai.utils.clients", "AsyncDatabricksOpenAI"),
    "McpServerToolkit": ("databricks_openai.mcp_server_toolkit", "McpServerToolkit"),
    "ToolInfo": ("databricks_openai.mcp_server_toolkit", "ToolInfo"),
    "VectorSearchRetrieverTool": (
        "databricks_openai.vector_search_retriever_tool",
        "VectorSearchRetrieverTool",
    ),
    "UCFunctionToolkit": ("unitycatalog.ai.openai.toolkit", "UCFunctionToolkit"),
    "DatabricksFunctionClient": (
        "unitycatalog.ai.core.databricks",
        "DatabricksFunctionClient",
    ),
    "set_uc_function_client": ("unitycatalog.ai.core.base", "set_uc_function_client"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
