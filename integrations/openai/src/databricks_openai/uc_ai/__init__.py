import warnings

warnings.warn(
    "Imports from this module are deprecated and will be removed in a future release. "
    "Please update the code to import directly from databricks_openai.\n\n"
    "For example, replace imports like: `from databricks_openai.uc_ai import UCFunctionToolkit`\n"
    "with: `from databricks_openai import UCFunctionToolkit`",
    DeprecationWarning,
    stacklevel=3,
)

from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.openai.toolkit import UCFunctionToolkit

# Alias all necessary imports from unitycatalog-ai here
__all__ = [
    "UCFunctionToolkit",
    "DatabricksFunctionClient",
    "set_uc_function_client",
]
