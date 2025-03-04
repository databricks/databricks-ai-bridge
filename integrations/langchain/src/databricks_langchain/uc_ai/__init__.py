import warnings
import logging
_logger = logging.getLogger(__name__)
_logger.error("deprecated")

warnings.warn(
    "Imports from this module are deprecated and will be removed in a future release. "
    "Please update your imports to import directly from databricks_langchain",
    DeprecationWarning,
    stacklevel=2,
)

from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit

# Alias all necessary imports from unitycatalog-ai here
__all__ = [
    "UCFunctionToolkit",
    "DatabricksFunctionClient",
    "set_uc_function_client",
]
