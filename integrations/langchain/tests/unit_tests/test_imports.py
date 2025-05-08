import pytest
from packaging import version
import databricks_langchain

if version.parse(databricks_langchain.__version__) < version.parse("0.4.0"):
    pytest.skip("Test requires databricks-langchain >= 0.4.0", allow_module_level=True)

from databricks_langchain import (
    ChatDatabricks,
    DatabricksEmbeddings,
    DatabricksFunctionClient,
    DatabricksVectorSearch,
    GenieAgent,
    UCFunctionToolkit,
    UnityCatalogTool,
    VectorSearchRetrieverTool,
    set_uc_function_client,
)

assert ChatDatabricks
assert DatabricksEmbeddings
assert DatabricksFunctionClient
assert DatabricksVectorSearch
assert GenieAgent
assert UCFunctionToolkit
assert UnityCatalogTool
assert VectorSearchRetrieverTool
assert set_uc_function_client
