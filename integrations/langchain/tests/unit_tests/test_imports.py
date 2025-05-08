from importlib.metadata import version as get_version

import pytest
from packaging import version as pkg_version

if pkg_version.parse(get_version("databricks-langchain")) < pkg_version.parse("0.4.0"):
    pytest.skip("Requires databricks-langchain >= 0.4.0", allow_module_level=True)

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
