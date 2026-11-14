import subprocess
import sys

from databricks_langchain import (
    ChatDatabricks,
    DatabricksEmbeddings,
    DatabricksFunctionClient,
    DatabricksSandbox,
    DatabricksSandboxSession,
    DatabricksSandboxTool,
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
assert DatabricksSandbox
assert DatabricksSandboxSession
assert DatabricksSandboxTool
assert DatabricksVectorSearch
assert GenieAgent
assert UCFunctionToolkit
assert UnityCatalogTool
assert VectorSearchRetrieverTool
assert set_uc_function_client


def test_chat_import_does_not_load_optional_integrations():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import databricks_langchain as package
assert 'ChatDatabricks' in dir(package)
assert 'databricks_langchain.chat_models' not in sys.modules
from databricks_langchain import ChatDatabricks
from databricks_langchain.chat_models import ChatDatabricks as implementation
assert ChatDatabricks is implementation is package.ChatDatabricks
for name in ('numpy', 'pandas', 'scipy', 'unitycatalog', 'databricks_langchain.vectorstores',
             'databricks_openai.mcp_server_toolkit', 'databricks_mcp', 'mlflow'):
    assert name not in sys.modules, name
try:
    package.unknown_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown exports must raise AttributeError')
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_all_public_exports_resolve():
    import databricks_langchain as package

    for name in package.__all__:
        assert getattr(package, name) is getattr(package, name)


def test_numerical_helpers_remain_available_on_demand():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import databricks_langchain.utils as utils
assert 'numpy' not in sys.modules
from databricks_langchain.utils import Matrix, cosine_similarity, maximal_marginal_relevance
import numpy as np
np.testing.assert_allclose(cosine_similarity([[1, 0], [0, 0]], [[1, 0]]), [[1], [0]])
assert maximal_marginal_relevance(np.array([1, 0]), [[1, 0], [.99, .01], [0, 1]], lambda_mult=0, k=2) == [0, 2]
assert utils.cosine_similarity is cosine_similarity
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
