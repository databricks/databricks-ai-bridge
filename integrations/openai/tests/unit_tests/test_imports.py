import subprocess
import sys

from databricks_openai import (
    AsyncDatabricksOpenAI,
    DatabricksFunctionClient,
    DatabricksOpenAI,
    DatabricksSandboxClient,
    DatabricksSandboxClientOptions,
    DatabricksSandboxSession,
    DatabricksSandboxSessionState,
    McpServerToolkit,
    ToolInfo,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
    set_uc_function_client,
)
from databricks_openai.agents import McpServer

assert DatabricksFunctionClient
assert UCFunctionToolkit
assert VectorSearchRetrieverTool
assert set_uc_function_client
assert DatabricksOpenAI
assert AsyncDatabricksOpenAI
assert McpServerToolkit
assert ToolInfo
assert McpServer
assert DatabricksSandboxClient
assert DatabricksSandboxClientOptions
assert DatabricksSandboxSession
assert DatabricksSandboxSessionState


def test_client_import_does_not_load_toolkits():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import databricks_openai as package
assert 'DatabricksOpenAI' in dir(package)
assert 'databricks_openai.utils.clients' not in sys.modules
from databricks_openai import DatabricksOpenAI, AsyncDatabricksOpenAI
from databricks_openai.utils.clients import DatabricksOpenAI as implementation
assert DatabricksOpenAI is implementation is package.DatabricksOpenAI
for name in ('databricks_openai.mcp_server_toolkit', 'databricks_openai.vector_search_retriever_tool',
             'databricks_mcp', 'unitycatalog', 'numpy', 'mlflow'):
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
    import databricks_openai as package

    for name in package.__all__:
        assert getattr(package, name) is getattr(package, name)
