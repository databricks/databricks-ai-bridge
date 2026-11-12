import subprocess
import sys

from databricks_mcp import DatabricksOAuthClientProvider

assert DatabricksOAuthClientProvider


def test_oauth_import_does_not_load_client_or_model_logging():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import databricks_mcp as package
assert 'DatabricksOAuthClientProvider' in dir(package)
assert 'databricks_mcp.oauth_provider' not in sys.modules
from databricks_mcp import DatabricksOAuthClientProvider
from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider as implementation
assert DatabricksOAuthClientProvider is implementation is package.DatabricksOAuthClientProvider
assert 'databricks_mcp.mcp' not in sys.modules
assert 'mlflow' not in sys.modules
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


def test_client_import_does_not_load_model_logging():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from databricks_mcp import DatabricksMCPClient
assert 'mlflow' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_all_public_exports_resolve():
    import databricks_mcp as package

    for name in package.__all__:
        assert getattr(package, name) is getattr(package, name)
