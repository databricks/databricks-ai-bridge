from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


# create lightweight stand-ins for modules for testing
def _ensure_optional_modules() -> None:
    if "psycopg" not in sys.modules:
        psycopg_mod = types.ModuleType("psycopg")

        class _Connection:
            pass

        psycopg_mod.Connection = _Connection
        sys.modules["psycopg"] = psycopg_mod
    else:
        psycopg_mod = sys.modules["psycopg"]

    if "psycopg.rows" not in sys.modules:
        rows_mod = types.ModuleType("psycopg.rows")

        def dict_row(record):
            return record

        rows_mod.dict_row = dict_row
        sys.modules["psycopg.rows"] = rows_mod
        psycopg_mod.rows = rows_mod

    if "psycopg_pool" not in sys.modules:
        pool_mod = types.ModuleType("psycopg_pool")

        class ConnectionPool:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("tests must monkeypatch psycopg_pool.ConnectionPool")

        pool_mod.ConnectionPool = ConnectionPool
        pool_mod.PoolClosed = type("PoolClosed", (Exception,), {})
        pool_mod.PoolTimeout = type("PoolTimeout", (Exception,), {})
        sys.modules["psycopg_pool"] = pool_mod

    databricks_pkg = sys.modules.setdefault("databricks", types.ModuleType("databricks"))
    if not getattr(databricks_pkg, "__path__", None):
        databricks_pkg.__path__ = []

    if "databricks.sdk" not in sys.modules:
        sdk_mod = types.ModuleType("databricks.sdk")
        sdk_mod.__path__ = []

        class WorkspaceClient:
            pass

        sdk_mod.WorkspaceClient = WorkspaceClient
        sys.modules["databricks.sdk"] = sdk_mod
        databricks_pkg.sdk = sdk_mod

    if "databricks.sdk.core" not in sys.modules:
        core_mod = types.ModuleType("databricks.sdk.core")

        class Config:
            pass

        core_mod.Config = Config
        sys.modules["databricks.sdk.core"] = core_mod

    if "databricks.sdk.credentials_provider" not in sys.modules:
        cred_mod = types.ModuleType("databricks.sdk.credentials_provider")

        class CredentialsProvider:
            def __call__(self):
                return {}

        class CredentialsStrategy:
            def __call__(self, cfg):
                return CredentialsProvider()

            def auth_type(self):
                return "default"

        class DefaultCredentials(CredentialsStrategy):
            pass

        cred_mod.CredentialsProvider = CredentialsProvider
        cred_mod.CredentialsStrategy = CredentialsStrategy
        cred_mod.DefaultCredentials = DefaultCredentials
        sys.modules["databricks.sdk.credentials_provider"] = cred_mod

    if "langgraph" not in sys.modules:
        langgraph_mod = types.ModuleType("langgraph")
        langgraph_mod.__path__ = []
        sys.modules["langgraph"] = langgraph_mod
    else:
        langgraph_mod = sys.modules["langgraph"]

    if "langgraph.checkpoint" not in sys.modules:
        checkpoint_mod = types.ModuleType("langgraph.checkpoint")
        checkpoint_mod.__path__ = []
        sys.modules["langgraph.checkpoint"] = checkpoint_mod
        langgraph_mod.checkpoint = checkpoint_mod
    else:
        checkpoint_mod = sys.modules["langgraph.checkpoint"]

    if "langgraph.checkpoint.postgres" not in sys.modules:
        postgres_mod = types.ModuleType("langgraph.checkpoint.postgres")

        class PostgresSaver:
            def __init__(self, conn):
                self._conn = conn

            def setup(self):
                pass

        postgres_mod.PostgresSaver = PostgresSaver
        sys.modules["langgraph.checkpoint.postgres"] = postgres_mod
        checkpoint_mod.postgres = postgres_mod

    if "unitycatalog" not in sys.modules:
        unity_mod = types.ModuleType("unitycatalog")
        unity_mod.__path__ = []
        sys.modules["unitycatalog"] = unity_mod
    else:
        unity_mod = sys.modules["unitycatalog"]

    if "unitycatalog.ai" not in sys.modules:
        ai_mod = types.ModuleType("unitycatalog.ai")
        ai_mod.__path__ = []
        sys.modules["unitycatalog.ai"] = ai_mod
        unity_mod.ai = ai_mod
    else:
        ai_mod = sys.modules["unitycatalog.ai"]

    if "unitycatalog.ai.core" not in sys.modules:
        core_pkg = types.ModuleType("unitycatalog.ai.core")
        core_pkg.__path__ = []
        sys.modules["unitycatalog.ai.core"] = core_pkg
        ai_mod.core = core_pkg
    else:
        core_pkg = sys.modules["unitycatalog.ai.core"]

    if "unitycatalog.ai.core.base" not in sys.modules:
        base_mod = types.ModuleType("unitycatalog.ai.core.base")

        def set_uc_function_client(*args, **kwargs):
            return None

        base_mod.set_uc_function_client = set_uc_function_client
        sys.modules["unitycatalog.ai.core.base"] = base_mod
        core_pkg.base = base_mod

    if "unitycatalog.ai.core.databricks" not in sys.modules:
        databricks_mod = types.ModuleType("unitycatalog.ai.core.databricks")

        class DatabricksFunctionClient:
            pass

        databricks_mod.DatabricksFunctionClient = DatabricksFunctionClient
        sys.modules["unitycatalog.ai.core.databricks"] = databricks_mod
        core_pkg.databricks = databricks_mod

    if "unitycatalog.ai.langchain" not in sys.modules:
        langchain_pkg = types.ModuleType("unitycatalog.ai.langchain")
        langchain_pkg.__path__ = []
        sys.modules["unitycatalog.ai.langchain"] = langchain_pkg
        ai_mod.langchain = langchain_pkg
    else:
        langchain_pkg = sys.modules["unitycatalog.ai.langchain"]

    if "unitycatalog.ai.langchain.toolkit" not in sys.modules:
        toolkit_mod = types.ModuleType("unitycatalog.ai.langchain.toolkit")

        class UCFunctionToolkit:
            pass

        class UnityCatalogTool:
            pass

        toolkit_mod.UCFunctionToolkit = UCFunctionToolkit
        toolkit_mod.UnityCatalogTool = UnityCatalogTool
        sys.modules["unitycatalog.ai.langchain.toolkit"] = toolkit_mod
        langchain_pkg.toolkit = toolkit_mod

    if "databricks_langchain.chat_models" not in sys.modules:
        chat_mod = types.ModuleType("databricks_langchain.chat_models")

        class ChatDatabricks:
            pass

        chat_mod.ChatDatabricks = ChatDatabricks
        sys.modules["databricks_langchain.chat_models"] = chat_mod

    if "databricks_langchain.embeddings" not in sys.modules:
        embeddings_mod = types.ModuleType("databricks_langchain.embeddings")

        class DatabricksEmbeddings:
            pass

        embeddings_mod.DatabricksEmbeddings = DatabricksEmbeddings
        sys.modules["databricks_langchain.embeddings"] = embeddings_mod

    if "databricks_langchain.genie" not in sys.modules:
        genie_mod = types.ModuleType("databricks_langchain.genie")

        class GenieAgent:
            pass

        genie_mod.GenieAgent = GenieAgent
        sys.modules["databricks_langchain.genie"] = genie_mod

    if "databricks_langchain.vector_search_retriever_tool" not in sys.modules:
        vs_mod = types.ModuleType("databricks_langchain.vector_search_retriever_tool")

        class VectorSearchRetrieverTool:
            pass

        vs_mod.VectorSearchRetrieverTool = VectorSearchRetrieverTool
        sys.modules["databricks_langchain.vector_search_retriever_tool"] = vs_mod

    if "databricks_langchain.vectorstores" not in sys.modules:
        vectorstores_mod = types.ModuleType("databricks_langchain.vectorstores")

        class DatabricksVectorSearch:
            pass

        vectorstores_mod.DatabricksVectorSearch = DatabricksVectorSearch
        sys.modules["databricks_langchain.vectorstores"] = vectorstores_mod


_ensure_optional_modules()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BRIDGE_SRC = PROJECT_ROOT / "src"
LANGCHAIN_SRC = PROJECT_ROOT / "integrations" / "langchain" / "src"
for path in (BRIDGE_SRC, LANGCHAIN_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from databricks_ai_bridge import lakebase  # type: ignore  # noqa: E402

from databricks_langchain import CheckpointSaver  # noqa: E402


def _make_workspace():
    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.current_service_principal.me.side_effect = RuntimeError("no sp")
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")
    return workspace


class RecordingConnectionPool:
    def __init__(self, log, connection_value="conn"):
        self.log = log
        self.connection_value = connection_value
        self.kwargs = None
        self.min_size = None
        self.max_size = None
        self.timeout = None
        self.open = None
        self.conninfo = None
        self.connection_class = None
        self.getconn_calls = 0
        self.putconn_calls = []
        self.closed = False

    def __call__(
        self,
        *,
        conninfo,
        connection_class=None,
        kwargs,
        min_size,
        max_size,
        timeout,
        open,
        **extra,
    ):
        self.conninfo = conninfo
        self.connection_class = connection_class
        self.kwargs = kwargs
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        self.open = open
        self.extra = extra
        return self

    def connection(self):
        self.log.append("pool_connection")

        class _Ctx:
            def __init__(self, outer):
                self.outer = outer
                self.entered = False
                self.exited = False

            def __enter__(self):
                self.outer.log.append("ctx_enter")
                self.entered = True
                return self.outer.connection_value

            def __exit__(self, exc_type, exc, tb):
                self.outer.log.append("ctx_exit")
                self.exited = True

        return _Ctx(self)

    def getconn(self):
        self.getconn_calls += 1
        return self.connection_value

    def putconn(self, conn):
        self.putconn_calls.append(conn)

    def close(self):
        self.closed = True


def test_checkpoint_saver_configures_lakebase(monkeypatch):
    log = []
    fake_pool = RecordingConnectionPool(log, connection_value="lake-conn")
    monkeypatch.setattr(lakebase, "ConnectionPool", fake_pool)

    workspace = _make_workspace()

    saver = CheckpointSaver(
        database_instance="lakebase-instance",
        workspace_client=workspace,
        host="db-host",
        database="analytics",
        min_size=2,
        max_size=5,
        timeout=9.5,
        open=False,
        connection_kwargs={"application_name": "pytest"},
        probe=False,
    )

    assert isinstance(saver, lakebase.PooledPostgresSaver)
    assert (
        fake_pool.conninfo
        == "dbname=analytics user=test@databricks.com host=db-host port=5432 sslmode=require"
    )
    assert fake_pool.connection_class is not None
    assert fake_pool.min_size == 2
    assert fake_pool.max_size == 5
    assert fake_pool.timeout == 9.5
    assert fake_pool.open is False
    assert fake_pool.kwargs["application_name"] == "pytest"

    with saver:
        pass

    assert fake_pool.putconn_calls == ["lake-conn"]
    assert fake_pool.closed is False

    saver.close()
    assert fake_pool.closed is True
