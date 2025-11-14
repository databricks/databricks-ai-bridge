from __future__ import annotations

import logging
import re
import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Provide lightweight stubs for optional dependencies. Each stub is only
# installed when the real module is unavailable so that local environments
# with the actual packages continue to work unchanged.
# ---------------------------------------------------------------------------


def _ensure_optional_modules() -> None:
    if "psycopg" not in sys.modules:
        psycopg_mod = types.ModuleType("psycopg")

        class _Connection:  # pragma: no cover - simple structural stub
            pass

        psycopg_mod.Connection = _Connection
        sys.modules["psycopg"] = psycopg_mod
    else:
        psycopg_mod = sys.modules["psycopg"]

    if "psycopg.rows" not in sys.modules:
        rows_mod = types.ModuleType("psycopg.rows")

        def dict_row(record):  # pragma: no cover - behaviour exercised indirectly
            return record

        rows_mod.dict_row = dict_row
        sys.modules["psycopg.rows"] = rows_mod
        psycopg_mod.rows = rows_mod

    if "psycopg_pool" not in sys.modules:
        pool_mod = types.ModuleType("psycopg_pool")

        class ConnectionPool:  # pragma: no cover - patched in tests
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

        class WorkspaceClient:  # pragma: no cover - not instantiated in tests
            pass

        sdk_mod.WorkspaceClient = WorkspaceClient
        sys.modules["databricks.sdk"] = sdk_mod
        databricks_pkg.sdk = sdk_mod

    langgraph_mod = sys.modules.setdefault("langgraph", types.ModuleType("langgraph"))
    if not getattr(langgraph_mod, "__path__", None):
        langgraph_mod.__path__ = []

    if "langgraph.checkpoint" not in sys.modules:
        checkpoint_mod = types.ModuleType("langgraph.checkpoint")
        checkpoint_mod.__path__ = []
        sys.modules["langgraph.checkpoint"] = checkpoint_mod
        langgraph_mod.checkpoint = checkpoint_mod
    else:
        checkpoint_mod = sys.modules["langgraph.checkpoint"]

    if "langgraph.checkpoint.postgres" not in sys.modules:
        postgres_mod = types.ModuleType("langgraph.checkpoint.postgres")

        class PostgresSaver:  # pragma: no cover - behaviour exercised via wrapper
            def __init__(self, conn):
                self._conn = conn

        postgres_mod.PostgresSaver = PostgresSaver
        sys.modules["langgraph.checkpoint.postgres"] = postgres_mod
        checkpoint_mod.postgres = postgres_mod

    if "databricks_ai_bridge.model_serving_obo_credential_strategy" not in sys.modules:
        strategy_mod = types.ModuleType(
            "databricks_ai_bridge.model_serving_obo_credential_strategy"
        )

        class ModelServingUserCredentials:  # pragma: no cover - structural stub
            pass

        strategy_mod.ModelServingUserCredentials = ModelServingUserCredentials
        sys.modules["databricks_ai_bridge.model_serving_obo_credential_strategy"] = strategy_mod


_ensure_optional_modules()

import databricks_ai_bridge.lakebase as lakebase
from databricks_ai_bridge.lakebase import (
    LakebasePool,
    pooled_connection,
)

try:
    from psycopg.rows import dict_row  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when stubbed manually
    dict_row = sys.modules["psycopg.rows"].dict_row  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixtures and shared helpers
# ---------------------------------------------------------------------------


def _make_workspace(
    *,
    sp_application_id: str | None = "sp-123",
    user_name: str = "test@databricks.com",
    credential_token: str = "token-1",
):
    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token=credential_token)
    instance = MagicMock()
    instance.read_write_dns = "db.host"
    instance.read_only_dns = "db-ro.host"
    workspace.database.get_database_instance.return_value = instance
    if sp_application_id is None:
        workspace.current_service_principal.me.side_effect = RuntimeError("missing sp")
    else:
        workspace.current_service_principal.me.return_value = MagicMock(
            application_id=sp_application_id
        )
    workspace.current_user.me.return_value = MagicMock(user_name=user_name)
    return workspace


class _ConnectionContext:
    def __init__(self, log, connection_value):
        self._log = log
        self._value = connection_value
        self.entered = False
        self.exited = False

    def __enter__(self):
        self._log.append("ctx_enter")
        self.entered = True
        return self._value

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - behaviour under test
        self._log.append("ctx_exit")
        self.exited = True


def _make_connection_pool_class(log, connection_value="pooled-conn"):
    class FakeConnectionPool:
        def __init__(
            self,
            *,
            conninfo,
            connection_class,
            min_size,
            max_size,
            timeout,
            open,
            kwargs,
        ):
            self.conninfo = conninfo
            self.connection_class = connection_class
            self.min_size = min_size
            self.max_size = max_size
            self.timeout = timeout
            self.open = open
            self.kwargs = kwargs
            self.log = log
            self.connection_value = connection_value
            self.context = _ConnectionContext(log, connection_value)
            self.getconn_calls = 0
            self.putconn_calls = []

        def connection(self):
            self.log.append("pool_connection")
            return self.context

        def getconn(self):
            self.getconn_calls += 1
            return self.connection_value

        def putconn(self, conn):
            self.putconn_calls.append(conn)

        def close(self):  # pragma: no cover - not used in current tests
            self.log.append("pool_close")

    return FakeConnectionPool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lakebase_pool_configures_connection_pool(monkeypatch):
    log: list[str] = []
    FakeConnectionPool = _make_connection_pool_class(log)
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", FakeConnectionPool)

    workspace = _make_workspace()
    workspace.database.get_database_instance.return_value.read_write_dns = "db.host"

    pool = LakebasePool(
        workspace_client=workspace,
        instance_name="lake-instance",
    )

    fake_pool = pool.pool
    assert fake_pool.conninfo == (
        "dbname=databricks_postgres user=sp-123 host=db.host port=5432 sslmode=require"
    )

    assert issubclass(fake_pool.connection_class, lakebase._RotatingCredentialConnection)


def test_lakebase_pool_logs_cache_seconds(monkeypatch, caplog):
    FakeConnectionPool = _make_connection_pool_class([])
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", FakeConnectionPool)

    workspace = _make_workspace()
    with caplog.at_level(logging.INFO):
        LakebasePool(
            workspace_client=workspace,
            instance_name="lake-instance",
        )

    assert any(
        record.levelno == logging.INFO and re.search(r"cache=3000s$", record.getMessage())
        for record in caplog.records
    )


def test_lakebase_pool_resolves_host_from_instance(monkeypatch):
    FakeConnectionPool = _make_connection_pool_class([])
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", FakeConnectionPool)

    workspace = _make_workspace()
    workspace.database.get_database_instance.return_value.read_write_dns = "rw.host"
    workspace.database.get_database_instance.return_value.read_only_dns = "ro.host"

    pool = LakebasePool(
        workspace_client=workspace,
        instance_name="lake-instance",
    )

    assert pool.host == "rw.host"


def test_lakebase_pool_infers_username_from_service_principal(monkeypatch):
    log: list[str] = []
    FakeConnectionPool = _make_connection_pool_class(log)
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", FakeConnectionPool)

    workspace = _make_workspace(
        sp_application_id="service_principal_client_id",
        user_name="test@databricks.com",
    )

    pool = LakebasePool(
        workspace_client=workspace,
        instance_name="lake-instance",
    )

    assert pool.username == "service_principal_client_id"
    assert "user=service_principal_client_id" in pool.pool.conninfo


def test_lakebase_pool_falls_back_to_user_when_service_principal_missing(monkeypatch):
    log: list[str] = []
    FakeConnectionPool = _make_connection_pool_class(log)
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", FakeConnectionPool)

    workspace = _make_workspace(
        sp_application_id=None,
        user_name="test@databricks.com",
    )

    pool = LakebasePool(
        workspace_client=workspace,
        instance_name="lake-instance",
    )

    assert pool.username == "test@databricks.com"
    assert "user=test@databricks.com" in pool.pool.conninfo


def test_pooled_connection_with_lakebase_pool(monkeypatch):
    log: list[str] = []
    FakeConnectionPool = _make_connection_pool_class(log, connection_value="lake-conn")
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", FakeConnectionPool)

    workspace = _make_workspace()
    lake_pool = LakebasePool(
        workspace_client=workspace,
        instance_name="lake-instance",
    )

    with pooled_connection(lake_pool) as conn:
        assert conn == "lake-conn"

    assert log == ["pool_connection", "ctx_enter", "ctx_exit"]
    assert lake_pool.pool.context.entered and lake_pool.pool.context.exited
