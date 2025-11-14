from __future__ import annotations

import logging
import re
from unittest.mock import MagicMock

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")

import databricks_ai_bridge.lakebase as lakebase
from databricks_ai_bridge.lakebase import LakebasePool

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
