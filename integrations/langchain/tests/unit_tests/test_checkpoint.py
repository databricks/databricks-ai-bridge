from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("langgraph.checkpoint.postgres")

from databricks_ai_bridge import lakebase

from databricks_langchain import AsyncCheckpointSaver, CheckpointSaver


class TestConnectionPool:
    def __init__(self, connection_value="conn"):
        self.connection_value = connection_value
        self.conninfo = ""

    def __call__(
        self,
        *,
        conninfo,
        connection_class=None,
        **kwargs,
    ):
        self.conninfo = conninfo
        return self

    def connection(self):
        class _Ctx:
            def __init__(self, outer):
                self.outer = outer

            def __enter__(self):
                return self.outer.connection_value

            def __exit__(self, exc_type, exc, tb):
                pass

        return _Ctx(self)


def test_checkpoint_saver_configures_lakebase(monkeypatch):
    test_pool = TestConnectionPool(connection_value="lake-conn")
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.database.get_database_instance.return_value.read_write_dns = "db-host"
    workspace.current_service_principal.me.side_effect = RuntimeError("no sp")
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    saver = CheckpointSaver(
        instance_name="lakebase-instance",
        workspace_client=workspace,
    )

    assert (
        test_pool.conninfo
        == "dbname=databricks_postgres user=test@databricks.com host=db-host port=5432 sslmode=require"
    )

    assert saver._lakebase.pool == test_pool

    with saver._lakebase.connection() as conn:
        assert conn == "lake-conn"


class TestAsyncConnectionPool:
    """Mock async connection pool for testing."""

    def __init__(self, connection_value="async-conn"):
        self.connection_value = connection_value
        self.conninfo = ""
        self._opened = False
        self._closed = False

    def __call__(
        self,
        *,
        conninfo,
        connection_class=None,
        **kwargs,
    ):
        self.conninfo = conninfo
        return self

    def connection(self):
        class _AsyncCtx:
            def __init__(self, outer):
                self.outer = outer

            async def __aenter__(self):
                return self.outer.connection_value

            async def __aexit__(self, exc_type, exc, tb):
                pass

        return _AsyncCtx(self)

    async def open(self):
        self._opened = True

    async def close(self):
        self._closed = True


@pytest.mark.asyncio
async def test_async_checkpoint_saver_configures_lakebase(monkeypatch):
    test_pool = TestAsyncConnectionPool(connection_value="async-lake-conn")
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.database.get_database_instance.return_value.read_write_dns = "db-host"
    workspace.current_service_principal.me.side_effect = RuntimeError("no sp")
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    saver = AsyncCheckpointSaver(
        instance_name="lakebase-instance",
        workspace_client=workspace,
    )

    assert (
        test_pool.conninfo
        == "dbname=databricks_postgres user=test@databricks.com host=db-host port=5432 sslmode=require"
    )

    assert saver._lakebase.pool == test_pool


@pytest.mark.asyncio
async def test_async_checkpoint_saver_context_manager(monkeypatch):
    test_pool = TestAsyncConnectionPool(connection_value="async-lake-conn")
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.database.get_database_instance.return_value.read_write_dns = "db-host"
    workspace.current_service_principal.me.side_effect = RuntimeError("no sp")
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    async with AsyncCheckpointSaver(
        instance_name="lakebase-instance",
        workspace_client=workspace,
    ) as saver:
        assert test_pool._opened
        assert saver._lakebase.pool == test_pool

    assert test_pool._closed


@pytest.mark.asyncio
async def test_async_checkpoint_saver_connection(monkeypatch):
    test_pool = TestAsyncConnectionPool(connection_value="async-lake-conn")
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.database.get_database_instance.return_value.read_write_dns = "db-host"
    workspace.current_service_principal.me.side_effect = RuntimeError("no sp")
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    async with AsyncCheckpointSaver(
        instance_name="lakebase-instance",
        workspace_client=workspace,
    ) as saver:
        async with saver._lakebase.connection() as conn:
            assert conn == "async-lake-conn"
