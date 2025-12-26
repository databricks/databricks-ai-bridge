from __future__ import annotations

import logging
import re
from unittest.mock import MagicMock

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")

import databricks_ai_bridge.lakebase as lakebase
from databricks_ai_bridge.lakebase import AsyncLakebasePool, LakebasePool


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


def _make_connection_pool_class():
    class TestConnectionPool:
        def __init__(
            self,
            *,
            conninfo,
            connection_class,
            **kwargs,
        ):
            self.conninfo = conninfo
            self.connection_class = connection_class

    return TestConnectionPool


def test_lakebase_pool_configures_connection_pool(monkeypatch):
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_workspace()
    workspace.database.get_database_instance.return_value.read_write_dns = "db.host"

    pool = LakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    test_pool = pool.pool
    assert test_pool.conninfo == (
        "dbname=databricks_postgres user=sp-123 host=db.host port=5432 sslmode=require"
    )

    assert test_pool.connection_class is not None
    assert issubclass(test_pool.connection_class, lakebase.psycopg.Connection)


def test_lakebase_pool_logs_cache_seconds(monkeypatch, caplog):
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_workspace()
    with caplog.at_level(logging.INFO):
        LakebasePool(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    assert any(
        record.levelno == logging.INFO and re.search(r"cache=3000s$", record.getMessage())
        for record in caplog.records
    )


def test_lakebase_pool_resolves_host_from_instance(monkeypatch):
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_workspace()
    workspace.database.get_database_instance.return_value.read_write_dns = "rw.host"
    workspace.database.get_database_instance.return_value.read_only_dns = "ro.host"

    pool = LakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert pool.host == "rw.host"


def test_lakebase_pool_uses_service_principal_username(monkeypatch):
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_workspace(
        sp_application_id="service_principal_client_id",
        user_name="test@databricks.com",
    )

    pool = LakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert pool.username == "service_principal_client_id"
    assert "user=service_principal_client_id" in pool.pool.conninfo


def test_lakebase_pool_falls_back_to_user_when_service_principal_missing(monkeypatch):
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_workspace(
        sp_application_id=None,
        user_name="test@databricks.com",
    )

    pool = LakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert pool.username == "test@databricks.com"
    assert "user=test@databricks.com" in pool.pool.conninfo


def test_lakebase_pool_refreshes_token_after_cache_expiry(monkeypatch):
    """Verify that a new token is minted when the cache duration expires."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    token_call_count = []

    def mock_generate_credential(**kwargs):
        token_call_count.append(1)
        return MagicMock(token=f"token-{len(token_call_count)}")

    workspace = _make_workspace()
    workspace.database.generate_database_credential = mock_generate_credential

    # Create pool with short cache duration of 1 second
    pool = LakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
        token_cache_duration_seconds=1,
    )

    # Mock time to control cache expiry
    import time

    test_time = [100.0]  # Start at time=100

    def mock_time():
        return test_time[0]

    monkeypatch.setattr(time, "time", mock_time)

    # First call should mint a token
    token1 = pool._get_token()
    assert token1 == "token-1"
    assert len(token_call_count) == 1

    # Second call within cache duration should return cached token
    test_time[0] = 100.5  # 0.5 seconds later (within 1 second cache)
    token2 = pool._get_token()
    assert token2 == "token-1"  # Same cached token
    assert len(token_call_count) == 1  # No new token minted

    # Third call after cache expiry should mint a new token
    test_time[0] = 101.5  # 1.5 seconds later (past 1 second cache)
    token3 = pool._get_token()
    assert token3 == "token-2"  # New token
    assert len(token_call_count) == 2  # New token was minted

    # Fourth call within new cache window should return cached token
    test_time[0] = 102.0  # 0.5 seconds after last mint
    token4 = pool._get_token()
    assert token4 == "token-2"  # Same cached token
    assert len(token_call_count) == 2  # No new token minted


# =============================================================================
# AsyncLakebasePool Tests
# =============================================================================


def _make_async_connection_pool_class():
    class TestAsyncConnectionPool:
        def __init__(
            self,
            *,
            conninfo,
            connection_class,
            **kwargs,
        ):
            self.conninfo = conninfo
            self.connection_class = connection_class
            self._opened = False
            self._closed = False

        async def open(self):
            self._opened = True

        async def close(self):
            self._closed = True

        def connection(self):
            class _AsyncCtx:
                def __init__(self, outer):
                    self.outer = outer

                async def __aenter__(self):
                    return "async-conn"

                async def __aexit__(self, exc_type, exc, tb):
                    pass

            return _AsyncCtx(self)

    return TestAsyncConnectionPool


@pytest.mark.asyncio
async def test_async_lakebase_pool_configures_connection_pool(monkeypatch):
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace()
    workspace.database.get_database_instance.return_value.read_write_dns = "db.host"

    pool = AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    test_pool = pool.pool
    assert test_pool.conninfo == (
        "dbname=databricks_postgres user=sp-123 host=db.host port=5432 sslmode=require"
    )

    assert test_pool.connection_class is not None
    assert issubclass(test_pool.connection_class, lakebase.psycopg.AsyncConnection)


@pytest.mark.asyncio
async def test_async_lakebase_pool_logs_cache_seconds(monkeypatch, caplog):
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace()
    with caplog.at_level(logging.INFO):
        AsyncLakebasePool(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    assert any(
        record.levelno == logging.INFO and re.search(r"cache=3000s$", record.getMessage())
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_async_lakebase_pool_resolves_host_from_instance(monkeypatch):
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace()
    workspace.database.get_database_instance.return_value.read_write_dns = "rw.host"
    workspace.database.get_database_instance.return_value.read_only_dns = "ro.host"

    pool = AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert pool.host == "rw.host"


@pytest.mark.asyncio
async def test_async_lakebase_pool_uses_service_principal_username(monkeypatch):
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace(
        sp_application_id="service_principal_client_id",
        user_name="test@databricks.com",
    )

    pool = AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert pool.username == "service_principal_client_id"
    assert "user=service_principal_client_id" in pool.pool.conninfo


@pytest.mark.asyncio
async def test_async_lakebase_pool_falls_back_to_user_when_service_principal_missing(monkeypatch):
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace(
        sp_application_id=None,
        user_name="test@databricks.com",
    )

    pool = AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert pool.username == "test@databricks.com"
    assert "user=test@databricks.com" in pool.pool.conninfo


@pytest.mark.asyncio
async def test_async_lakebase_pool_refreshes_token_after_cache_expiry(monkeypatch):
    """Verify that a new token is minted when the cache duration expires."""
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    token_call_count = []

    def mock_generate_credential(**kwargs):
        token_call_count.append(1)
        return MagicMock(token=f"token-{len(token_call_count)}")

    workspace = _make_workspace()
    workspace.database.generate_database_credential = mock_generate_credential

    # Create pool with short cache duration of 1 second
    pool = AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
        token_cache_duration_seconds=1,
    )

    # Mock time to control cache expiry
    import time

    test_time = [100.0]  # Start at time=100

    def mock_time():
        return test_time[0]

    monkeypatch.setattr(time, "time", mock_time)

    # First call should mint a token
    token1 = pool._get_token_sync()
    assert token1 == "token-1"
    assert len(token_call_count) == 1

    # Second call within cache duration should return cached token
    test_time[0] = 100.5  # 0.5 seconds later (within 1 second cache)
    token2 = pool._get_token_sync()
    assert token2 == "token-1"  # Same cached token
    assert len(token_call_count) == 1  # No new token minted

    # Third call after cache expiry should mint a new token
    test_time[0] = 101.5  # 1.5 seconds later (past 1 second cache)
    token3 = pool._get_token_sync()
    assert token3 == "token-2"  # New token
    assert len(token_call_count) == 2  # New token was minted

    # Fourth call within new cache window should return cached token
    test_time[0] = 102.0  # 0.5 seconds after last mint
    token4 = pool._get_token_sync()
    assert token4 == "token-2"  # Same cached token
    assert len(token_call_count) == 2  # No new token minted


@pytest.mark.asyncio
async def test_async_lakebase_pool_context_manager(monkeypatch):
    """Test async context manager opens and closes the pool."""
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace()

    async with AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    ) as pool:
        assert pool.pool._opened
        assert not pool.pool._closed

    assert pool.pool._closed


@pytest.mark.asyncio
async def test_async_lakebase_pool_connection(monkeypatch):
    """Test getting a connection from the async pool."""
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace()

    async with AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    ) as pool:
        async with pool.connection() as conn:
            assert conn == "async-conn"


@pytest.mark.asyncio
async def test_async_lakebase_pool_open_close_methods(monkeypatch):
    """Test explicit open and close methods."""
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace()

    pool = AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert not pool.pool._opened
    assert not pool.pool._closed

    await pool.open()
    assert pool.pool._opened

    await pool.close()
    assert pool.pool._closed
