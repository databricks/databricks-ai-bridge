from __future__ import annotations

import logging
import re
from unittest.mock import MagicMock

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")

import databricks_ai_bridge.lakebase as lakebase
from databricks_ai_bridge.lakebase import (
    AsyncLakebasePool,
    LakebaseClient,
    LakebasePool,
    SchemaPrivilege,
    SequencePrivilege,
    TablePrivilege,
)


def _make_workspace(
    *,
    user_name: str = "test@databricks.com",
    credential_token: str = "token-1",
):
    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token=credential_token)
    instance = MagicMock()
    instance.read_write_dns = "db.host"
    instance.read_only_dns = "db-ro.host"
    workspace.database.get_database_instance.return_value = instance
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
        "dbname=databricks_postgres user=test@databricks.com host=db.host port=5432 sslmode=require"
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


def test_lakebase_pool_gets_username(monkeypatch):
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_workspace(user_name="myuser@databricks.com")

    pool = LakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert pool.username == "myuser@databricks.com"
    assert isinstance(pool.pool.conninfo, str)
    assert "user=myuser@databricks.com" in pool.pool.conninfo


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
        "dbname=databricks_postgres user=test@databricks.com host=db.host port=5432 sslmode=require"
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
async def test_async_lakebase_pool_gets_username(monkeypatch):
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace(user_name="myuser@databricks.com")

    pool = AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert pool.username == "myuser@databricks.com"
    assert isinstance(pool.pool.conninfo, str)
    assert "user=myuser@databricks.com" in pool.pool.conninfo


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
    token1 = await pool._get_token_async()
    assert token1 == "token-1"
    assert len(token_call_count) == 1

    # Second call within cache duration should return cached token
    test_time[0] = 100.5  # 0.5 seconds later (within 1 second cache)
    token2 = await pool._get_token_async()
    assert token2 == "token-1"  # Same cached token
    assert len(token_call_count) == 1  # No new token minted

    # Third call after cache expiry should mint a new token
    test_time[0] = 101.5  # 1.5 seconds later (past 1 second cache)
    token3 = await pool._get_token_async()
    assert token3 == "token-2"  # New token
    assert len(token_call_count) == 2  # New token was minted

    # Fourth call within new cache window should return cached token
    test_time[0] = 102.0  # 0.5 seconds after last mint
    token4 = await pool._get_token_async()
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


# =============================================================================
# LakebaseClient Tests
# =============================================================================


def _make_mock_pool():
    """Create a mock LakebasePool for testing LakebaseClient."""
    pool = MagicMock(spec=LakebasePool)
    cursor = MagicMock()
    cursor.description = None  # DDL statements return no rows
    cursor.fetchall.return_value = []
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    connection.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    connection.cursor.return_value.__exit__ = MagicMock(return_value=False)
    pool.connection.return_value = connection
    return pool, cursor


class TestLakebaseClientInit:
    """Tests for LakebaseClient initialization."""

    def test_client_requires_pool_or_instance_name(self):
        """Client must be given either pool or instance_name."""
        with pytest.raises(ValueError, match="Must provide either 'pool' or 'instance_name'"):
            LakebaseClient()


class TestLakebaseClientCreateRole:
    """Tests for LakebaseClient.create_role()."""

    def test_create_role_executes_correct_sql(self):
        """create_role should execute CREATE EXTENSION and databricks_create_role."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        client.create_role("sp-uuid-123", "SERVICE_PRINCIPAL")

        # Should have executed two queries
        assert cursor.execute.call_count == 2

        # First call: CREATE EXTENSION
        first_call = cursor.execute.call_args_list[0]
        assert "CREATE EXTENSION IF NOT EXISTS databricks_auth" in first_call[0][0]

        # Second call: databricks_create_role
        second_call = cursor.execute.call_args_list[1]
        assert "databricks_create_role" in second_call[0][0]
        assert "SERVICE_PRINCIPAL" in second_call[0][0]
        assert second_call[0][1] == ("sp-uuid-123",)

    def test_create_role_handles_duplicate_object(self, caplog):
        """create_role should log info and return None if role already exists."""
        pool, cursor = _make_mock_pool()

        # Make the second execute call raise DuplicateObject
        call_count = [0]

        def execute_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call is create_role
                raise lakebase.psycopg.errors.DuplicateObject("role already exists")

        cursor.execute.side_effect = execute_side_effect

        client = LakebaseClient(pool=pool)

        with caplog.at_level(logging.INFO):
            result = client.create_role("existing-role", "SERVICE_PRINCIPAL")

        assert result is None
        assert any("already exists" in record.message for record in caplog.records)

    def test_create_role_with_group_identity_type(self):
        """create_role works with GROUP identity type."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        client.create_role("group-id-123", "GROUP")

        second_call = cursor.execute.call_args_list[1]
        assert "GROUP" in second_call[0][0]

    def test_create_role_handles_invalid_identity(self):
        """create_role should raise ValueError with helpful message if identity not found."""
        pool, cursor = _make_mock_pool()

        # Make the second execute call raise InvalidParameterValue
        call_count = [0]

        def execute_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call is create_role
                raise lakebase.psycopg.errors.InvalidParameterValue(
                    "[Databricks Auth] Identity not found."
                )

        cursor.execute.side_effect = execute_side_effect

        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError) as exc_info:
            client.create_role("nonexistent-sp-uuid", "SERVICE_PRINCIPAL")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert "nonexistent-sp-uuid" in error_msg
        assert "service principal" in error_msg  # Should format identity type nicely


class TestLakebaseClientGrantSchema:
    """Tests for LakebaseClient.grant_schema()."""

    def test_grant_schema_single_schema(self, caplog):
        """grant_schema should execute GRANT on schema."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with caplog.at_level(logging.INFO):
            client.grant_schema(
                grantee="sp-uuid",
                privileges=[SchemaPrivilege.USAGE, SchemaPrivilege.CREATE],
                schemas=["public"],
            )

        assert cursor.execute.call_count == 1

        log_messages = [record.message for record in caplog.records]
        assert any(
            "USAGE, CREATE" in msg and "schema" in msg and "public" in msg and "sp-uuid" in msg
            for msg in log_messages
        )

    def test_grant_schema_multiple_schemas(self):
        """grant_schema should execute GRANT for each schema."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        client.grant_schema(
            grantee="sp-uuid",
            privileges=[SchemaPrivilege.USAGE],
            schemas=["drizzle", "ai_chatbot", "public"],
        )

        assert cursor.execute.call_count == 3

    def test_grant_schema_all_privileges(self, caplog):
        """grant_schema with ALL should use ALL PRIVILEGES."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with caplog.at_level(logging.INFO):
            client.grant_schema(
                grantee="sp-uuid",
                privileges=[SchemaPrivilege.ALL],
                schemas=["public"],
            )

        log_messages = [record.message for record in caplog.records]
        assert any("ALL PRIVILEGES" in msg for msg in log_messages)


class TestLakebaseClientGrantAllTablesInSchema:
    """Tests for LakebaseClient.grant_all_tables_in_schema()."""

    def test_grant_all_tables_in_schema(self, caplog):
        """grant_all_tables_in_schema should execute GRANT on all tables in schema."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with caplog.at_level(logging.INFO):
            client.grant_all_tables_in_schema(
                grantee="sp-uuid",
                privileges=[TablePrivilege.SELECT, TablePrivilege.INSERT],
                schemas=["drizzle"],
            )

        assert cursor.execute.call_count == 1

        log_messages = [record.message for record in caplog.records]
        assert any(
            "SELECT, INSERT" in msg and "all tables in schema" in msg and "drizzle" in msg
            for msg in log_messages
        )

    def test_grant_all_tables_multiple_schemas(self):
        """grant_all_tables_in_schema should execute for each schema."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        client.grant_all_tables_in_schema(
            grantee="sp-uuid",
            privileges=[TablePrivilege.SELECT],
            schemas=["schema1", "schema2"],
        )

        assert cursor.execute.call_count == 2


class TestLakebaseClientGrantTable:
    """Tests for LakebaseClient.grant_table()."""

    def test_grant_table_single_table(self, caplog):
        """grant_table should execute GRANT on specific table."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with caplog.at_level(logging.INFO):
            client.grant_table(
                grantee="sp-uuid",
                privileges=[TablePrivilege.SELECT, TablePrivilege.INSERT, TablePrivilege.UPDATE],
                tables=["public.users"],
            )

        assert cursor.execute.call_count == 1

        log_messages = [record.message for record in caplog.records]
        assert any(
            "SELECT, INSERT, UPDATE" in msg and "table" in msg and "public.users" in msg
            for msg in log_messages
        )

    def test_grant_table_multiple_tables(self):
        """grant_table should execute GRANT for each table."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        client.grant_table(
            grantee="sp-uuid",
            privileges=[TablePrivilege.SELECT],
            tables=[
                "public.checkpoint_migrations",
                "public.checkpoint_writes",
                "public.checkpoints",
                "public.checkpoint_blobs",
            ],
        )

        assert cursor.execute.call_count == 4

    def test_grant_table_without_schema_prefix(self):
        """grant_table should work with tables without schema prefix."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        client.grant_table(
            grantee="sp-uuid",
            privileges=[TablePrivilege.SELECT],
            tables=["users"],
        )

        assert cursor.execute.call_count == 1

    def test_grant_table_all_privileges(self, caplog):
        """grant_table with ALL should use ALL PRIVILEGES."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with caplog.at_level(logging.INFO):
            client.grant_table(
                grantee="sp-uuid",
                privileges=[TablePrivilege.ALL],
                tables=["public.users"],
            )

        log_messages = [record.message for record in caplog.records]
        assert any("ALL PRIVILEGES" in msg for msg in log_messages)


class TestLakebaseClientGrantAllSequencesInSchema:
    """Tests for LakebaseClient.grant_all_sequences_in_schema()."""

    def test_grant_all_sequences_in_schema(self, caplog):
        """grant_all_sequences_in_schema should execute GRANT on all sequences in schema."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with caplog.at_level(logging.INFO):
            client.grant_all_sequences_in_schema(
                grantee="sp-uuid",
                privileges=[
                    SequencePrivilege.USAGE,
                    SequencePrivilege.SELECT,
                    SequencePrivilege.UPDATE,
                ],
                schemas=["public"],
            )

        assert cursor.execute.call_count == 1

        log_messages = [record.message for record in caplog.records]
        assert any(
            "USAGE, SELECT, UPDATE" in msg and "all sequences in schema" in msg and "public" in msg
            for msg in log_messages
        )

    def test_grant_all_sequences_multiple_schemas(self):
        """grant_all_sequences_in_schema should execute for each schema."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        client.grant_all_sequences_in_schema(
            grantee="sp-uuid",
            privileges=[SequencePrivilege.USAGE],
            schemas=["public", "app_schema", "drizzle"],
        )

        assert cursor.execute.call_count == 3

    def test_grant_all_sequences_all_privileges(self, caplog):
        """grant_all_sequences_in_schema with ALL should use ALL PRIVILEGES."""
        pool, cursor = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with caplog.at_level(logging.INFO):
            client.grant_all_sequences_in_schema(
                grantee="sp-uuid",
                privileges=[SequencePrivilege.ALL],
                schemas=["public"],
            )

        log_messages = [record.message for record in caplog.records]
        assert any("ALL PRIVILEGES" in msg for msg in log_messages)


class TestPrivilegeFormatting:
    """Tests for privilege formatting helpers."""

    def test_format_privileges_str_single(self):
        """_format_privileges_str formats single privilege."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        result = client._format_privileges_str([TablePrivilege.SELECT])
        assert result == "SELECT"

    def test_format_privileges_str_multiple(self):
        """_format_privileges_str formats multiple privileges."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        result = client._format_privileges_str(
            [
                TablePrivilege.SELECT,
                TablePrivilege.INSERT,
                TablePrivilege.UPDATE,
            ]
        )
        assert result == "SELECT, INSERT, UPDATE"

    def test_format_privileges_str_all(self):
        """_format_privileges_str returns ALL PRIVILEGES for ALL."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        result = client._format_privileges_str([TablePrivilege.ALL])
        assert result == "ALL PRIVILEGES"

    def test_format_privileges_str_schema_privileges(self):
        """_format_privileges_str works with schema privileges."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        result = client._format_privileges_str([SchemaPrivilege.USAGE, SchemaPrivilege.CREATE])
        assert result == "USAGE, CREATE"

    def test_format_privileges_str_sequence_privileges(self):
        """_format_privileges_str works with sequence privileges."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        result = client._format_privileges_str(
            [SequencePrivilege.USAGE, SequencePrivilege.SELECT, SequencePrivilege.UPDATE]
        )
        assert result == "USAGE, SELECT, UPDATE"


class TestValidationErrors:
    """Tests for input validation and helpful error messages."""

    def test_grant_schema_empty_schemas_raises_error(self):
        """grant_schema should raise ValueError when schemas is empty."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="'schemas' cannot be empty"):
            client.grant_schema(
                grantee="sp-uuid",
                privileges=[SchemaPrivilege.USAGE],
                schemas=[],
            )

    def test_grant_schema_empty_privileges_raises_error(self):
        """grant_schema should raise ValueError when privileges is empty."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="'privileges' cannot be empty"):
            client.grant_schema(
                grantee="sp-uuid",
                privileges=[],
                schemas=["public"],
            )

    def test_grant_all_tables_empty_schemas_raises_error(self):
        """grant_all_tables_in_schema should raise ValueError when schemas is empty."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="'schemas' cannot be empty"):
            client.grant_all_tables_in_schema(
                grantee="sp-uuid",
                privileges=[TablePrivilege.SELECT],
                schemas=[],
            )

    def test_grant_all_tables_empty_privileges_raises_error(self):
        """grant_all_tables_in_schema should raise ValueError when privileges is empty."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="'privileges' cannot be empty"):
            client.grant_all_tables_in_schema(
                grantee="sp-uuid",
                privileges=[],
                schemas=["public"],
            )

    def test_grant_table_empty_tables_raises_error(self):
        """grant_table should raise ValueError when tables is empty."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="'tables' cannot be empty"):
            client.grant_table(
                grantee="sp-uuid",
                privileges=[TablePrivilege.SELECT],
                tables=[],
            )

    def test_grant_table_empty_privileges_raises_error(self):
        """grant_table should raise ValueError when privileges is empty."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="'privileges' cannot be empty"):
            client.grant_table(
                grantee="sp-uuid",
                privileges=[],
                tables=["public.users"],
            )

    def test_grant_all_sequences_empty_schemas_raises_error(self):
        """grant_all_sequences_in_schema should raise ValueError when schemas is empty."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="'schemas' cannot be empty"):
            client.grant_all_sequences_in_schema(
                grantee="sp-uuid",
                privileges=[SequencePrivilege.USAGE],
                schemas=[],
            )

    def test_grant_all_sequences_empty_privileges_raises_error(self):
        """grant_all_sequences_in_schema should raise ValueError when privileges is empty."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="'privileges' cannot be empty"):
            client.grant_all_sequences_in_schema(
                grantee="sp-uuid",
                privileges=[],
                schemas=["public"],
            )

    def test_grant_table_invalid_table_format_raises_error(self):
        """grant_table should raise ValueError for invalid table name format."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="Invalid table format"):
            client.grant_table(
                grantee="sp-uuid",
                privileges=[TablePrivilege.SELECT],
                tables=["schema."],  # Invalid: missing table name
            )

    def test_grant_table_empty_table_name_raises_error(self):
        """grant_table should raise ValueError for empty table name."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="Table name cannot be empty"):
            client.grant_table(
                grantee="sp-uuid",
                privileges=[TablePrivilege.SELECT],
                tables=[""],
            )

    def test_grant_table_whitespace_table_name_raises_error(self):
        """grant_table should raise ValueError for whitespace-only table name."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="Table name cannot be empty"):
            client.grant_table(
                grantee="sp-uuid",
                privileges=[TablePrivilege.SELECT],
                tables=["   "],
            )

    def test_create_role_empty_identity_raises_error(self):
        """create_role should raise ValueError for empty identity name."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="identity_name cannot be empty"):
            client.create_role("", "SERVICE_PRINCIPAL")

    def test_create_role_whitespace_identity_raises_error(self):
        """create_role should raise ValueError for whitespace-only identity name."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="identity_name cannot be empty"):
            client.create_role("   ", "SERVICE_PRINCIPAL")


class TestTableIdentifierParsing:
    """Tests for table identifier parsing."""

    def test_parse_table_identifier_simple_name(self):
        """_parse_table_identifier handles simple table names."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        ident = client._parse_table_identifier("users")
        assert ident is not None

    def test_parse_table_identifier_schema_qualified(self):
        """_parse_table_identifier handles schema.table format."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        ident = client._parse_table_identifier("public.users")
        assert ident is not None

    def test_parse_table_identifier_strips_whitespace(self):
        """_parse_table_identifier strips whitespace from names."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        ident = client._parse_table_identifier("  public.users  ")
        assert ident is not None

    def test_parse_table_identifier_invalid_format(self):
        """_parse_table_identifier raises ValueError for invalid formats."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="Invalid table format"):
            client._parse_table_identifier(".table")

        with pytest.raises(ValueError, match="Invalid table format"):
            client._parse_table_identifier("schema.")

    def test_parse_table_identifier_empty_raises_error(self):
        """_parse_table_identifier raises ValueError for empty string."""
        pool, _ = _make_mock_pool()
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError, match="Table name cannot be empty"):
            client._parse_table_identifier("")

        with pytest.raises(ValueError, match="Table name cannot be empty"):
            client._parse_table_identifier("   ")


class TestExecuteGrantErrorHandling:
    """Tests for _execute_grant error handling with helpful messages."""

    def test_execute_grant_handles_invalid_schema_name(self):
        """_execute_grant should raise ValueError with helpful message for invalid schema."""
        pool, cursor = _make_mock_pool()
        cursor.execute.side_effect = lakebase.psycopg.errors.InvalidSchemaName(
            'schema "nonexistent" does not exist'
        )
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError) as exc_info:
            client.grant_schema(
                grantee="sp-uuid",
                privileges=[SchemaPrivilege.USAGE],
                schemas=["nonexistent"],
            )

        error_msg = str(exc_info.value)
        assert "schema does not exist" in error_msg

    def test_execute_grant_handles_undefined_table(self):
        """_execute_grant should raise ValueError with helpful message for undefined table."""
        pool, cursor = _make_mock_pool()
        cursor.execute.side_effect = lakebase.psycopg.errors.UndefinedTable(
            'relation "public.nonexistent" does not exist'
        )
        client = LakebaseClient(pool=pool)

        with pytest.raises(ValueError) as exc_info:
            client.grant_table(
                grantee="sp-uuid",
                privileges=[TablePrivilege.SELECT],
                tables=["public.nonexistent"],
            )

        error_msg = str(exc_info.value)
        assert "table does not exist" in error_msg

    def test_execute_grant_handles_insufficient_privilege(self):
        """_execute_grant should raise PermissionError for insufficient privileges."""
        pool, cursor = _make_mock_pool()
        cursor.execute.side_effect = lakebase.psycopg.errors.InsufficientPrivilege(
            "permission denied for schema public"
        )
        client = LakebaseClient(pool=pool)

        with pytest.raises(PermissionError) as exc_info:
            client.grant_schema(
                grantee="sp-uuid",
                privileges=[SchemaPrivilege.USAGE],
                schemas=["public"],
            )

        error_msg = str(exc_info.value)
        assert "Insufficient privileges" in error_msg
        assert "CAN MANAGE" in error_msg


# =============================================================================
# Hostname Resolution Tests
# =============================================================================


def test_is_hostname_detects_database_hostname():
    """Test that _is_hostname correctly identifies database hostnames."""
    from databricks_ai_bridge.lakebase import _LakebasePoolBase

    # Should be detected as hostnames
    assert _LakebasePoolBase._is_hostname(
        "instance-f757b615-f2fd-4614-87cc-9ba35f2eeb61.database.staging.cloud.databricks.com"
    )
    assert _LakebasePoolBase._is_hostname("instance-abc123.database.prod.cloud.databricks.com")
    assert _LakebasePoolBase._is_hostname("my-db.database.example.net")

    # Should NOT be detected as hostnames (regular instance names)
    assert not _LakebasePoolBase._is_hostname("lakebase")
    assert not _LakebasePoolBase._is_hostname("my-database-instance")
    assert not _LakebasePoolBase._is_hostname("production_db")


def test_lakebase_pool_accepts_hostname(monkeypatch):
    """Test that LakebasePool accepts hostname and resolves instance name."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_workspace()

    # Mock list_database_instances to return an instance matching the hostname
    hostname = "instance-abc123.database.staging.cloud.databricks.com"
    mock_instance = MagicMock()
    mock_instance.name = "my-lakebase-instance"
    mock_instance.read_write_dns = hostname
    mock_instance.read_only_dns = None
    workspace.database.list_database_instances.return_value = [mock_instance]

    pool = LakebasePool(
        instance_name=hostname,  # Pass hostname instead of instance name
        workspace_client=workspace,
    )

    # Should have resolved to the instance name
    assert pool.instance_name == "my-lakebase-instance"
    assert pool.host == hostname

    # get_database_instance should NOT have been called (we used list instead)
    workspace.database.get_database_instance.assert_not_called()


def test_lakebase_pool_hostname_not_found_raises_error(monkeypatch):
    """Test that LakebasePool raises error when hostname doesn't match any instance."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_workspace()

    # Mock list_database_instances to return instances that don't match
    other_instance = MagicMock()
    other_instance.name = "other-instance"
    other_instance.read_write_dns = "other-host.database.staging.cloud.databricks.com"
    other_instance.read_only_dns = None
    workspace.database.list_database_instances.return_value = [other_instance]

    hostname = "instance-not-found.database.staging.cloud.databricks.com"

    with pytest.raises(ValueError, match="Unable to find database instance matching hostname"):
        LakebasePool(
            instance_name=hostname,
            workspace_client=workspace,
        )


@pytest.mark.asyncio
async def test_async_lakebase_pool_accepts_hostname(monkeypatch):
    """Test that AsyncLakebasePool accepts hostname and resolves instance name."""
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_workspace()

    # Mock list_database_instances to return an instance matching the hostname
    hostname = "instance-xyz789.database.prod.cloud.databricks.com"
    mock_instance = MagicMock()
    mock_instance.name = "prod-lakebase"
    mock_instance.read_write_dns = hostname
    mock_instance.read_only_dns = None
    workspace.database.list_database_instances.return_value = [mock_instance]

    pool = AsyncLakebasePool(
        instance_name=hostname,  # Pass hostname instead of instance name
        workspace_client=workspace,
    )

    # Should have resolved to the instance name
    assert pool.instance_name == "prod-lakebase"
    assert pool.host == hostname
