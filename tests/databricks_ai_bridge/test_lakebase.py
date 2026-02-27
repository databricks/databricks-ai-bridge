from __future__ import annotations

import logging
import re
from unittest.mock import MagicMock

import pytest

pytest.importorskip("databricks_ai_bridge.lakebase")

import databricks_ai_bridge.lakebase as lakebase
from databricks_ai_bridge.lakebase import (
    AsyncLakebasePool,
    AsyncLakebaseSQLAlchemy,
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
        record.levelno == logging.INFO and re.search(r"cache=900s$", record.getMessage())
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
        record.levelno == logging.INFO and re.search(r"cache=900s$", record.getMessage())
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
        with pytest.raises(ValueError, match="Must provide either 'pool' or connection parameters"):
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
# AsyncLakebaseSQLAlchemy Tests
# =============================================================================

pytest.importorskip("sqlalchemy")


def _make_sqlalchemy_patches(workspace):
    """Return a context manager that patches SQLAlchemy internals for AsyncLakebaseSQLAlchemy."""
    from unittest.mock import patch

    mock_engine = MagicMock(sync_engine=MagicMock())

    return (
        patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=mock_engine,
        ),
        patch(
            "sqlalchemy.event.listens_for",
            return_value=lambda f: f,
        ),
        mock_engine,
    )


def test_async_lakebase_sqlalchemy_resolves_host():
    """Test that AsyncLakebaseSQLAlchemy resolves the Lakebase host from instance name."""
    workspace = _make_workspace()
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    assert sa.host == "db.host"
    workspace.database.get_database_instance.assert_called_once_with("lake-instance")


def test_async_lakebase_sqlalchemy_infers_username():
    """Test that AsyncLakebaseSQLAlchemy infers the username from workspace client."""
    workspace = _make_workspace(user_name="alice@databricks.com")
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    assert sa.username == "alice@databricks.com"


def test_async_lakebase_sqlalchemy_engine_property():
    """Test that engine property returns the created AsyncEngine."""
    workspace = _make_workspace()
    patch_engine, patch_event, mock_engine = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    assert sa.engine is mock_engine


def test_async_lakebase_sqlalchemy_creates_engine_with_correct_url():
    """Test that the engine is created with the correct SQLAlchemy URL."""
    from unittest.mock import patch

    workspace = _make_workspace()
    mock_engine = MagicMock(sync_engine=MagicMock())

    with (
        patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=mock_engine,
        ) as mock_create,
        patch(
            "sqlalchemy.event.listens_for",
            return_value=lambda f: f,
        ),
    ):
        AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    url = mock_create.call_args[0][0]
    assert url.drivername == "postgresql+psycopg"
    assert url.username == "test@databricks.com"
    assert url.host == "db.host"
    assert url.port == 5432
    assert url.database == "databricks_postgres"


def test_async_lakebase_sqlalchemy_passes_extra_engine_kwargs():
    """Test that additional kwargs are forwarded to create_async_engine."""
    from unittest.mock import patch

    workspace = _make_workspace()
    mock_engine = MagicMock(sync_engine=MagicMock())

    with (
        patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=mock_engine,
        ) as mock_create,
        patch(
            "sqlalchemy.event.listens_for",
            return_value=lambda f: f,
        ),
    ):
        AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
            echo=True,
            pool_pre_ping=True,
        )

    call_kwargs = mock_create.call_args[1]
    assert call_kwargs["echo"] is True
    assert call_kwargs["pool_pre_ping"] is True


def test_async_lakebase_sqlalchemy_do_connect_injects_token():
    """Test that the do_connect handler injects the OAuth token into cparams."""
    from unittest.mock import patch

    workspace = _make_workspace(credential_token="my-secret-token")
    mock_engine = MagicMock(sync_engine=MagicMock())

    captured_handler = None

    def capture_handler(engine, event_name):
        def decorator(fn):
            nonlocal captured_handler
            captured_handler = fn
            return fn

        return decorator

    with (
        patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=mock_engine,
        ),
        patch(
            "sqlalchemy.event.listens_for",
            side_effect=capture_handler,
        ),
    ):
        AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    assert captured_handler is not None
    cparams = {}
    captured_handler(None, None, None, cparams)
    assert cparams["password"] == "my-secret-token"


def test_async_lakebase_sqlalchemy_get_token_caches():
    """Test that get_token returns cached token on repeated calls."""
    workspace = _make_workspace()
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    token1 = sa.get_token()
    token2 = sa.get_token()

    assert token1 == token2 == "token-1"
    assert workspace.database.generate_database_credential.call_count == 1


def test_async_lakebase_sqlalchemy_get_token_refreshes_after_expiry(monkeypatch):
    """Test that get_token mints a new token after cache expiry."""
    import time

    call_count = []

    def mock_generate_credential(**kwargs):
        call_count.append(1)
        return MagicMock(token=f"token-{len(call_count)}")

    workspace = _make_workspace()
    workspace.database.generate_database_credential = mock_generate_credential
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
            token_cache_duration_seconds=1,
        )

    test_time = [100.0]
    monkeypatch.setattr(time, "time", lambda: test_time[0])

    token1 = sa.get_token()
    assert token1 == "token-1"

    # Within cache window
    test_time[0] = 100.5
    token2 = sa.get_token()
    assert token2 == "token-1"
    assert len(call_count) == 1

    # After cache expiry
    test_time[0] = 101.5
    token3 = sa.get_token()
    assert token3 == "token-2"
    assert len(call_count) == 2


def test_async_lakebase_sqlalchemy_invalid_instance_raises():
    """Test that an invalid instance name raises ValueError."""
    workspace = _make_workspace()
    workspace.database.get_database_instance.side_effect = Exception("Not found")
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        with pytest.raises(ValueError, match="Unable to resolve Lakebase instance"):
            AsyncLakebaseSQLAlchemy(
                instance_name="bad-instance",
                workspace_client=workspace,
            )


# =============================================================================
# V2 (autoscaling) Tests
# =============================================================================


def _make_v2_workspace(
    *,
    user_name: str = "test@databricks.com",
    credential_token: str = "v2-token-1",
    project_display_name: str = "my-project",
    project_resource_name: str = "projects/proj-123",
    branch_id: str = "branch-456",
    endpoint_host: str = "v2.host",
    endpoint_name: str = "projects/proj-123/branches/branch-456/endpoints/ep-789",
):
    """Create a mock workspace for V2 (autoscaling) tests.

    Mirrors real SDK structure where display_name is on proj.status,
    endpoint_type is on ep.status, and host is on ep.status.hosts.host.
    """
    workspace = MagicMock()
    workspace.current_user.me.return_value = MagicMock(user_name=user_name)

    # Mock postgres.list_projects() — display_name is on proj.status
    project = MagicMock()
    project.status.display_name = project_display_name
    project.name = project_resource_name
    workspace.postgres.list_projects.return_value = [project]

    # Mock postgres.list_endpoints() — endpoint_type and host are on ep.status
    endpoint = MagicMock()
    endpoint.status.endpoint_type = "ENDPOINT_TYPE_READ_WRITE"
    endpoint.status.hosts.host = endpoint_host
    endpoint.name = endpoint_name
    workspace.postgres.list_endpoints.return_value = [endpoint]

    # Mock postgres.generate_database_credential()
    workspace.postgres.generate_database_credential.return_value = MagicMock(
        token=credential_token
    )

    return workspace


class TestV2Validation:
    """Tests for V2 parameter validation."""

    def test_both_v1_and_v2_raises(self):
        """Providing both instance_name and project/branch should raise."""
        workspace = _make_workspace()
        with pytest.raises(ValueError, match="not both"):
            LakebasePool(
                instance_name="my-instance",
                project="my-project",
                branch="branch-1",
                workspace_client=workspace,
            )

    def test_neither_v1_nor_v2_raises(self):
        """Providing neither instance_name nor project/branch should raise."""
        with pytest.raises(ValueError, match="Must provide"):
            LakebasePool(workspace_client=MagicMock())

    def test_v2_project_without_branch_raises(self):
        """V2 requires both project and branch."""
        with pytest.raises(ValueError, match="Both 'project' and 'branch' are required"):
            LakebasePool(project="my-project", workspace_client=MagicMock())

    def test_v2_branch_without_project_raises(self):
        """V2 requires both project and branch."""
        with pytest.raises(ValueError, match="Both 'project' and 'branch' are required"):
            LakebasePool(branch="branch-1", workspace_client=MagicMock())


class TestV2Pool:
    """Tests for V2 LakebasePool."""

    def test_v2_pool_resolves_host(self, monkeypatch):
        """V2 pool resolves host from endpoint."""
        TestConnectionPool = _make_connection_pool_class()
        monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

        workspace = _make_v2_workspace(endpoint_host="v2.lakebase.host")
        pool = LakebasePool(
            project="my-project",
            branch="branch-456",
            workspace_client=workspace,
        )

        assert pool.host == "v2.lakebase.host"

    def test_v2_pool_conninfo_uses_uri_format(self, monkeypatch):
        """V2 conninfo should use URI format with URL-encoded username."""
        TestConnectionPool = _make_connection_pool_class()
        monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

        workspace = _make_v2_workspace(
            user_name="user@databricks.com",
            endpoint_host="v2.host",
        )
        pool = LakebasePool(
            project="my-project",
            branch="branch-456",
            workspace_client=workspace,
        )

        expected = "dbname=databricks_postgres user=user@databricks.com host=v2.host port=5432 sslmode=require"
        assert pool.pool.conninfo == expected

    def test_v2_pool_mints_token_via_postgres_service(self, monkeypatch):
        """V2 pool mints token via postgres.generate_database_credential."""
        TestConnectionPool = _make_connection_pool_class()
        monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

        workspace = _make_v2_workspace(credential_token="v2-secret")
        pool = LakebasePool(
            project="my-project",
            branch="branch-456",
            workspace_client=workspace,
        )

        token = pool._get_token()
        assert token == "v2-secret"
        workspace.postgres.generate_database_credential.assert_called_once_with(
            endpoint="projects/proj-123/branches/branch-456/endpoints/ep-789",
        )

    def test_v2_pool_project_not_found_raises(self, monkeypatch):
        """V2 pool raises ValueError when project not found."""
        TestConnectionPool = _make_connection_pool_class()
        monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

        workspace = _make_v2_workspace()
        workspace.postgres.list_projects.return_value = []

        with pytest.raises(ValueError, match="not found"):
            LakebasePool(
                project="nonexistent-project",
                branch="branch-456",
                workspace_client=workspace,
            )

    def test_v2_pool_no_rw_endpoint_raises(self, monkeypatch):
        """V2 pool raises ValueError when no READ_WRITE endpoint found."""
        TestConnectionPool = _make_connection_pool_class()
        monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

        workspace = _make_v2_workspace()
        # Return a READ_ONLY endpoint instead of READ_WRITE
        ro_endpoint = MagicMock()
        ro_endpoint.status.endpoint_type = "ENDPOINT_TYPE_READ_ONLY"
        ro_endpoint.status.hosts.host = "ro.host"
        ro_endpoint.name = "ep-ro"
        workspace.postgres.list_endpoints.return_value = [ro_endpoint]

        with pytest.raises(ValueError, match="No READ_WRITE endpoint"):
            LakebasePool(
                project="my-project",
                branch="branch-456",
                workspace_client=workspace,
            )

    def test_v2_pool_token_cache_and_refresh(self, monkeypatch):
        """V2 pool: cached token is reused, expired token is re-minted."""
        TestConnectionPool = _make_connection_pool_class()
        monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

        token_call_count = []

        def mock_generate_credential(**kwargs):
            token_call_count.append(1)
            return MagicMock(token=f"v2-token-{len(token_call_count)}")

        workspace = _make_v2_workspace()
        workspace.postgres.generate_database_credential = mock_generate_credential

        pool = LakebasePool(
            project="my-project",
            branch="branch-456",
            workspace_client=workspace,
            token_cache_duration_seconds=1,
        )

        import time

        test_time = [100.0]

        def mock_time():
            return test_time[0]

        monkeypatch.setattr(time, "time", mock_time)

        # First call mints
        token1 = pool._get_token()
        assert token1 == "v2-token-1"
        assert len(token_call_count) == 1

        # Within cache window - reuses
        test_time[0] = 100.5
        token2 = pool._get_token()
        assert token2 == "v2-token-1"
        assert len(token_call_count) == 1

        # After expiry - re-mints
        test_time[0] = 101.5
        token3 = pool._get_token()
        assert token3 == "v2-token-2"
        assert len(token_call_count) == 2


class TestV2AsyncPool:
    """Tests for V2 AsyncLakebasePool."""

    @pytest.mark.asyncio
    async def test_v2_async_pool_resolves_host(self, monkeypatch):
        TestAsyncConnectionPool = _make_async_connection_pool_class()
        monkeypatch.setattr(
            "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
        )

        workspace = _make_v2_workspace(endpoint_host="v2-async.host")
        pool = AsyncLakebasePool(
            project="my-project",
            branch="branch-456",
            workspace_client=workspace,
        )

        assert pool.host == "v2-async.host"

    @pytest.mark.asyncio
    async def test_v2_async_pool_conninfo_uses_uri_format(self, monkeypatch):
        TestAsyncConnectionPool = _make_async_connection_pool_class()
        monkeypatch.setattr(
            "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
        )

        workspace = _make_v2_workspace(user_name="user@databricks.com")
        pool = AsyncLakebasePool(
            project="my-project",
            branch="branch-456",
            workspace_client=workspace,
        )

        expected = "dbname=databricks_postgres user=user@databricks.com host=v2.host port=5432 sslmode=require"
        assert pool.pool.conninfo == expected

    @pytest.mark.asyncio
    async def test_v2_async_pool_mints_token_via_postgres_service(self, monkeypatch):
        TestAsyncConnectionPool = _make_async_connection_pool_class()
        monkeypatch.setattr(
            "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
        )

        workspace = _make_v2_workspace(credential_token="v2-async-token")
        pool = AsyncLakebasePool(
            project="my-project",
            branch="branch-456",
            workspace_client=workspace,
        )

        token = await pool._get_token_async()
        assert token == "v2-async-token"
        workspace.postgres.generate_database_credential.assert_called_once_with(
            endpoint="projects/proj-123/branches/branch-456/endpoints/ep-789",
        )


class TestV2LakebaseClient:
    """Tests for V2 LakebaseClient."""

    def test_v2_client_creates_pool_with_v2_params(self, monkeypatch):
        """V2 client creates an internal pool with project+branch."""
        TestConnectionPool = _make_connection_pool_class()
        monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

        workspace = _make_v2_workspace()
        client = LakebaseClient(
            project="my-project",
            branch="branch-456",
            workspace_client=workspace,
        )

        assert client.pool.host == "v2.host"
        assert client.pool._is_v2 is True

    def test_client_pool_and_project_raises(self):
        """Providing both pool and project should raise."""
        pool = MagicMock(spec=LakebasePool)
        with pytest.raises(ValueError, match="not both"):
            LakebaseClient(pool=pool, project="my-project", branch="b1")
