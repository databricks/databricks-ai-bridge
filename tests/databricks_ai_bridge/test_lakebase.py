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
        with pytest.raises(
            ValueError,
            match="Must provide 'pool', 'instance_name' .provisioned.",
        ):
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

    def test_create_role_handles_insufficient_privilege(self):
        """create_role should raise PermissionError with helpful message for insufficient privileges."""
        pool, cursor = _make_mock_pool()

        call_count = [0]

        def execute_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call is create_role
                raise lakebase.psycopg.errors.InsufficientPrivilege(
                    "[Databricks Auth] Permission denied to create role."
                )

        cursor.execute.side_effect = execute_side_effect

        client = LakebaseClient(pool=pool)

        with pytest.raises(PermissionError) as exc_info:
            client.create_role("sp-uuid-123", "SERVICE_PRINCIPAL")

        error_msg = str(exc_info.value)
        assert "Insufficient privileges" in error_msg
        assert "CAN MANAGE" in error_msg
        assert "Postgres Role which can create other Postgres roles" in error_msg
        assert "docs.databricks.com" in error_msg


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
        assert "Postgres Role on the Lakebase instance" in error_msg


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
        with pytest.raises(ValueError, match="Unable to resolve Lakebase provisioned instance"):
            AsyncLakebaseSQLAlchemy(
                instance_name="bad-instance",
                workspace_client=workspace,
            )


# =============================================================================
# Autoscaling (project/branch) Tests
# =============================================================================


def _make_autoscaling_workspace(
    *,
    user_name: str = "test@databricks.com",
    credential_token: str = "autoscaling-token-1",
    host: str = "autoscaling.db.host",
    endpoint_name: str = "projects/my-project/branches/my-branch/endpoints/rw-ep",
):
    """Create a mock workspace client for autoscaling (project/branch) mode."""
    workspace = MagicMock()
    workspace.current_user.me.return_value = MagicMock(user_name=user_name)

    # Mock postgres.generate_database_credential
    workspace.postgres.generate_database_credential.return_value = MagicMock(token=credential_token)

    # Mock postgres.list_endpoints → returns one READ_WRITE endpoint
    rw_endpoint = MagicMock()
    rw_endpoint.name = endpoint_name
    rw_endpoint.status.endpoint_type = "READ_WRITE"
    rw_endpoint.status.hosts.host = host
    workspace.postgres.list_endpoints.return_value = [rw_endpoint]

    return workspace


# --- Parameter validation tests ---


def test_autoscaling_requires_both_project_and_branch():
    """Passing only project without branch raises ValueError."""
    workspace = _make_autoscaling_workspace()
    with pytest.raises(ValueError, match="Both 'project' and 'branch' are required"):
        LakebasePool(
            project="my-project",
            workspace_client=workspace,
        )


def test_autoscaling_requires_both_branch_and_project():
    """Passing only branch without project raises ValueError."""
    workspace = _make_autoscaling_workspace()
    with pytest.raises(ValueError, match="'project' is required"):
        LakebasePool(
            branch="my-branch",
            workspace_client=workspace,
        )


def test_no_params_raises_error():
    """Passing no connection parameters raises ValueError."""
    workspace = _make_autoscaling_workspace()
    with pytest.raises(ValueError, match="Must provide either 'instance_name'"):
        LakebasePool(workspace_client=workspace)


def test_async_pool_no_params_raises_error():
    """AsyncLakebasePool with no connection parameters raises ValueError."""
    workspace = _make_autoscaling_workspace()
    with pytest.raises(ValueError, match="Must provide either 'instance_name'"):
        AsyncLakebasePool(workspace_client=workspace)


def test_async_pool_only_project_raises_error():
    """AsyncLakebasePool with only project raises ValueError."""
    workspace = _make_autoscaling_workspace()
    with pytest.raises(ValueError, match="Both 'project' and 'branch' are required"):
        AsyncLakebasePool(project="my-project", workspace_client=workspace)


def test_lakebase_client_no_params_raises_error():
    """LakebaseClient with no pool or connection parameters raises ValueError."""
    with pytest.raises(
        ValueError,
        match="Must provide 'pool', 'instance_name' .provisioned.",
    ):
        LakebaseClient()


def test_lakebase_client_only_branch_raises_error(monkeypatch):
    """LakebaseClient with only branch (no project) raises ValueError."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_autoscaling_workspace()
    with pytest.raises(ValueError, match="'project' is required"):
        LakebaseClient(branch="my-branch", workspace_client=workspace)


def test_async_sqlalchemy_no_params_raises_error():
    """AsyncLakebaseSQLAlchemy with no connection parameters raises ValueError."""
    workspace = _make_autoscaling_workspace()
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        with pytest.raises(ValueError, match="Must provide either 'instance_name'"):
            AsyncLakebaseSQLAlchemy(workspace_client=workspace)


def test_async_sqlalchemy_only_project_raises_error():
    """AsyncLakebaseSQLAlchemy with only project (no branch) raises ValueError."""
    workspace = _make_autoscaling_workspace()
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        with pytest.raises(ValueError, match="Both 'project' and 'branch' are required"):
            AsyncLakebaseSQLAlchemy(project="my-project", workspace_client=workspace)


def test_both_provisioned_and_autoscaling_raises_error(monkeypatch):
    """Providing both instance_name and project/branch raises ValueError."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_autoscaling_workspace(host="autoscaling.db.host")

    with pytest.raises(
        ValueError,
        match="Cannot provide 'instance_name' .provisioned. together with autoscaling parameters",
    ):
        LakebasePool(
            instance_name="my-instance",
            project="my-project",
            branch="my-branch",
            workspace_client=workspace,
        )


# --- LakebasePool autoscaling tests ---


def test_lakebase_pool_autoscaling_configures_connection_pool(monkeypatch):
    """LakebasePool with project/branch resolves host via autoscaling API."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_autoscaling_workspace(host="auto.db.host")

    pool = LakebasePool(
        project="my-project",
        branch="my-branch",
        workspace_client=workspace,
    )

    assert pool.host == "auto.db.host"
    assert pool._is_autoscaling is True
    assert pool.username == "test@databricks.com"
    assert "host=auto.db.host" in str(pool.pool.conninfo)

    workspace.postgres.list_endpoints.assert_called_once_with(
        parent="projects/my-project/branches/my-branch"
    )


def test_lakebase_pool_autoscaling_mints_token(monkeypatch):
    """Autoscaling pool uses postgres.generate_database_credential for tokens."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_autoscaling_workspace(credential_token="auto-token")

    pool = LakebasePool(
        project="my-project",
        branch="my-branch",
        workspace_client=workspace,
    )

    token = pool._get_token()
    assert token == "auto-token"
    workspace.postgres.generate_database_credential.assert_called_once_with(
        endpoint="projects/my-project/branches/my-branch/endpoints/rw-ep"
    )
    # Provisioned credential API should NOT be called
    workspace.database.generate_database_credential.assert_not_called()


def test_lakebase_pool_provisioned_mints_token(monkeypatch):
    """Provisioned pool uses database.generate_database_credential for tokens."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_workspace(credential_token="provisioned-token")

    pool = LakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    token = pool._get_token()
    assert token == "provisioned-token"
    workspace.database.generate_database_credential.assert_called_once()
    # Autoscaling credential API should NOT be called
    workspace.postgres.generate_database_credential.assert_not_called()


def test_lakebase_pool_autoscaling_no_rw_endpoint_raises(monkeypatch):
    """Raises ValueError when no READ_WRITE endpoint is found."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_autoscaling_workspace()
    # Return only a READ_ONLY endpoint
    ro_endpoint = MagicMock()
    ro_endpoint.status.endpoint_type = "READ_ONLY"
    ro_endpoint.status.hosts.host = "ro.host"
    workspace.postgres.list_endpoints.return_value = [ro_endpoint]

    with pytest.raises(ValueError, match="No READ_WRITE endpoint found"):
        LakebasePool(
            project="my-project",
            branch="my-branch",
            workspace_client=workspace,
        )


def test_lakebase_pool_autoscaling_list_endpoints_fails_raises(monkeypatch):
    """Raises ValueError when list_endpoints fails."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_autoscaling_workspace()
    workspace.postgres.list_endpoints.side_effect = Exception("Not found")

    with pytest.raises(ValueError, match="Unable to list endpoints"):
        LakebasePool(
            project="my-project",
            branch="my-branch",
            workspace_client=workspace,
        )


# --- AsyncLakebasePool autoscaling tests ---


@pytest.mark.asyncio
async def test_async_lakebase_pool_autoscaling_configures_pool(monkeypatch):
    """AsyncLakebasePool with project/branch resolves host via autoscaling API."""
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_autoscaling_workspace(host="async-auto.db.host")

    pool = AsyncLakebasePool(
        project="my-project",
        branch="my-branch",
        workspace_client=workspace,
    )

    assert pool.host == "async-auto.db.host"
    assert pool._is_autoscaling is True
    assert "host=async-auto.db.host" in str(pool.pool.conninfo)

    workspace.postgres.list_endpoints.assert_called_once_with(
        parent="projects/my-project/branches/my-branch"
    )


# --- LakebaseClient autoscaling tests ---


def test_lakebase_client_autoscaling_creates_pool(monkeypatch):
    """LakebaseClient with project/branch creates an autoscaling pool internally."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_autoscaling_workspace(host="client-auto.db.host")

    client = LakebaseClient(
        project="my-project",
        branch="my-branch",
        workspace_client=workspace,
    )

    assert client.pool.host == "client-auto.db.host"
    assert client.pool._is_autoscaling is True
    assert client._owns_pool is True


def test_lakebase_client_rejects_pool_plus_autoscaling_params(monkeypatch):
    """LakebaseClient rejects passing both pool and project/branch."""
    pool = MagicMock(spec=LakebasePool)

    with pytest.raises(ValueError, match="Provide either 'pool' or connection parameters"):
        LakebaseClient(
            pool=pool,
            project="my-project",
            branch="my-branch",
        )


# --- AsyncLakebaseSQLAlchemy autoscaling tests ---


def test_async_lakebase_sqlalchemy_autoscaling_resolves_host():
    """AsyncLakebaseSQLAlchemy with project/branch resolves via autoscaling API."""
    workspace = _make_autoscaling_workspace(host="sa-auto.db.host")
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            project="my-project",
            branch="my-branch",
            workspace_client=workspace,
        )

    assert sa.host == "sa-auto.db.host"
    assert sa._is_autoscaling is True
    workspace.postgres.list_endpoints.assert_called_once_with(
        parent="projects/my-project/branches/my-branch"
    )


def test_async_lakebase_sqlalchemy_autoscaling_mints_correct_token():
    """AsyncLakebaseSQLAlchemy in autoscaling mode uses postgres credential API."""
    workspace = _make_autoscaling_workspace(credential_token="sa-auto-token")
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            project="my-project",
            branch="my-branch",
            workspace_client=workspace,
        )

    token = sa.get_token()
    assert token == "sa-auto-token"
    workspace.postgres.generate_database_credential.assert_called_once()


def test_async_lakebase_sqlalchemy_provisioned_mints_correct_token():
    """AsyncLakebaseSQLAlchemy in provisioned mode uses database credential API."""
    workspace = _make_workspace(credential_token="sa-prov-token")
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    token = sa.get_token()
    assert token == "sa-prov-token"
    workspace.database.generate_database_credential.assert_called_once()


# =============================================================================
# Autoscaling: autoscaling_endpoint Tests
# =============================================================================


def _make_endpoint_workspace(
    *,
    user_name: str = "test@databricks.com",
    credential_token: str = "endpoint-token-1",
    host: str = "endpoint.db.host",
    endpoint_name: str = "projects/p/branches/b/endpoints/ep1",
):
    """Create a mock workspace client for autoscaling_endpoint mode."""
    workspace = MagicMock()
    workspace.current_user.me.return_value = MagicMock(user_name=user_name)
    workspace.postgres.generate_database_credential.return_value = MagicMock(token=credential_token)

    ep = MagicMock()
    ep.status.hosts.host = host
    workspace.postgres.get_endpoint.return_value = ep

    return workspace


def test_lakebase_pool_autoscaling_endpoint_configures(monkeypatch):
    """LakebasePool with autoscaling_endpoint resolves host via get_endpoint."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_endpoint_workspace(host="ep.db.host")

    pool = LakebasePool(
        autoscaling_endpoint="projects/p/branches/b/endpoints/ep1",
        workspace_client=workspace,
    )

    assert pool.host == "ep.db.host"
    assert pool._is_autoscaling is True
    assert pool._endpoint_name == "projects/p/branches/b/endpoints/ep1"
    workspace.postgres.get_endpoint.assert_called_once_with(
        name="projects/p/branches/b/endpoints/ep1"
    )


def test_lakebase_pool_autoscaling_endpoint_mints_token(monkeypatch):
    """Autoscaling endpoint pool uses postgres.generate_database_credential for tokens."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_endpoint_workspace(credential_token="ep-token")

    pool = LakebasePool(
        autoscaling_endpoint="projects/p/branches/b/endpoints/ep1",
        workspace_client=workspace,
    )

    token = pool._get_token()
    assert token == "ep-token"
    workspace.postgres.generate_database_credential.assert_called_once_with(
        endpoint="projects/p/branches/b/endpoints/ep1"
    )


@pytest.mark.asyncio
async def test_async_lakebase_pool_autoscaling_endpoint_configures(monkeypatch):
    """AsyncLakebasePool with autoscaling_endpoint resolves host via get_endpoint."""
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_endpoint_workspace(host="async-ep.db.host")

    pool = AsyncLakebasePool(
        autoscaling_endpoint="projects/p/branches/b/endpoints/ep1",
        workspace_client=workspace,
    )

    assert pool.host == "async-ep.db.host"
    assert pool._is_autoscaling is True


def test_lakebase_client_autoscaling_endpoint_creates_pool(monkeypatch):
    """LakebaseClient with autoscaling_endpoint creates pool internally."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_endpoint_workspace(host="client-ep.db.host")

    client = LakebaseClient(
        autoscaling_endpoint="projects/p/branches/b/endpoints/ep1",
        workspace_client=workspace,
    )

    assert client.pool.host == "client-ep.db.host"
    assert client.pool._is_autoscaling is True
    assert client._owns_pool is True


def test_async_lakebase_sqlalchemy_autoscaling_endpoint_resolves_host():
    """AsyncLakebaseSQLAlchemy with autoscaling_endpoint resolves via get_endpoint API."""
    workspace = _make_endpoint_workspace(host="sa-ep.db.host")
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            autoscaling_endpoint="projects/p/branches/b/endpoints/ep1",
            workspace_client=workspace,
        )

    assert sa.host == "sa-ep.db.host"
    assert sa._is_autoscaling is True
    workspace.postgres.get_endpoint.assert_called_once_with(
        name="projects/p/branches/b/endpoints/ep1"
    )


# =============================================================================
# Autoscaling: branch as resource path Tests
# =============================================================================


def test_lakebase_pool_branch_resource_path_no_project(monkeypatch):
    """LakebasePool with branch as full resource path (no project needed)."""
    TestConnectionPool = _make_connection_pool_class()
    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestConnectionPool)

    workspace = _make_autoscaling_workspace(host="branch-rp.db.host")

    pool = LakebasePool(
        branch="projects/my-project/branches/my-branch",
        workspace_client=workspace,
    )

    assert pool.host == "branch-rp.db.host"
    assert pool._is_autoscaling is True
    workspace.postgres.list_endpoints.assert_called_once_with(
        parent="projects/my-project/branches/my-branch"
    )


@pytest.mark.asyncio
async def test_async_lakebase_pool_branch_resource_path(monkeypatch):
    """AsyncLakebasePool with branch as full resource path."""
    TestAsyncConnectionPool = _make_async_connection_pool_class()
    monkeypatch.setattr(
        "databricks_ai_bridge.lakebase.AsyncConnectionPool", TestAsyncConnectionPool
    )

    workspace = _make_autoscaling_workspace(host="async-branch-rp.db.host")

    pool = AsyncLakebasePool(
        branch="projects/my-project/branches/my-branch",
        workspace_client=workspace,
    )

    assert pool.host == "async-branch-rp.db.host"
    assert pool._is_autoscaling is True


def test_async_lakebase_sqlalchemy_branch_resource_path():
    """AsyncLakebaseSQLAlchemy with branch as full resource path."""
    workspace = _make_autoscaling_workspace(host="sa-branch-rp.db.host")
    patch_engine, patch_event, _ = _make_sqlalchemy_patches(workspace)

    with patch_engine, patch_event:
        sa = AsyncLakebaseSQLAlchemy(
            branch="projects/my-project/branches/my-branch",
            workspace_client=workspace,
        )

    assert sa.host == "sa-branch-rp.db.host"
    assert sa._is_autoscaling is True


# =============================================================================
# Validation: autoscaling_endpoint conflicts
# =============================================================================


def test_autoscaling_endpoint_with_branch_raises_error():
    """autoscaling_endpoint + branch raises ValueError."""
    workspace = _make_endpoint_workspace()
    with pytest.raises(ValueError, match="Cannot provide 'autoscaling_endpoint' together with"):
        LakebasePool(
            autoscaling_endpoint="projects/p/branches/b/endpoints/ep1",
            branch="my-branch",
            workspace_client=workspace,
        )


def test_autoscaling_endpoint_with_project_raises_error():
    """autoscaling_endpoint + project raises ValueError."""
    workspace = _make_endpoint_workspace()
    with pytest.raises(ValueError, match="Cannot provide 'autoscaling_endpoint' together with"):
        LakebasePool(
            autoscaling_endpoint="projects/p/branches/b/endpoints/ep1",
            project="my-project",
            workspace_client=workspace,
        )


def test_autoscaling_endpoint_with_instance_name_raises_error():
    """autoscaling_endpoint + instance_name raises ValueError."""
    workspace = _make_endpoint_workspace()
    with pytest.raises(
        ValueError, match="Cannot provide 'instance_name' .provisioned. together with"
    ):
        LakebasePool(
            autoscaling_endpoint="projects/p/branches/b/endpoints/ep1",
            instance_name="my-instance",
            workspace_client=workspace,
        )


def test_branch_resource_path_with_project_raises_error():
    """branch as full resource path + project raises ValueError."""
    workspace = _make_autoscaling_workspace()
    with pytest.raises(ValueError, match="already a full resource path"):
        LakebasePool(
            branch="projects/my-project/branches/my-branch",
            project="my-project",
            workspace_client=workspace,
        )


def test_branch_plain_name_without_project_raises_error():
    """branch as plain name without project raises ValueError."""
    workspace = _make_autoscaling_workspace()
    with pytest.raises(ValueError, match="'project' is required"):
        LakebasePool(
            branch="my-branch",
            workspace_client=workspace,
        )


# =============================================================================
# Schema (search_path) Tests
# =============================================================================


def test_lakebase_pool_configure_sets_search_path_when_schema_specified(monkeypatch):
    """LakebasePool with schema passes a configure callback that sets search_path."""
    captured_configure = [None]

    class TestPool:
        def __init__(self, *, conninfo, connection_class, configure=None, **kwargs):
            self.conninfo = conninfo
            captured_configure[0] = configure

    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestPool)
    workspace = _make_workspace()

    LakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
        schema="my_schema",
    )

    assert captured_configure[0] is not None, "configure callback should be set when schema is provided"

    # Simulate calling the configure callback
    mock_conn = MagicMock()
    captured_configure[0](mock_conn)
    mock_conn.execute.assert_called_once()
    executed_sql = mock_conn.execute.call_args[0][0]
    # The SQL should reference the schema name
    assert "my_schema" in str(executed_sql) or "search_path" in str(executed_sql)


def test_lakebase_pool_no_configure_when_no_schema(monkeypatch):
    """LakebasePool without schema passes no configure callback."""
    captured_configure = [None]

    class TestPool:
        def __init__(self, *, conninfo, connection_class, configure=None, **kwargs):
            self.conninfo = conninfo
            captured_configure[0] = configure

    monkeypatch.setattr("databricks_ai_bridge.lakebase.ConnectionPool", TestPool)
    workspace = _make_workspace()

    LakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
    )

    assert captured_configure[0] is None, "configure callback should be None when no schema"


@pytest.mark.asyncio
async def test_async_lakebase_pool_configure_sets_search_path_when_schema_specified(monkeypatch):
    """AsyncLakebasePool with schema passes a configure callback that sets search_path."""
    captured_configure = [None]

    class TestPool:
        def __init__(self, *, conninfo, connection_class, configure=None, **kwargs):
            self.conninfo = conninfo
            captured_configure[0] = configure
            self._opened = False
            self._closed = False

        async def open(self):
            self._opened = True

        async def close(self):
            self._closed = True

    monkeypatch.setattr("databricks_ai_bridge.lakebase.AsyncConnectionPool", TestPool)
    workspace = _make_workspace()

    AsyncLakebasePool(
        instance_name="lake-instance",
        workspace_client=workspace,
        schema="my_schema",
    )

    assert captured_configure[0] is not None, "configure callback should be set when schema is provided"


def test_sqlalchemy_no_checkout_event_when_no_schema(monkeypatch):
    """AsyncLakebaseSQLAlchemy does not register a checkout event when no schema is set."""
    captured_events = []

    def mock_listens_for(engine, event_name):
        def decorator(fn):
            captured_events.append((event_name, fn))
            return fn

        return decorator

    workspace = _make_workspace()

    with (
        monkeypatch.context() as m,
    ):
        m.setattr("sqlalchemy.event.listens_for", mock_listens_for)
        m.setattr(
            "sqlalchemy.ext.asyncio.create_async_engine",
            lambda *args, **kwargs: MagicMock(sync_engine=MagicMock()),
        )

        AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
        )

    event_names = [name for name, _ in captured_events]
    assert "do_connect" in event_names
    assert "checkout" not in event_names, "checkout event should not be registered without schema"


def test_sqlalchemy_checkout_handler_sets_search_path(monkeypatch):
    """The checkout event handler should execute SET search_path with the schema name."""
    captured_events = []

    def mock_listens_for(engine, event_name):
        def decorator(fn):
            captured_events.append((event_name, fn))
            return fn

        return decorator

    workspace = _make_workspace()

    with (
        monkeypatch.context() as m,
    ):
        m.setattr("sqlalchemy.event.listens_for", mock_listens_for)
        m.setattr(
            "sqlalchemy.ext.asyncio.create_async_engine",
            lambda *args, **kwargs: MagicMock(sync_engine=MagicMock()),
        )

        AsyncLakebaseSQLAlchemy(
            instance_name="lake-instance",
            workspace_client=workspace,
            schema="my_schema",
        )

    checkout_handlers = [(name, fn) for name, fn in captured_events if name == "checkout"]
    assert len(checkout_handlers) == 1

    handler = checkout_handlers[0][1]

    # Simulate checkout event: handler(dbapi_conn, connection_record, connection_proxy)
    mock_cursor = MagicMock()
    mock_dbapi_conn = MagicMock()
    mock_dbapi_conn.cursor.return_value = mock_cursor

    handler(mock_dbapi_conn, MagicMock(), MagicMock())

    mock_cursor.execute.assert_called_once()
    mock_cursor.close.assert_called_once()
    executed_sql = mock_cursor.execute.call_args[0][0]
    assert "search_path" in str(executed_sql).lower()
    assert "my_schema" in str(executed_sql)
