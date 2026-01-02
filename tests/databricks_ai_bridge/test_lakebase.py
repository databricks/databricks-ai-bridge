from __future__ import annotations

import logging
import re
from unittest.mock import MagicMock

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")

import databricks_ai_bridge.lakebase as lakebase
from databricks_ai_bridge.lakebase import (
    LakebaseClient,
    LakebasePool,
    SchemaPrivilege,
    TablePrivilege,
)


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
