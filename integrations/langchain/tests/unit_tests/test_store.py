from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("langgraph.checkpoint.postgres")

from databricks_ai_bridge import lakebase

from databricks_langchain import DatabricksStore


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


def test_databricks_store_configures_lakebase(monkeypatch):
    mock_conn = MagicMock()
    test_pool = TestConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    from langgraph.store.postgres import PostgresStore

    monkeypatch.setattr(PostgresStore, "setup", MagicMock())

    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.database.get_database_instance.return_value.read_write_dns = "db-host"
    workspace.current_service_principal.me.side_effect = RuntimeError("no sp")
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    store = DatabricksStore(
        instance_name="lakebase-instance",
        workspace_client=workspace,
    )

    assert (
        test_pool.conninfo
        == "dbname=databricks_postgres user=test@databricks.com host=db-host port=5432 sslmode=require"
    )
    assert isinstance(store, DatabricksStore)

    with store._lakebase.connection() as conn:
        assert conn == mock_conn


class TestDatabricksStoreNamespace:
    """Test the DatabricksStore.namespace() static method."""

    def test_namespace_with_email(self):
        """Test namespace normalization with a typical email address."""
        result = DatabricksStore.namespace("first.last@databricks.com")
        assert result == ("users", "first-last-at-databricks-com")

    def test_namespace_with_uppercase(self):
        """Test that uppercase letters are converted to lowercase."""
        result = DatabricksStore.namespace("FIRST.LAST@DATABRICKS.COM")
        assert result == ("users", "first-last-at-databricks-com")

    def test_namespace_with_empty_identifier(self):
        """Test that empty identifier returns 'anon'."""
        result = DatabricksStore.namespace("")
        assert result == ("users", "anon")

    def test_namespace_with_whitespace_only(self):
        """Test that whitespace-only identifier returns 'anon'."""
        result = DatabricksStore.namespace("   ")
        assert result == ("users", "anon")

    def test_namespace_with_custom_prefix(self):
        """Test namespace with a custom prefix."""
        result = DatabricksStore.namespace("user123", prefix="agents")
        assert result == ("agents", "user123")

    def test_namespace_with_special_characters(self):
        """Test that special characters are replaced with dashes."""
        result = DatabricksStore.namespace("user!name@test#site.com")
        assert result == ("users", "user-name-at-test-site-com")

    def test_namespace_with_leading_trailing_special_chars(self):
        """Test that leading/trailing dashes are stripped."""
        result = DatabricksStore.namespace("!!!user@test.com!!!")
        assert result == ("users", "user-at-test-com")

    def test_namespace_with_underscores_and_hyphens(self):
        """Test that underscores and hyphens are preserved."""
        result = DatabricksStore.namespace("user_name-123")
        assert result == ("users", "user_name-123")

    def test_namespace_with_numbers(self):
        """Test that numbers are preserved."""
        result = DatabricksStore.namespace("user123@test456.com")
        assert result == ("users", "user123-at-test456-com")

    def test_namespace_with_long_identifier(self):
        """Test that long identifiers are truncated with hash suffix."""
        # Create an identifier longer than 64 characters
        long_identifier = "a" * 70 + "@example.com"
        result = DatabricksStore.namespace(long_identifier)

        # Should be exactly 64 characters
        assert len(result[1]) == 64

        # Should start with truncated original and end with hash
        assert result[1].startswith("a" * 47)
        assert "-" in result[1]

        # Verify it's a valid namespace tuple
        assert result[0] == "users"
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_namespace_with_at_and_special_characters(self):
        """Test identifier with at and special characters."""
        result = DatabricksStore.namespace("@@@###$$$")
        assert result == ("users", "at-at-at")

    def test_namespace_with_mixed_valid_invalid_chars(self):
        """Test identifier with mix of valid and invalid characters."""
        result = DatabricksStore.namespace("test$user%123@site&domain.com")
        assert result == ("users", "test-user-123-at-site-domain-com")

    def test_namespace_with_unicode_characters(self):
        """Test that unicode characters are removed or replaced."""
        result = DatabricksStore.namespace("user\u00e9@test.com")  # user with é
        assert result[0] == "users"
        assert "at-test-com" in result[1]

    def test_namespace_returns_tuple(self):
        """Test that namespace always returns a tuple."""
        result = DatabricksStore.namespace("test@example.com")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)
