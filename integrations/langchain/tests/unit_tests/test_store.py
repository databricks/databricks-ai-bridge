from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("langgraph.checkpoint.postgres")

from databricks_ai_bridge import lakebase

from databricks_langchain import AsyncDatabricksStore, DatabricksStore
from databricks_langchain.embeddings import DatabricksEmbeddings

# =============================================================================
# Synchronous DatabricksStore Tests
# ====================


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

    # Without embeddings, index_config should be None
    assert store.embeddings is None
    assert store.index_config is None


def _create_mock_workspace():
    """Helper to create a mock workspace client."""
    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.database.get_database_instance.return_value.read_write_dns = "db-host"
    workspace.current_service_principal.me.side_effect = RuntimeError("no sp")
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")
    return workspace


def test_databricks_store_with_embedding_endpoint(monkeypatch):
    """Test that embedding_endpoint creates embeddings and index_config."""
    mock_conn = MagicMock()
    test_pool = TestConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    from langgraph.store.postgres import PostgresStore

    monkeypatch.setattr(PostgresStore, "setup", MagicMock())

    workspace = _create_mock_workspace()

    with patch.object(DatabricksEmbeddings, "__init__", return_value=None) as mock_init:
        store = DatabricksStore(
            instance_name="lakebase-instance",
            workspace_client=workspace,
            embedding_endpoint="databricks-bge-large-en",
            embedding_dims=1024,
        )

        mock_init.assert_called_once_with(endpoint="databricks-bge-large-en")

    assert store.embeddings is not None
    assert store.index_config is not None
    assert store.index_config["dims"] == 1024
    assert store.index_config["embed"] is store.embeddings
    assert store.index_config["fields"] == ["$"]


def test_databricks_store_embedding_endpoint_requires_dims(monkeypatch):
    """Test that embedding_dims is required when embedding_endpoint is specified."""
    mock_conn = MagicMock()
    test_pool = TestConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    workspace = _create_mock_workspace()

    with pytest.raises(ValueError, match="embedding_dims is required"):
        DatabricksStore(
            instance_name="lakebase-instance",
            workspace_client=workspace,
            embedding_endpoint="databricks-bge-large-en",
        )


def test_databricks_store_with_store_passes_index_config(monkeypatch):
    """Test that _with_store passes index_config to PostgresStore when configured."""
    mock_conn = MagicMock()
    test_pool = TestConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    from langgraph.store.postgres import PostgresStore

    workspace = _create_mock_workspace()

    with patch.object(DatabricksEmbeddings, "__init__", return_value=None):
        store = DatabricksStore(
            instance_name="lakebase-instance",
            workspace_client=workspace,
            embedding_endpoint="databricks-bge-large-en",
            embedding_dims=1024,
        )

    # Mock PostgresStore to capture how it's called
    with patch.object(PostgresStore, "__init__", return_value=None) as mock_pg_init:
        mock_pg_store = MagicMock()
        mock_pg_store.setup = MagicMock()

        with patch(
            "databricks_langchain.store.PostgresStore", return_value=mock_pg_store
        ) as mock_pg_class:
            store.setup()

            # Verify PostgresStore was called with index config
            mock_pg_class.assert_called_once()
            call_kwargs = mock_pg_class.call_args[1]
            assert "index" in call_kwargs
            assert call_kwargs["index"]["dims"] == 1024


def test_databricks_store_warns_when_both_embeddings_and_endpoint_specified(monkeypatch):
    """Test that a warning is issued when both embeddings and embedding_endpoint are specified."""
    mock_conn = MagicMock()
    test_pool = TestConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    workspace = _create_mock_workspace()

    mock_embeddings = MagicMock(spec=DatabricksEmbeddings)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store = DatabricksStore(
            instance_name="lakebase-instance",
            workspace_client=workspace,
            embeddings=mock_embeddings,
            embedding_endpoint="databricks-bge-large-en",  # This should be ignored
            embedding_dims=1024,
        )

        # Verify warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Both 'embeddings' and 'embedding_endpoint' were specified" in str(w[0].message)

    # Verify embeddings instance takes precedence
    assert store.embeddings is mock_embeddings
    assert store.index_config is not None
    assert store.index_config["embed"] is mock_embeddings


# =============================================================================
# AsyncDatabricksStore Tests
# =============================================================================


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
async def test_async_databricks_store_configures_lakebase(monkeypatch):
    mock_conn = MagicMock()
    test_pool = TestAsyncConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    from langgraph.store.postgres import AsyncPostgresStore

    monkeypatch.setattr(AsyncPostgresStore, "setup", MagicMock())

    workspace = _create_mock_workspace()

    store = AsyncDatabricksStore(
        instance_name="lakebase-instance",
        workspace_client=workspace,
    )

    assert (
        test_pool.conninfo
        == "dbname=databricks_postgres user=test@databricks.com host=db-host port=5432 sslmode=require"
    )
    assert isinstance(store, AsyncDatabricksStore)

    # Without embeddings, index_config should be None
    assert store.embeddings is None
    assert store.index_config is None


@pytest.mark.asyncio
async def test_async_databricks_store_context_manager(monkeypatch):
    """Test async context manager opens and closes the pool."""
    mock_conn = MagicMock()
    test_pool = TestAsyncConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = _create_mock_workspace()

    async with AsyncDatabricksStore(
        instance_name="lakebase-instance",
        workspace_client=workspace,
    ) as store:
        assert test_pool._opened
        assert not test_pool._closed

    assert test_pool._closed


@pytest.mark.asyncio
async def test_async_databricks_store_with_embedding_endpoint(monkeypatch):
    """Test that embedding_endpoint creates embeddings and index_config."""
    mock_conn = MagicMock()
    test_pool = TestAsyncConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    from langgraph.store.postgres import AsyncPostgresStore

    monkeypatch.setattr(AsyncPostgresStore, "setup", MagicMock())

    workspace = _create_mock_workspace()

    with patch.object(DatabricksEmbeddings, "__init__", return_value=None) as mock_init:
        store = AsyncDatabricksStore(
            instance_name="lakebase-instance",
            workspace_client=workspace,
            embedding_endpoint="databricks-bge-large-en",
            embedding_dims=1024,
        )

        mock_init.assert_called_once_with(endpoint="databricks-bge-large-en")

    assert store.embeddings is not None
    assert store.index_config is not None
    assert store.index_config["dims"] == 1024
    assert store.index_config["embed"] is store.embeddings
    assert store.index_config["fields"] == ["$"]


@pytest.mark.asyncio
async def test_async_databricks_store_embedding_endpoint_requires_dims(monkeypatch):
    """Test that embedding_dims is required when embedding_endpoint is specified."""
    mock_conn = MagicMock()
    test_pool = TestAsyncConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = _create_mock_workspace()

    with pytest.raises(ValueError, match="embedding_dims is required"):
        AsyncDatabricksStore(
            instance_name="lakebase-instance",
            workspace_client=workspace,
            embedding_endpoint="databricks-bge-large-en",
        )


@pytest.mark.asyncio
async def test_async_databricks_store_with_store_passes_index_config(monkeypatch):
    """Test that _with_store passes index_config to AsyncPostgresStore when configured."""
    import asyncio

    mock_conn = MagicMock()
    test_pool = TestAsyncConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = _create_mock_workspace()

    with patch.object(DatabricksEmbeddings, "__init__", return_value=None):
        store = AsyncDatabricksStore(
            instance_name="lakebase-instance",
            workspace_client=workspace,
            embedding_endpoint="databricks-bge-large-en",
            embedding_dims=1024,
        )

    mock_pg_store = MagicMock()
    future = asyncio.Future()
    future.set_result(None)
    mock_pg_store.setup = MagicMock(return_value=future)

    with patch(
        "databricks_langchain.store.AsyncPostgresStore", return_value=mock_pg_store
    ) as mock_pg_class:
        await store.setup()

        # Verify AsyncPostgresStore was called with index config
        mock_pg_class.assert_called_once()
        call_kwargs = mock_pg_class.call_args[1]
        assert "index" in call_kwargs
        assert call_kwargs["index"]["dims"] == 1024


@pytest.mark.asyncio
async def test_async_databricks_store_warns_when_both_embeddings_and_endpoint_specified(
    monkeypatch,
):
    """Test that a warning is issued when both embeddings and embedding_endpoint are specified."""
    mock_conn = MagicMock()
    test_pool = TestAsyncConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = _create_mock_workspace()

    mock_embeddings = MagicMock(spec=DatabricksEmbeddings)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store = AsyncDatabricksStore(
            instance_name="lakebase-instance",
            workspace_client=workspace,
            embeddings=mock_embeddings,
            embedding_endpoint="databricks-bge-large-en",  # This should be ignored
            embedding_dims=1024,
        )

        # Verify warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Both 'embeddings' and 'embedding_endpoint' were specified" in str(w[0].message)

    # Verify embeddings instance takes precedence
    assert store.embeddings is mock_embeddings
    assert store.index_config is not None
    assert store.index_config["embed"] is mock_embeddings


@pytest.mark.asyncio
async def test_async_databricks_store_connection(monkeypatch):
    """Test getting a connection from the async store's pool."""
    mock_conn = MagicMock()
    test_pool = TestAsyncConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = _create_mock_workspace()

    async with AsyncDatabricksStore(
        instance_name="lakebase-instance",
        workspace_client=workspace,
    ) as store:
        async with store._lakebase.connection() as conn:
            assert conn == mock_conn


# =============================================================================
# V2 (autoscaling) Tests
# =============================================================================


def _create_v2_mock_workspace():
    """Create a mock workspace for V2 tests."""
    workspace = MagicMock()
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    project = MagicMock()
    project.status.display_name = "my-project"
    project.name = "projects/proj-123"
    workspace.postgres.list_projects.return_value = [project]

    endpoint = MagicMock()
    endpoint.status.endpoint_type = "ENDPOINT_TYPE_READ_WRITE"
    endpoint.status.hosts.host = "v2.host"
    endpoint.name = "projects/proj-123/branches/branch-456/endpoints/ep-789"
    workspace.postgres.list_endpoints.return_value = [endpoint]

    workspace.postgres.generate_database_credential.return_value = MagicMock(token="v2-token")
    return workspace


def test_databricks_store_v2_configures_lakebase(monkeypatch):
    """V2 DatabricksStore creates pool with URI conninfo."""
    mock_conn = MagicMock()
    test_pool = TestConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    workspace = _create_v2_mock_workspace()
    store = DatabricksStore(
        project="my-project",
        branch="branch-456",
        workspace_client=workspace,
    )

    expected = "dbname=databricks_postgres user=test@databricks.com host=v2.host port=5432 sslmode=require"
    assert test_pool.conninfo == expected
    assert store._lakebase._is_v2 is True


@pytest.mark.asyncio
async def test_async_databricks_store_v2_configures_lakebase(monkeypatch):
    """V2 AsyncDatabricksStore creates pool with URI conninfo."""
    mock_conn = MagicMock()
    test_pool = TestAsyncConnectionPool(connection_value=mock_conn)
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = _create_v2_mock_workspace()
    store = AsyncDatabricksStore(
        project="my-project",
        branch="branch-456",
        workspace_client=workspace,
    )

    expected = "dbname=databricks_postgres user=test@databricks.com host=v2.host port=5432 sslmode=require"
    assert test_pool.conninfo == expected
    assert store._lakebase._is_v2 is True
