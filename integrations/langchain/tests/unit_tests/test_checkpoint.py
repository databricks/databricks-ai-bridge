from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("databricks_ai_bridge.lakebase")
pytest.importorskip("langgraph.checkpoint.postgres")

from databricks_ai_bridge import lakebase

from databricks_langchain import AsyncCheckpointSaver, CheckpointSaver


async def _async_noop():
    """No-op coroutine used to mock async setup() in tests."""
    pass


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
    monkeypatch.setattr(AsyncCheckpointSaver, "setup", lambda self: _async_noop())

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
    monkeypatch.setattr(AsyncCheckpointSaver, "setup", lambda self: _async_noop())

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


# =============================================================================
# Autoscaling (project/branch) Tests
# =============================================================================


def _create_autoscaling_workspace():
    """Helper to create a mock workspace client for autoscaling mode."""
    workspace = MagicMock()
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")
    workspace.postgres.generate_database_credential.return_value = MagicMock(
        token="autoscaling-token"
    )
    rw_endpoint = MagicMock()
    rw_endpoint.name = "projects/p/branches/b/endpoints/rw"
    rw_endpoint.status.endpoint_type = "READ_WRITE"
    rw_endpoint.status.hosts.host = "auto-db-host"
    workspace.postgres.list_endpoints.return_value = [rw_endpoint]
    return workspace


def test_checkpoint_saver_autoscaling_configures_lakebase(monkeypatch):
    test_pool = TestConnectionPool(connection_value="lake-conn")
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    workspace = _create_autoscaling_workspace()

    saver = CheckpointSaver(
        project="my-project",
        branch="my-branch",
        workspace_client=workspace,
    )

    assert "host=auto-db-host" in test_pool.conninfo
    assert saver._lakebase._is_autoscaling is True
    workspace.postgres.list_endpoints.assert_called_once_with(
        parent="projects/my-project/branches/my-branch"
    )


@pytest.mark.asyncio
async def test_async_checkpoint_saver_autoscaling_configures_lakebase(monkeypatch):
    test_pool = TestAsyncConnectionPool(connection_value="async-lake-conn")
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = _create_autoscaling_workspace()

    saver = AsyncCheckpointSaver(
        project="my-project",
        branch="my-branch",
        workspace_client=workspace,
    )

    assert "host=auto-db-host" in test_pool.conninfo
    assert saver._lakebase._is_autoscaling is True


@pytest.mark.asyncio
async def test_async_checkpoint_saver_autoscaling_context_manager(monkeypatch):
    test_pool = TestAsyncConnectionPool(connection_value="async-lake-conn")
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)
    monkeypatch.setattr(AsyncCheckpointSaver, "setup", lambda self: _async_noop())

    workspace = _create_autoscaling_workspace()

    async with AsyncCheckpointSaver(
        project="my-project",
        branch="my-branch",
        workspace_client=workspace,
    ) as saver:
        assert test_pool._opened
        assert saver._lakebase._is_autoscaling is True

    assert test_pool._closed


# =============================================================================
# Validation: missing parameters
# =============================================================================


def test_checkpoint_saver_no_params_raises_error(monkeypatch):
    """CheckpointSaver with no connection parameters raises ValueError."""
    test_pool = TestConnectionPool()
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    workspace = MagicMock()
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    with pytest.raises(ValueError, match="Must provide either 'instance_name'"):
        CheckpointSaver(workspace_client=workspace)


def test_checkpoint_saver_only_project_raises_error(monkeypatch):
    """CheckpointSaver with only project (no branch) raises ValueError."""
    test_pool = TestConnectionPool()
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    workspace = MagicMock()
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    with pytest.raises(ValueError, match="Both 'project' and 'branch' are required"):
        CheckpointSaver(project="my-project", workspace_client=workspace)


@pytest.mark.asyncio
async def test_async_checkpoint_saver_no_params_raises_error(monkeypatch):
    """AsyncCheckpointSaver with no connection parameters raises ValueError."""
    test_pool = TestAsyncConnectionPool()
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = MagicMock()
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")

    with pytest.raises(ValueError, match="Must provide either 'instance_name'"):
        AsyncCheckpointSaver(workspace_client=workspace)


# =============================================================================
# Autoscaling (direct endpoint) Tests
# =============================================================================


def _create_endpoint_workspace():
    """Helper to create a mock workspace client for direct endpoint mode."""
    workspace = MagicMock()
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")
    workspace.postgres.generate_database_credential.return_value = MagicMock(
        token="endpoint-token"
    )
    ep = MagicMock()
    ep.host = "ep-db-host"
    workspace.postgres.get_endpoint.return_value = ep
    return workspace


def test_checkpoint_saver_endpoint_configures_lakebase(monkeypatch):
    test_pool = TestConnectionPool(connection_value="lake-conn")
    monkeypatch.setattr(lakebase, "ConnectionPool", test_pool)

    workspace = _create_endpoint_workspace()

    saver = CheckpointSaver(
        endpoint="projects/p/branches/b/endpoints/rw",
        workspace_client=workspace,
    )

    assert "host=ep-db-host" in test_pool.conninfo
    assert saver._lakebase._is_autoscaling is True
    workspace.postgres.get_endpoint.assert_called_once_with(
        name="projects/p/branches/b/endpoints/rw"
    )


@pytest.mark.asyncio
async def test_async_checkpoint_saver_endpoint_configures_lakebase(monkeypatch):
    test_pool = TestAsyncConnectionPool(connection_value="async-lake-conn")
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)

    workspace = _create_endpoint_workspace()

    saver = AsyncCheckpointSaver(
        endpoint="projects/p/branches/b/endpoints/rw",
        workspace_client=workspace,
    )

    assert "host=ep-db-host" in test_pool.conninfo
    assert saver._lakebase._is_autoscaling is True


@pytest.mark.asyncio
async def test_async_checkpoint_saver_endpoint_context_manager(monkeypatch):
    test_pool = TestAsyncConnectionPool(connection_value="async-lake-conn")
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", test_pool)
    monkeypatch.setattr(AsyncCheckpointSaver, "setup", lambda self: _async_noop())

    workspace = _create_endpoint_workspace()

    async with AsyncCheckpointSaver(
        endpoint="projects/p/branches/b/endpoints/rw",
        workspace_client=workspace,
    ) as saver:
        assert test_pool._opened
        assert saver._lakebase._is_autoscaling is True

    assert test_pool._closed
