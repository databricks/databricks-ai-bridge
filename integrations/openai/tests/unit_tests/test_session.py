"""Unit tests for AsyncDatabricksSession."""

from unittest.mock import MagicMock, patch

import pytest

try:
    from databricks_ai_bridge.lakebase import _LakebaseBase  # noqa: F401
    from psycopg.rows import DictRow  # noqa: F401
    from sqlalchemy.ext.asyncio import AsyncEngine  # noqa: F401
except ImportError as e:
    raise ImportError(
        "AsyncDatabricksSession tests require databricks-openai[memory]. "
        "Please install with: pip install databricks-openai[memory]"
    ) from e


@pytest.fixture(autouse=True)
def clear_engine_cache():
    """Clear AsyncDatabricksSession engine cache before each test."""
    try:
        from databricks_openai.agents.session import AsyncDatabricksSession

        AsyncDatabricksSession._lakebase_sql_alchemy_cache.clear()
    except ImportError:
        pass
    yield
    try:
        from databricks_openai.agents.session import AsyncDatabricksSession

        AsyncDatabricksSession._lakebase_sql_alchemy_cache.clear()
    except ImportError:
        pass


@pytest.fixture
def mock_workspace_client():
    """Create a mock WorkspaceClient."""
    mock_client = MagicMock()
    mock_client.config.host = "https://test.databricks.com"

    # Mock current_user.me() for username inference
    mock_user = MagicMock()
    mock_user.user_name = "test_user@databricks.com"
    mock_client.current_user.me.return_value = mock_user

    # Mock database.get_database_instance() for host resolution
    mock_instance = MagicMock()
    mock_instance.read_write_dns = "test-instance.lakebase.databricks.com"
    mock_client.database.get_database_instance.return_value = mock_instance

    # Mock database.generate_database_credential() for token minting
    mock_credential = MagicMock()
    mock_credential.token = "test-oauth-token"
    mock_client.database.generate_database_credential.return_value = mock_credential

    return mock_client


@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy AsyncEngine."""
    mock_eng = MagicMock()
    mock_eng.sync_engine = MagicMock()
    return mock_eng


@pytest.fixture
def mock_event_listens_for():
    """Create a mock for event.listens_for that captures the handler."""

    def create_decorator(engine, event_name):
        def decorator(fn):
            return fn

        return decorator

    return create_decorator


class TestAsyncLakebaseSQLAlchemy:
    """Tests for AsyncLakebaseSQLAlchemy class."""

    def test_init_resolves_host(self, mock_workspace_client):
        """Test that initialization resolves the Lakebase host."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch("sqlalchemy.event.listens_for", return_value=lambda f: f),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=MagicMock(sync_engine=MagicMock()),
            ),
        ):
            from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy

            lakebase = AsyncLakebaseSQLAlchemy(
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            assert lakebase.host == "test-instance.lakebase.databricks.com"
            mock_workspace_client.database.get_database_instance.assert_called_once_with(
                "test-instance"
            )

    def test_init_infers_username(self, mock_workspace_client):
        """Test that initialization infers the username."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch("sqlalchemy.event.listens_for", return_value=lambda f: f),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=MagicMock(sync_engine=MagicMock()),
            ),
        ):
            from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy

            lakebase = AsyncLakebaseSQLAlchemy(
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            assert lakebase.username == "test_user@databricks.com"
            mock_workspace_client.current_user.me.assert_called()

    def test_get_token_mints_new_token(self, mock_workspace_client):
        """Test that get_token mints a new token when cache is empty."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch("sqlalchemy.event.listens_for", return_value=lambda f: f),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=MagicMock(sync_engine=MagicMock()),
            ),
        ):
            from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy

            lakebase = AsyncLakebaseSQLAlchemy(
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            token = lakebase.get_token()

            assert token == "test-oauth-token"
            mock_workspace_client.database.generate_database_credential.assert_called_once()

    def test_get_token_returns_cached_token(self, mock_workspace_client):
        """Test that get_token returns cached token when valid."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch("sqlalchemy.event.listens_for", return_value=lambda f: f),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=MagicMock(sync_engine=MagicMock()),
            ),
        ):
            from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy

            lakebase = AsyncLakebaseSQLAlchemy(
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            # First call - mints token
            token1 = lakebase.get_token()
            # Second call - should return cached
            token2 = lakebase.get_token()

            assert token1 == token2 == "test-oauth-token"
            # Should only mint once
            assert mock_workspace_client.database.generate_database_credential.call_count == 1

    def test_get_token_refreshes_expired_token(self, mock_workspace_client):
        """Test that get_token refreshes token when expired."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch("sqlalchemy.event.listens_for", return_value=lambda f: f),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=MagicMock(sync_engine=MagicMock()),
            ),
        ):
            import time

            from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy

            lakebase = AsyncLakebaseSQLAlchemy(
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                token_cache_duration_seconds=0,  # Immediate expiration
            )

            # First call
            lakebase.get_token()
            # Wait a tiny bit
            time.sleep(0.01)
            # Second call - should mint new token
            lakebase.get_token()

            # Should mint twice due to expiration
            assert mock_workspace_client.database.generate_database_credential.call_count == 2

    def test_init_raises_on_invalid_instance(self, mock_workspace_client):
        """Test that initialization raises ValueError for invalid instance."""
        mock_workspace_client.database.get_database_instance.side_effect = Exception(
            "Instance not found"
        )

        with patch(
            "databricks_ai_bridge.lakebase.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy

            with pytest.raises(ValueError, match="Unable to resolve Lakebase instance"):
                AsyncLakebaseSQLAlchemy(
                    instance_name="invalid-instance",
                    workspace_client=mock_workspace_client,
                )

    def test_engine_property_returns_engine(self, mock_workspace_client):
        """Test that engine property returns the created engine."""
        mock_eng = MagicMock(sync_engine=MagicMock())
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch("sqlalchemy.event.listens_for", return_value=lambda f: f),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_eng,
            ),
        ):
            from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy

            lakebase = AsyncLakebaseSQLAlchemy(
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            assert lakebase.engine is mock_eng


class TestAsyncDatabricksSessionInit:
    """Tests for AsyncDatabricksSession initialization."""

    def test_init_creates_engine_with_correct_url(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test that initialization creates engine with correct connection URL."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ) as mock_create_engine,
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            # Verify engine was created with correct URL object
            call_args = mock_create_engine.call_args
            url = call_args[0][0]
            # URL.create() returns a URL object, check its properties
            assert url.drivername == "postgresql+psycopg"
            assert url.username == "test_user@databricks.com"
            assert url.host == "test-instance.lakebase.databricks.com"
            assert url.port == 5432
            assert url.database == "databricks_postgres"

    def test_init_uses_pool_recycle(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test that initialization uses pool_recycle for connection recycling."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ) as mock_create_engine,
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            from databricks_ai_bridge.lakebase import DEFAULT_POOL_RECYCLE_SECONDS

            from databricks_openai.agents.session import AsyncDatabricksSession

            AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            call_kwargs = mock_create_engine.call_args[1]
            # Should use pool_recycle for connection recycling (default QueuePool)
            assert call_kwargs["pool_recycle"] == DEFAULT_POOL_RECYCLE_SECONDS
            # Should NOT use NullPool (uses default QueuePool instead)
            assert "poolclass" not in call_kwargs

    def test_init_sets_ssl_mode(self, mock_workspace_client, mock_engine, mock_event_listens_for):
        """Test that initialization sets SSL mode to require."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ) as mock_create_engine,
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            call_kwargs = mock_create_engine.call_args[1]
            assert call_kwargs["connect_args"]["sslmode"] == "require"

    def test_init_registers_do_connect_event(self, mock_workspace_client, mock_engine):
        """Test that initialization registers do_connect event for token injection."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ),
            patch("sqlalchemy.event.listens_for") as mock_listens_for,
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            # Verify event listener was registered on sync_engine for "do_connect"
            mock_listens_for.assert_called_once_with(mock_engine.sync_engine, "do_connect")

    def test_init_with_custom_table_names(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test initialization with custom table names."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ),
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
            patch(
                "agents.extensions.memory.SQLAlchemySession.__init__",
                return_value=None,
            ) as mock_parent_init,
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                sessions_table="custom_sessions",
                messages_table="custom_messages",
            )

            # Verify parent was called with custom table names
            call_kwargs = mock_parent_init.call_args[1]
            assert call_kwargs["sessions_table"] == "custom_sessions"
            assert call_kwargs["messages_table"] == "custom_messages"

    def test_init_with_create_tables_false(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test initialization with create_tables=False."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ),
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
            patch(
                "agents.extensions.memory.SQLAlchemySession.__init__",
                return_value=None,
            ) as mock_parent_init,
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                create_tables=False,
            )

            call_kwargs = mock_parent_init.call_args[1]
            assert call_kwargs["create_tables"] is False


class TestAsyncDatabricksSessionTokenInjection:
    """Tests for token injection via do_connect event."""

    def test_do_connect_injects_token(self, mock_workspace_client, mock_engine):
        """Test that do_connect event handler injects token into connection params."""
        captured_handler = None

        def capture_handler(engine, event_name):
            def decorator(fn):
                nonlocal captured_handler
                captured_handler = fn
                return fn

            return decorator

        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ),
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=capture_handler,
            ),
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            # Simulate do_connect event
            assert captured_handler is not None
            cparams = {}
            captured_handler(None, None, None, cparams)

            # Verify token was injected
            assert cparams["password"] == "test-oauth-token"


class TestAsyncDatabricksSessionEngineKwargs:
    """Tests for passing additional engine kwargs."""

    def test_extra_engine_kwargs_passed(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test that extra kwargs are passed to create_async_engine."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ) as mock_create_engine,
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                echo=True,  # Extra kwarg for SQLAlchemy
                pool_pre_ping=True,  # Another extra kwarg
            )

            call_kwargs = mock_create_engine.call_args[1]
            assert call_kwargs.get("echo") is True
            assert call_kwargs.get("pool_pre_ping") is True


class TestAsyncDatabricksSessionEngineCaching:
    """Tests for engine caching behavior."""

    def test_sessions_share_engine_for_same_instance(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test that multiple sessions with the same instance_name share an engine."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ) as mock_create_engine,
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            # Create two sessions with the same instance_name
            session1 = AsyncDatabricksSession(
                session_id="session-1",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )
            session2 = AsyncDatabricksSession(
                session_id="session-2",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            # Engine should only be created once
            assert mock_create_engine.call_count == 1

            # Both sessions should reference the same engine
            assert session1._engine is session2._engine

    def test_different_instances_get_different_engines(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test that sessions with different instance_names get different engines."""
        engine1 = MagicMock()
        engine1.sync_engine = MagicMock()
        engine2 = MagicMock()
        engine2.sync_engine = MagicMock()

        engines = [engine1, engine2]
        engine_iter = iter(engines)

        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                side_effect=lambda *args, **kwargs: next(engine_iter),
            ) as mock_create_engine,
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            # Create sessions with different instance_names
            session1 = AsyncDatabricksSession(
                session_id="session-1",
                instance_name="instance-a",
                workspace_client=mock_workspace_client,
            )
            session2 = AsyncDatabricksSession(
                session_id="session-2",
                instance_name="instance-b",
                workspace_client=mock_workspace_client,
            )

            # Engine should be created twice (once per instance)
            assert mock_create_engine.call_count == 2

            # Sessions should have different engines
            assert session1._engine is not session2._engine

    def test_different_engine_kwargs_get_different_engines(
        self, mock_workspace_client, mock_event_listens_for
    ):
        """Test that same instance_name with different engine_kwargs get different engines."""
        engine1 = MagicMock()
        engine1.sync_engine = MagicMock()
        engine2 = MagicMock()
        engine2.sync_engine = MagicMock()

        engines = [engine1, engine2]
        engine_iter = iter(engines)

        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                side_effect=lambda *args, **kwargs: next(engine_iter),
            ) as mock_create_engine,
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            session1 = AsyncDatabricksSession(
                session_id="session-1",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                echo=False,
            )
            session2 = AsyncDatabricksSession(
                session_id="session-2",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                echo=True,
            )

            # Engine should be created twice (different engine_kwargs)
            assert mock_create_engine.call_count == 2

            # Sessions should have different engines
            assert session1._engine is not session2._engine

    def test_use_cached_engine_false_creates_new_engine(
        self, mock_workspace_client, mock_event_listens_for
    ):
        """Test that use_cached_engine=False always creates a new engine."""
        engine1 = MagicMock()
        engine1.sync_engine = MagicMock()
        engine2 = MagicMock()
        engine2.sync_engine = MagicMock()

        engines = [engine1, engine2]
        engine_iter = iter(engines)

        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                side_effect=lambda *args, **kwargs: next(engine_iter),
            ) as mock_create_engine,
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            session1 = AsyncDatabricksSession(
                session_id="session-1",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                use_cached_engine=False,
            )
            session2 = AsyncDatabricksSession(
                session_id="session-2",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                use_cached_engine=False,
            )

            # Engine should be created twice despite same instance_name
            assert mock_create_engine.call_count == 2

            # Sessions should have different engines
            assert session1._engine is not session2._engine

    def test_same_instance_and_kwargs_share_engine(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test that same instance_name with same engine_kwargs reuse the cached engine."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ) as mock_create_engine,
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            from databricks_openai.agents.session import AsyncDatabricksSession

            session1 = AsyncDatabricksSession(
                session_id="session-1",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                echo=True,
            )
            session2 = AsyncDatabricksSession(
                session_id="session-2",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
                echo=True,
            )

            # Engine should only be created once
            assert mock_create_engine.call_count == 1

            # Both sessions should share the same engine
            assert session1._engine is session2._engine


class TestAsyncDatabricksSessionAsyncOnly:
    """Tests verifying AsyncDatabricksSession is async-only."""

    def test_get_items_returns_coroutine_without_await(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test that calling get_items() without await returns a coroutine."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ),
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            import inspect

            from databricks_openai.agents.session import AsyncDatabricksSession

            session = AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            # Calling without await returns a coroutine, not actual data
            result = session.get_items()
            assert inspect.iscoroutine(result)

            # Clean up the coroutine to avoid RuntimeWarning
            result.close()

    def test_add_items_returns_coroutine_without_await(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test that calling add_items() without await returns a coroutine."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ),
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            import inspect

            from databricks_openai.agents.session import AsyncDatabricksSession

            session = AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            # Calling without await returns a coroutine, not actual data
            result = session.add_items([{"role": "user", "content": "test"}])
            assert inspect.iscoroutine(result)

            # Clean up the coroutine to avoid RuntimeWarning
            result.close()

    def test_methods_are_coroutine_functions(
        self, mock_workspace_client, mock_engine, mock_event_listens_for
    ):
        """Test that all session methods are async (coroutine functions)."""
        with (
            patch(
                "databricks_ai_bridge.lakebase.WorkspaceClient",
                return_value=mock_workspace_client,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ),
            patch(
                "sqlalchemy.event.listens_for",
                side_effect=mock_event_listens_for,
            ),
        ):
            import inspect

            from databricks_openai.agents.session import AsyncDatabricksSession

            session = AsyncDatabricksSession(
                session_id="test-session-123",
                instance_name="test-instance",
                workspace_client=mock_workspace_client,
            )

            # All data methods should be async
            assert inspect.iscoroutinefunction(session.get_items)
            assert inspect.iscoroutinefunction(session.add_items)
            assert inspect.iscoroutinefunction(session.pop_item)
            assert inspect.iscoroutinefunction(session.clear_session)
