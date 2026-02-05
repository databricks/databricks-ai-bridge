"""
AsyncDatabricksSession - Async SQLAlchemy-based session storage for Databricks Lakebase.

This module provides a AsyncDatabricksSession class that subclasses OpenAI's SQLAlchemySession
to provide persistent conversation history storage in Databricks Lakebase.

Note:
    This class is **async-only** as it follows the Session Protocol. Use within async context
    https://openai.github.io/openai-agents-python/ref/memory/session/#agents.memory.session.Session

Usage::

    import asyncio
    from databricks_openai.agents import AsyncDatabricksSession
    from agents import Agent, Runner


    async def main():
        session = AsyncDatabricksSession(
            session_id="user-123",
            instance_name="my-lakebase-instance",
        )

        agent = Agent(name="Assistant")
        result = await Runner.run(agent, "Hello!", session=session)


    asyncio.run(main())
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

try:
    from agents.extensions.memory import SQLAlchemySession
    from databricks.sdk import WorkspaceClient
    from databricks_ai_bridge.lakebase import _LakebaseBase
    from sqlalchemy import URL, event
    from sqlalchemy.ext.asyncio import create_async_engine
except ImportError as e:
    raise ImportError(
        "AsyncDatabricksSession requires databricks-openai[memory]. "
        "Please install with: pip install databricks-openai[memory]"
    ) from e

logger = logging.getLogger(__name__)

# Constants for Lakebase connection
DEFAULT_TOKEN_CACHE_DURATION_SECONDS = 50 * 60  # 50 minutes
DEFAULT_POOL_RECYCLE_SECONDS = 45 * 60  # 45 minutes (before token cache expires)
DEFAULT_SSLMODE = "require"
DEFAULT_PORT = 5432
DEFAULT_DATABASE = "databricks_postgres"


class _LakebaseCredentials(_LakebaseBase):
    """
    Lightweight credential provider that reuses _LakebaseBase for:
    - Instance name → host resolution
    - Username inference from workspace client
    - Token minting and caching

    Does NOT create a connection pool - just provides credentials for SQLAlchemy.
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: Optional[WorkspaceClient] = None,
        token_cache_duration_seconds: int = DEFAULT_TOKEN_CACHE_DURATION_SECONDS,
    ) -> None:
        super().__init__(
            instance_name=instance_name,
            workspace_client=workspace_client,
            token_cache_duration_seconds=token_cache_duration_seconds,
        )
        self._cache_lock = Lock()

    def get_token(self) -> str:
        """Get cached token or mint a new one (thread-safe)."""
        with self._cache_lock:
            if cached := self._get_cached_token():
                return cached
            token = self._mint_token()
            self._cached_token = token
            self._cache_ts = time.time()
            return token


class AsyncDatabricksSession(SQLAlchemySession):
    """
    Async OpenAI Agents SDK Session implementation for Databricks Lakebase.

    This class subclasses SQLAlchemySession to provide:
    - Lakebase instance resolution
    - OAuth token rotation for authentication
    - Connection pooling with automatic token refresh
    - SQL logic inherited from SQLAlchemySession

    Note:
        This class is **async-only**. All session methods (get_items, add_items,
        clear_session, etc.) are coroutines and must be awaited.

    Note:
        Engines are cached and reused across sessions with the same instance_name.
        This means multiple AsyncDatabricksSession instances share a single connection pool,
        rather than creating a new pool per session

    The session stores conversation history in two tables:
    - agent_sessions: Tracks session metadata (session_id, created_at, updated_at)
    - agent_messages: Stores conversation items (id, session_id, message_data, created_at)

    Example:
        ```python
        import asyncio
        from databricks_openai.agents import AsyncDatabricksSession
        from agents import Agent, Runner


        async def main():
            session = AsyncDatabricksSession(
                session_id="user-123",
                instance_name="my-lakebase-instance",
            )
            agent = Agent(name="Assistant")
            result = await Runner.run(agent, "Hello!", session=session)


        asyncio.run(main())
        ```

    For more information on the Session protocol, see:
    https://openai.github.io/openai-agents-python/ref/memory/session/
    """

    # Class-level cache for engines and credentials, keyed by instance_name.
    # This allows multiple AsyncDatabricksSession instances to share a single engine/pool.
    _engine_cache: "dict[str, tuple[AsyncEngine, _LakebaseCredentials]]" = {}
    _engine_cache_lock = Lock()

    def __init__(
        self,
        session_id: str,
        *,
        instance_name: str,
        workspace_client: Optional[WorkspaceClient] = None,
        token_cache_duration_seconds: int = DEFAULT_TOKEN_CACHE_DURATION_SECONDS,
        create_tables: bool = True,
        sessions_table: str = "agent_sessions",
        messages_table: str = "agent_messages",
        **engine_kwargs,
    ) -> None:
        """
        Initialize a AsyncDatabricksSession for Databricks Lakebase.

        Args:
            session_id: Unique identifier for the conversation session.
            instance_name: Name of the Lakebase instance.
            workspace_client: Optional WorkspaceClient for authentication.
                If not provided, a default client will be created.
            token_cache_duration_seconds: How long to cache OAuth tokens.
                Defaults to 50 minutes.
            create_tables: Whether to auto-create tables on first use.
                Defaults to True.
            sessions_table: Name of the sessions table.
                Defaults to "agent_sessions".
            messages_table: Name of the messages table.
                Defaults to "agent_messages".
            **engine_kwargs: Additional keyword arguments passed to
                SQLAlchemy's create_async_engine().
        """
        engine, credentials = self._get_or_create_engine(
            instance_name=instance_name,
            workspace_client=workspace_client,
            token_cache_duration_seconds=token_cache_duration_seconds,
            **engine_kwargs,
        )
        self._credentials = credentials

        # Initialize parent SQLAlchemySession - inherits all SQL logic
        super().__init__(
            session_id=session_id,
            engine=engine,
            create_tables=create_tables,
            sessions_table=sessions_table,
            messages_table=messages_table,
        )

        logger.info(
            "AsyncDatabricksSession initialized: instance=%s host=%s session_id=%s",
            instance_name,
            self._credentials.host,
            session_id,
        )

    @classmethod
    def _get_or_create_engine(
        cls,
        *,
        instance_name: str,
        workspace_client: Optional[WorkspaceClient],
        token_cache_duration_seconds: int,
        **engine_kwargs,
    ) -> "tuple[AsyncEngine, _LakebaseCredentials]":
        """Get cached engine or create a new one (thread-safe).

        Engines are cached by instance_name so multiple sessions can share
        the same connection pool.
        """
        with cls._engine_cache_lock:
            if instance_name in cls._engine_cache:
                logger.debug("Reusing cached engine for instance=%s", instance_name)
                return cls._engine_cache[instance_name]

            credentials = _LakebaseCredentials(
                instance_name=instance_name,
                workspace_client=workspace_client,
                token_cache_duration_seconds=token_cache_duration_seconds,
            )

            engine = cls._create_engine(credentials, **engine_kwargs)
            cls._engine_cache[instance_name] = (engine, credentials)
            logger.info(
                "Created new engine for instance=%s host=%s",
                instance_name,
                credentials.host,
            )

            return engine, credentials

    @classmethod
    def clear_engine_cache(cls, instance_name: Optional[str] = None) -> None:
        """Clear cached engines.

        Args:
            instance_name: If provided, only clear the engine for this instance.
                If None, clear all cached engines.

        Note:
            This does not close the engines. Use this when you need to force
            creation of a new engine with different settings.
        """
        with cls._engine_cache_lock:
            if instance_name is not None:
                cls._engine_cache.pop(instance_name, None)
                logger.info("Cleared engine cache for instance=%s", instance_name)
            else:
                cls._engine_cache.clear()
                logger.info("Cleared all engine caches")

    @staticmethod
    def _create_url(credentials: _LakebaseCredentials):
        """Create a SQLAlchemy URL for Lakebase connection.

        https://docs.sqlalchemy.org/en/21/core/engines.html#creating-urls-programmatically
        """
        return URL.create(
            drivername="postgresql+psycopg",
            username=credentials.username,
            host=credentials.host,
            port=DEFAULT_PORT,
            database=DEFAULT_DATABASE,
        )

    @staticmethod
    def _create_engine(
        credentials: _LakebaseCredentials, **engine_kwargs
    ) -> "AsyncEngine":
        """Create an AsyncEngine with do_connect event for token injection."""
        url = AsyncDatabricksSession._create_url(credentials)

        engine = create_async_engine(
            url,
            pool_recycle=DEFAULT_POOL_RECYCLE_SECONDS,
            connect_args={"sslmode": DEFAULT_SSLMODE},
            **engine_kwargs,
        )

        # AsyncEngine wraps a sync Engine internally - connection events like
        # do_connect must be registered on sync_engine, not the async wrapper.
        # https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#using-events-with-the-asyncio-extension
        @event.listens_for(engine.sync_engine, "do_connect")
        def inject_lakebase_token(dialect, conn_rec, cargs, cparams):
            cparams["password"] = credentials.get_token()
            logger.debug("Injected Lakebase token for connection")

        return engine

    @property
    def instance_name(self) -> str:
        """The Lakebase instance name."""
        return self._credentials.instance_name

    @property
    def host(self) -> str:
        """The resolved Lakebase host."""
        return self._credentials.host

    @property
    def username(self) -> str:
        """The database username."""
        return self._credentials.username


