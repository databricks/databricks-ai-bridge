"""
MemorySession - SQLAlchemy-based session storage for Databricks Lakebase.

This module provides a MemorySession class that subclasses OpenAI's SQLAlchemySession
to provide persistent conversation history storage in Databricks Lakebase.

Usage::

    from databricks_openai.agents.session import MemorySession
    from agents import Agent, Runner

    session = MemorySession(
        session_id="user-123",
        instance_name="my-lakebase-instance",
    )

    agent = Agent(name="Assistant")
    result = await Runner.run(agent, "Hello!", session=session)
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
    from databricks_ai_bridge.lakebase import _LakebasePoolBase
    from sqlalchemy import URL, event
    from sqlalchemy.ext.asyncio import create_async_engine
except ImportError as e:
    raise ImportError(
        "MemorySession requires databricks-openai[memory]. "
        "Please install with: pip install databricks-openai[memory]"
    ) from e

logger = logging.getLogger(__name__)

# Constants for Lakebase connection
DEFAULT_TOKEN_CACHE_DURATION_SECONDS = 50 * 60  # 50 minutes
DEFAULT_POOL_RECYCLE_SECONDS = 45 * 60  # 45 minutes (before token cache expires)
DEFAULT_SSLMODE = "require"
DEFAULT_PORT = 5432
DEFAULT_DATABASE = "databricks_postgres"


class _LakebaseCredentials(_LakebasePoolBase):
    """
    Lightweight credential provider that reuses _LakebasePoolBase for:
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


class MemorySession(SQLAlchemySession):
    """
    OpenAI Agents SDK Session implementation for Databricks Lakebase.

    This class subclasses SQLAlchemySession to provide:
    - Lakebase instance resolution
    - OAuth token rotation for authentication
    - SQL logic inherited from SQLAlchemySession

    The session stores conversation history in two tables:
    - agent_sessions: Tracks session metadata (session_id, created_at, updated_at)
    - agent_messages: Stores conversation items (id, session_id, message_data, created_at)

    Example:
        ```python
        from databricks_openai.agents.session import MemorySession
        from agents import Agent, Runner


        async def run_agent(session_id: str, message: str):
            session = MemorySession(
                session_id=session_id,
                instance_name="my-lakebase-instance",
            )
            agent = Agent(name="Assistant")
            return await Runner.run(agent, message, session=session)
        ```

    For more information on the Session protocol, see:
    https://openai.github.io/openai-agents-python/ref/memory/session/
    """

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
        Initialize a MemorySession for Databricks Lakebase.

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
        # Create credential provider
        self._credentials = _LakebaseCredentials(
            instance_name=instance_name,
            workspace_client=workspace_client,
            token_cache_duration_seconds=token_cache_duration_seconds,
        )

        engine = self._create_engine(**engine_kwargs)

        # Initialize parent SQLAlchemySession - inherits all SQL logic
        super().__init__(
            session_id=session_id,
            engine=engine,
            create_tables=create_tables,
            sessions_table=sessions_table,
            messages_table=messages_table,
        )

        logger.info(
            "MemorySession initialized: instance=%s host=%s session_id=%s",
            instance_name,
            self._credentials.host,
            session_id,
        )

    def _create_engine(self, **engine_kwargs) -> "AsyncEngine":
        """Create an AsyncEngine with do_connect event for token injection."""
        # https://docs.sqlalchemy.org/en/21/core/engines.html#creating-urls-programmatically
        url = URL.create(
            drivername="postgresql+psycopg",
            username=self._credentials.username,
            host=self._credentials.host,
            port=DEFAULT_PORT,
            database=DEFAULT_DATABASE,
        )

        # Use default QueuePool with connection recycling.
        # Connections are recycled before token cache expires (50 min),
        # ensuring fresh tokens are injected via do_connect event.
        engine = create_async_engine(
            url,
            pool_recycle=DEFAULT_POOL_RECYCLE_SECONDS,
            connect_args={"sslmode": DEFAULT_SSLMODE},
            **engine_kwargs,
        )

        # Attach event to inject Lakebase token before each connection
        # Note: do_connect fires on sync_engine even for async operations
        credentials = self._credentials

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

    @property
    def connection_url(self) -> str:
        """The SQLAlchemy connection URL (without password, for debugging)."""
        url = URL.create(
            drivername="postgresql+psycopg",
            username=self._credentials.username,
            host=self._credentials.host,
            port=DEFAULT_PORT,
            database=DEFAULT_DATABASE,
            query={"sslmode": DEFAULT_SSLMODE},
        )
        return url.render_as_string(hide_password=True)
