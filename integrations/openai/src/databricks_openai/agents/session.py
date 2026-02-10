"""
AsyncDatabricksSession - Async SQLAlchemy-based session storage for Databricks Lakebase.

This module provides an AsyncDatabricksSession class that subclasses OpenAI's SQLAlchemySession
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

import json
import logging
from threading import Lock
from typing import Any, Optional

try:
    from agents.extensions.memory import SQLAlchemySession
    from databricks.sdk import WorkspaceClient
    from databricks_ai_bridge.lakebase import (
        DEFAULT_POOL_RECYCLE_SECONDS,
        DEFAULT_TOKEN_CACHE_DURATION_SECONDS,
        AsyncLakebaseSQLAlchemy,
    )

    _session_imports_available = True
except ImportError:
    SQLAlchemySession = object  # type: ignore
    _session_imports_available = False

logger = logging.getLogger(__name__)


class AsyncDatabricksSession(SQLAlchemySession):
    """
    Async OpenAI Agents SDK Session implementation for Databricks Lakebase.
    For more information on the Session protocol, see:
    https://openai.github.io/openai-agents-python/ref/memory/session/

    Note:
        This class is **async-only**. All session methods (get_items, add_items,
        clear_session, etc.) are coroutines and must be awaited.

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
    """

    # Class-level cache for AsyncLakebaseSQLAlchemy instances keyed by
    # (instance_name, engine_kwargs).  This allows multiple sessions to share
    # a single engine/connection pool when the configuration is identical.
    _lakebase_sql_alchemy_cache: dict[str, AsyncLakebaseSQLAlchemy] = {}
    _lakebase_sql_alchemy_cache_lock = Lock()

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
        use_cached_engine: bool = True,
        **engine_kwargs,
    ) -> None:
        """
        Initialize an AsyncDatabricksSession for Databricks Lakebase.

        Args:
            session_id: Unique identifier for the conversation session.
            instance_name: Name of the Lakebase instance.
            workspace_client: Optional WorkspaceClient for authentication.
                If not provided, a default client will be created.
            token_cache_duration_seconds: How long to cache OAuth tokens.
                Defaults to 15 minutes.
            create_tables: Whether to auto-create tables on first use.
                Defaults to True.
            sessions_table: Name of the sessions table.
                Defaults to "agent_sessions".
            messages_table: Name of the messages table.
                Defaults to "agent_messages".
            use_cached_engine: Whether to reuse a cached engine for the same
                instance_name and engine_kwargs combination. Set to False to
                always create a new engine. Defaults to True.
            **engine_kwargs: Additional keyword arguments passed to
                SQLAlchemy's create_async_engine().
        """
        if not _session_imports_available:
            raise ImportError(
                "AsyncDatabricksSession requires databricks-openai[memory]. "
                "Please install with: pip install databricks-openai[memory]"
            )

        self._lakebase = self._get_or_create_lakebase(
            instance_name=instance_name,
            workspace_client=workspace_client,
            token_cache_duration_seconds=token_cache_duration_seconds,
            pool_recycle=engine_kwargs.pop("pool_recycle", DEFAULT_POOL_RECYCLE_SECONDS),
            use_cached_engine=use_cached_engine,
            **engine_kwargs,
        )

        # Initialize parent SQLAlchemySession - inherits all SQL logic
        super().__init__(
            session_id=session_id,
            engine=self._lakebase.engine,
            create_tables=create_tables,
            sessions_table=sessions_table,
            messages_table=messages_table,
        )

        logger.info(
            "AsyncDatabricksSession initialized: instance=%s session_id=%s",
            instance_name,
            session_id,
        )

    @classmethod
    def _build_cache_key(cls, instance_name: str, **engine_kwargs: Any) -> str:
        """Build a cache key from instance_name and engine_kwargs."""
        # Sort kwargs for deterministic key; use JSON for serializable values
        kwargs_key = json.dumps(engine_kwargs, sort_keys=True, default=str)
        return f"{instance_name}::{kwargs_key}"

    @classmethod
    def _get_or_create_lakebase(
        cls,
        *,
        instance_name: str,
        workspace_client: Optional[WorkspaceClient],
        token_cache_duration_seconds: int,
        pool_recycle: int,
        use_cached_engine: bool = True,
        **engine_kwargs,
    ) -> AsyncLakebaseSQLAlchemy:
        """Get cached AsyncLakebaseSQLAlchemy or create a new one.
        The cache key uses both instance_name and engine_kwargs
        """
        cache_key = cls._build_cache_key(instance_name, pool_recycle=pool_recycle, **engine_kwargs)

        if use_cached_engine:
            with cls._lakebase_sql_alchemy_cache_lock:
                if cache_key in cls._lakebase_sql_alchemy_cache:
                    logger.debug("Reusing cached engine for key=%s", cache_key)
                    return cls._lakebase_sql_alchemy_cache[cache_key]

        lakebase = AsyncLakebaseSQLAlchemy(
            instance_name=instance_name,
            workspace_client=workspace_client,
            token_cache_duration_seconds=token_cache_duration_seconds,
            pool_recycle=pool_recycle,
            **engine_kwargs,
        )

        if use_cached_engine:
            with cls._lakebase_sql_alchemy_cache_lock:
                cls._lakebase_sql_alchemy_cache[cache_key] = lakebase

        return lakebase
