from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional, Tuple, Union, cast
from uuid import UUID

from databricks.sdk import WorkspaceClient

try:
    from agents.items import TResponseInputItem
    from agents.memory.session import SessionABC
    from databricks_ai_bridge.lakebase import AsyncLakebasePool, LakebasePool
    from psycopg import sql
    from psycopg.sql import Composed
except ImportError as e:
    raise ImportError(
        "MemorySession requires databricks-openai[memory]. "
        "Please install with: pip install databricks-openai[memory]"
    ) from e

logger = logging.getLogger(__name__)

# Module-level pool cache: instance_name -> LakebasePool
_pool_cache: Dict[str, LakebasePool] = {}
_pool_cache_lock = Lock()

# Module-level async pool cache: instance_name -> AsyncLakebasePool
_async_pool_cache: Dict[str, AsyncLakebasePool] = {}
_async_pool_cache_lock = asyncio.Lock()


def _get_or_create_pool(
    instance_name: str,
    workspace_client: Optional[WorkspaceClient] = None,
    **pool_kwargs,
) -> LakebasePool:
    """Get cached pool or create new one for this instance."""
    cache_key = instance_name

    with _pool_cache_lock:
        if cache_key not in _pool_cache:
            logger.info(f"Creating new LakebasePool for {cache_key}")
            _pool_cache[cache_key] = LakebasePool(
                instance_name=instance_name,
                workspace_client=workspace_client,
                **pool_kwargs,
            )
        return _pool_cache[cache_key]


async def _get_or_create_async_pool(
    instance_name: str,
    workspace_client: Optional[WorkspaceClient] = None,
    **pool_kwargs,
) -> AsyncLakebasePool:
    """Get cached async pool or create new one for this instance."""
    cache_key = instance_name

    async with _async_pool_cache_lock:
        if cache_key not in _async_pool_cache:
            logger.info(f"Creating new AsyncLakebasePool for {cache_key}")
            pool = AsyncLakebasePool(
                instance_name=instance_name,
                workspace_client=workspace_client,
                **pool_kwargs,
            )
            await pool.open()
            _async_pool_cache[cache_key] = pool
        return _async_pool_cache[cache_key]


class _MemorySessionBase(SessionABC):
    """
    Base class with shared SQL, configuration, and helper methods for memory sessions.

    Subclasses implement sync or async pool initialization and database operations.
    """

    # Table names
    SESSIONS_TABLE = "agent_sessions"
    MESSAGES_TABLE = "agent_messages"

    CREATE_SESSIONS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {sessions_table} (
        session_id UUID PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """

    CREATE_MESSAGES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {messages_table} (
        id BIGSERIAL PRIMARY KEY,
        session_id UUID NOT NULL REFERENCES {sessions_table}(session_id) ON DELETE CASCADE,
        message_data JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS {idx_session_id} 
        ON {messages_table}(session_id);
    CREATE INDEX IF NOT EXISTS {idx_session_order} 
        ON {messages_table}(session_id, id);
    """

    session_id: UUID

    def __init__(
        self,
        session_id: UUID,
        *,
        sessions_table: str = SESSIONS_TABLE,
        messages_table: str = MESSAGES_TABLE,
    ) -> None:
        """
        Initialize base session attributes.

        Args:
            session_id: UUID identifier for this conversation session.
            sessions_table: Name of the sessions table. Defaults to "agent_sessions".
            messages_table: Name of the messages table. Defaults to "agent_messages".
        """
        self.session_id = session_id
        self.sessions_table = sessions_table
        self.messages_table = messages_table

    # --- SQL Building Helpers ---

    def _build_create_sessions_sql(self) -> Composed:
        """Build SQL to create the sessions table."""
        return sql.SQL(self.CREATE_SESSIONS_TABLE_SQL).format(
            sessions_table=sql.Identifier(self.sessions_table)
        )

    def _build_create_messages_sql(self) -> Composed:
        """Build SQL to create the messages table."""
        return sql.SQL(self.CREATE_MESSAGES_TABLE_SQL).format(
            sessions_table=sql.Identifier(self.sessions_table),
            messages_table=sql.Identifier(self.messages_table),
            idx_session_id=sql.Identifier(f"idx_{self.messages_table}_session_id"),
            idx_session_order=sql.Identifier(f"idx_{self.messages_table}_session_order"),
        )

    def _build_ensure_session_sql(self) -> Composed:
        """Build SQL to insert session record if not exists."""
        return sql.SQL(
            """
            INSERT INTO {} (session_id, created_at, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """
        ).format(sql.Identifier(self.sessions_table))

    def _build_get_items_query(
        self, limit: int | None
    ) -> Tuple[Composed, Tuple[UUID, ...] | Tuple[UUID, int]]:
        """Build SQL query and params to get items."""
        if limit is not None:
            query = sql.SQL(
                """
                SELECT message_data FROM (
                    SELECT message_data, id
                    FROM {}
                    WHERE session_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                ) sub
                ORDER BY id ASC
                """
            ).format(sql.Identifier(self.messages_table))
            params: Tuple[UUID, ...] | Tuple[UUID, int] = (self.session_id, limit)
        else:
            query = sql.SQL(
                """
                SELECT message_data 
                FROM {}
                WHERE session_id = %s
                ORDER BY id ASC
                """
            ).format(sql.Identifier(self.messages_table))
            params = (self.session_id,)
        return query, params

    def _build_add_items_sql(self) -> Composed:
        """Build SQL to insert message items."""
        return sql.SQL(
            """
            INSERT INTO {} (session_id, message_data)
            VALUES (%s, %s)
            """
        ).format(sql.Identifier(self.messages_table))

    def _build_update_session_timestamp_sql(self) -> Composed:
        """Build SQL to update session timestamp."""
        return sql.SQL(
            """
            UPDATE {} 
            SET updated_at = CURRENT_TIMESTAMP 
            WHERE session_id = %s
            """
        ).format(sql.Identifier(self.sessions_table))

    def _build_update_session_timestamp_with_value_sql(self) -> Composed:
        """Build SQL to update session timestamp with explicit value."""
        return sql.SQL(
            """
            UPDATE {} 
            SET updated_at = %s 
            WHERE session_id = %s
            """
        ).format(sql.Identifier(self.sessions_table))

    def _build_pop_item_sql(self) -> Composed:
        """Build SQL to delete and return most recent item."""
        messages_table_id = sql.Identifier(self.messages_table)
        return sql.SQL(
            """
            DELETE FROM {messages_table}
            WHERE id = (
                SELECT id 
                FROM {messages_table}
                WHERE session_id = %s
                ORDER BY id DESC
                LIMIT 1
            )
            RETURNING message_data
            """
        ).format(messages_table=messages_table_id)

    def _build_clear_session_sql(self) -> Composed:
        """Build SQL to delete all messages for session."""
        return sql.SQL("DELETE FROM {} WHERE session_id = %s").format(
            sql.Identifier(self.messages_table)
        )

    def _prepare_items_for_insert(self, items: list[TResponseInputItem]) -> list[Tuple[UUID, str]]:
        """Prepare items for database insertion."""
        return [(self.session_id, json.dumps(item)) for item in items]

    def _parse_message_data(self, message_data: Union[str, dict[str, Any]]) -> TResponseInputItem:
        """Parse message_data from database (may be JSON string or dict)."""
        if isinstance(message_data, str):
            return cast(TResponseInputItem, json.loads(message_data))
        return cast(TResponseInputItem, message_data)

    def _parse_rows_to_items(self, rows: list) -> list[TResponseInputItem]:
        """Parse database rows to list of items."""
        return [self._parse_message_data(row["message_data"]) for row in rows]


class MemorySession(_MemorySessionBase):
    """
    OpenAI Agents SDK Session implementation using Lakebase for persistent storage.

    This class follows the Session protocol for conversation memory,
    storing session data in two tables:
    - agent_sessions: Tracks session metadata (session_id, created_at, updated_at)
    - agent_messages: Stores conversation items (id, session_id, message_data, created_at)

    SessionABC: https://openai.github.io/openai-agents-python/ref/memory/session/#agents.memory.session.SessionABC

    Example:
    ```python
        from uuid import UUID
        from databricks_openai.agents.session import MemorySession
        from agents import Agent, Runner

        async def run_agent(thread_id: UUID | None, message: str):
            # Use uuid7 for time-ordered UUIDs (better for database indexing)
            session_id = thread_id
            session = MemorySession(
                session_id=session_id,
                instance_name="my-lakebase-instance"
            )
            agent = Agent(name="Assistant")
            return await Runner.run(agent, message, session=session)
    ```
    """

    def __init__(
        self,
        session_id: UUID,
        *,
        instance_name: str,
        workspace_client: Optional[WorkspaceClient] = None,
        sessions_table: str = _MemorySessionBase.SESSIONS_TABLE,
        messages_table: str = _MemorySessionBase.MESSAGES_TABLE,
        **pool_kwargs,
    ) -> None:
        """
        Initialize a MemorySession.

        On first initialization for a given Lakebase instance, this will automatically
        create the required tables if they don't exist.

        Args:
            session_id: UUID identifier for this conversation session.
            instance_name: Name of the Lakebase instance.
            workspace_client: Optional WorkspaceClient for authentication.
            sessions_table: Name of the sessions table. Defaults to "agent_sessions".
            messages_table: Name of the messages table. Defaults to "agent_messages".
            **pool_kwargs: Additional arguments passed to LakebasePool.
        """
        super().__init__(
            session_id=session_id,
            sessions_table=sessions_table,
            messages_table=messages_table,
        )

        self._pool = _get_or_create_pool(
            instance_name=instance_name,
            workspace_client=workspace_client,
            **pool_kwargs,
        )

        if not self._tables_exist():
            self._create_tables()

        self._ensure_session()

    def _tables_exist(self) -> bool:
        """Check if both session tables already exist."""
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM information_schema.tables 
                WHERE table_schema = current_schema()
                  AND table_name IN (%s, %s)
                """,
                (self.sessions_table, self.messages_table),
            )
            row = result.fetchone()
            return row["cnt"] == 2

    def _create_tables(self) -> None:
        """Create the required tables."""
        with self._pool.connection() as conn:
            conn.execute(self._build_create_sessions_sql())
            conn.execute(self._build_create_messages_sql())
        logger.info(f"Created tables {self.sessions_table}, {self.messages_table}")

    def _ensure_session(self) -> None:
        """Ensure the session record exists in agent_sessions table."""
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            conn.execute(
                self._build_ensure_session_sql(),
                (self.session_id, now, now),
            )
        logger.debug(f"Ensured session {self.session_id} exists")

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """
        Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items.
                   When specified, returns the latest N items in chronological order.

        Returns:
            List of input items representing the conversation history.
        """
        query, params = self._build_get_items_query(limit)
        with self._pool.connection() as conn:
            result = conn.execute(query, params)
            rows = result.fetchall()
        return self._parse_rows_to_items(rows)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """
        Add new items to the conversation history.

        Args:
            items: List of input items to add to the history.
        """
        if not items:
            return

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    self._build_add_items_sql(),
                    self._prepare_items_for_insert(items),
                )
            conn.execute(
                self._build_update_session_timestamp_sql(),
                (self.session_id,),
            )
        logger.debug(f"Added {len(items)} items to session {self.session_id}")

    async def pop_item(self) -> TResponseInputItem | None:
        """
        Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if the session is empty.
        """
        with self._pool.connection() as conn:
            result = conn.execute(
                self._build_pop_item_sql(),
                (self.session_id,),
            )
            row = result.fetchone()

            if row:
                now = datetime.now(timezone.utc)
                conn.execute(
                    self._build_update_session_timestamp_with_value_sql(),
                    (now, self.session_id),
                )

        if row:
            logger.debug(f"Popped item from session {self.session_id}")
            return self._parse_message_data(row["message_data"])
        return None

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        with self._pool.connection() as conn:
            result = conn.execute(
                self._build_clear_session_sql(),
                (self.session_id,),
            )
            count = result.rowcount

            now = datetime.now(timezone.utc)
            conn.execute(
                self._build_update_session_timestamp_with_value_sql(),
                (now, self.session_id),
            )
        logger.info(f"Cleared {count} items from session {self.session_id}")


class AsyncMemorySession(_MemorySessionBase):
    """
    OpenAI Agents SDK Session implementation using Lakebase for persistent storage (async version).

    This class follows the Session protocol for conversation memory,
    storing session data in two tables:
    - agent_sessions: Tracks session metadata (session_id, created_at, updated_at)
    - agent_messages: Stores conversation items (id, session_id, message_data, created_at)

    SessionABC: https://openai.github.io/openai-agents-python/ref/memory/session/#agents.memory.session.SessionABC

    Example:
    ```python
        from uuid import UUID
        from databricks_openai.agents.session import AsyncMemorySession
        from agents import Agent, Runner

        async def run_agent(thread_id: UUID | None, message: str):
            session_id = thread_id
            session = AsyncMemorySession(
                session_id=session_id,
                instance_name="my-lakebase-instance"
            )
            agent = Agent(name="Assistant")
            return await Runner.run(agent, message, session=session)
    ```
    """

    def __init__(
        self,
        session_id: UUID,
        *,
        instance_name: str,
        workspace_client: Optional[WorkspaceClient] = None,
        sessions_table: str = _MemorySessionBase.SESSIONS_TABLE,
        messages_table: str = _MemorySessionBase.MESSAGES_TABLE,
        **pool_kwargs,
    ) -> None:
        """
        Initialize an AsyncMemorySession.

        Note: The async pool and tables are initialized lazily on first use.

        Args:
            session_id: UUID identifier for this conversation session.
            instance_name: Name of the Lakebase instance.
            workspace_client: Optional WorkspaceClient for authentication.
            sessions_table: Name of the sessions table. Defaults to "agent_sessions".
            messages_table: Name of the messages table. Defaults to "agent_messages".
            **pool_kwargs: Additional arguments passed to AsyncLakebasePool.
        """
        super().__init__(
            session_id=session_id,
            sessions_table=sessions_table,
            messages_table=messages_table,
        )

        self._instance_name = instance_name
        self._workspace_client = workspace_client
        self._pool_kwargs = pool_kwargs

        self._pool: Optional[AsyncLakebasePool] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Ensure the pool is created and tables exist (lazy initialization)."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            self._pool = await _get_or_create_async_pool(
                instance_name=self._instance_name,
                workspace_client=self._workspace_client,
                **self._pool_kwargs,
            )

            if not await self._tables_exist():
                await self._create_tables()

            await self._ensure_session()
            self._initialized = True

    async def _tables_exist(self) -> bool:
        """Check if both session tables already exist."""
        assert self._pool is not None
        async with self._pool.connection() as conn:
            result = await conn.execute(
                """
                SELECT COUNT(*) as cnt FROM information_schema.tables 
                WHERE table_schema = current_schema()
                  AND table_name IN (%s, %s)
                """,
                (self.sessions_table, self.messages_table),
            )
            row = await result.fetchone()
            return row["cnt"] == 2

    async def _create_tables(self) -> None:
        """Create the required tables."""
        assert self._pool is not None
        async with self._pool.connection() as conn:
            await conn.execute(self._build_create_sessions_sql())
            await conn.execute(self._build_create_messages_sql())
        logger.info(f"Created tables {self.sessions_table}, {self.messages_table}")

    async def _ensure_session(self) -> None:
        """Ensure the session record exists in agent_sessions table."""
        assert self._pool is not None
        now = datetime.now(timezone.utc)
        async with self._pool.connection() as conn:
            await conn.execute(
                self._build_ensure_session_sql(),
                (self.session_id, now, now),
            )
        logger.debug(f"Ensured session {self.session_id} exists")

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """
        Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items.
                   When specified, returns the latest N items in chronological order.

        Returns:
            List of input items representing the conversation history.
        """
        await self._ensure_initialized()
        assert self._pool is not None

        query, params = self._build_get_items_query(limit)
        async with self._pool.connection() as conn:
            result = await conn.execute(query, params)
            rows = await result.fetchall()
        return self._parse_rows_to_items(rows)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """
        Add new items to the conversation history.

        Args:
            items: List of input items to add to the history.
        """
        if not items:
            return

        await self._ensure_initialized()
        assert self._pool is not None

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    self._build_add_items_sql(),
                    self._prepare_items_for_insert(items),
                )
            await conn.execute(
                self._build_update_session_timestamp_sql(),
                (self.session_id,),
            )
        logger.debug(f"Added {len(items)} items to session {self.session_id}")

    async def pop_item(self) -> TResponseInputItem | None:
        """
        Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if the session is empty.
        """
        await self._ensure_initialized()
        assert self._pool is not None

        async with self._pool.connection() as conn:
            result = await conn.execute(
                self._build_pop_item_sql(),
                (self.session_id,),
            )
            row = await result.fetchone()

            if row:
                now = datetime.now(timezone.utc)
                await conn.execute(
                    self._build_update_session_timestamp_with_value_sql(),
                    (now, self.session_id),
                )

        if row:
            logger.debug(f"Popped item from session {self.session_id}")
            return self._parse_message_data(row["message_data"])
        return None

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        await self._ensure_initialized()
        assert self._pool is not None

        async with self._pool.connection() as conn:
            result = await conn.execute(
                self._build_clear_session_sql(),
                (self.session_id,),
            )
            count = result.rowcount

            now = datetime.now(timezone.utc)
            await conn.execute(
                self._build_update_session_timestamp_with_value_sql(),
                (now, self.session_id),
            )
        logger.info(f"Cleared {count} items from session {self.session_id}")
