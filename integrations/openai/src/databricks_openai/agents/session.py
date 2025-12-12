from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from threading import Lock
from typing import List, Optional, Dict, Any

from databricks.sdk import WorkspaceClient

try:
    from agents.memory.session import SessionABC
    from agents.items import TResponseInputItem
    from databricks_ai_bridge.lakebase import LakebasePool
except ImportError as e:
    raise ImportError(
        "LakebaseSession requires databricks-openai[memory]. "
        "Please install with: pip install databricks-openai[memory]"
    ) from e

logger = logging.getLogger(__name__)

# Module-level pool cache: instance_name:schema -> LakebasePool
_pool_cache: Dict[str, LakebasePool] = {}
_pool_cache_lock = Lock()


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


class LakebaseSession(SessionABC):
    """
        OpenAI Agents SDK Session implementation using Lakebase for persistent storage.

        This class follows the Session protocol for conversation memory,
        storing session data in two tables:
        - agent_sessions: Tracks session metadata (session_id, created_at, updated_at)
        - agent_messages: Stores conversation items (id, session_id, message_data, created_at)

        SessionABC: https://openai.github.io/openai-agents-python/ref/memory/session/#agents.memory.session.SessionABC

        Example (pool managed internally):
    ```python
            from databricks_openai.agents.session import LakebaseSession
            from agents import Agent, Runner

            async def run_agent(thread_id: str, message: str):
                session = LakebaseSession(
                    session_id=thread_id,
                    instance_name="my-lakebase-instance"
                )
                agent = Agent(name="Assistant")
                return await Runner.run(agent, message, session=session)
    ```

        Or pass in a pool directly:
    ```python
            pool = LakebasePool(instance_name="my-instance")
            session = LakebaseSession(session_id=thread_id, pool=pool)
    ```
    """

    # Table names
    SESSIONS_TABLE = "agent_sessions"
    MESSAGES_TABLE = "agent_messages"

    CREATE_SESSIONS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {sessions_table} (
        session_id TEXT PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """

    CREATE_MESSAGES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {messages_table} (
        id BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES {sessions_table}(session_id) ON DELETE CASCADE,
        message_data JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_{messages_table}_session_id 
        ON {messages_table}(session_id);
    CREATE INDEX IF NOT EXISTS idx_{messages_table}_session_order 
        ON {messages_table}(session_id, id);
    """

    session_id: str

    def __init__(
        self,
        session_id: str,
        *,
        # Option 1: Pass a pool directly
        pool: Optional[LakebasePool] = None,
        # Option 2: Pass instance name (pool managed internally)
        instance_name: Optional[str] = None,
        workspace_client: Optional[WorkspaceClient] = None,
        # Table configuration
        sessions_table: str = SESSIONS_TABLE,
        messages_table: str = MESSAGES_TABLE,
        create_tables: bool = True,
        **pool_kwargs,
    ) -> None:
        """
        Initialize a LakebaseSession.

        Args:
            session_id: Unique identifier for this conversation session.
            pool: Pre-existing LakebasePool to use. If provided, instance_name is ignored.
            instance_name: Name of the Lakebase instance. Required if pool is not provided.
            workspace_client: Optional WorkspaceClient for authentication.
            sessions_table: Name of the sessions table. Defaults to "agent_sessions".
            messages_table: Name of the messages table. Defaults to "agent_messages".
            create_tables: Whether to create tables on init. Defaults to True.
            **pool_kwargs: Additional arguments passed to LakebasePool if creating one.

        Raises:
            ValueError: If neither pool nor instance_name is provided.
        """
        if pool is None and instance_name is None:
            raise ValueError("Either 'pool' or 'instance_name' must be provided")

        self.session_id = session_id
        self.sessions_table = sessions_table
        self.messages_table = messages_table

        # Use provided pool or get/create from cache
        if pool is not None:
            self._pool = pool
        else:
            self._pool = _get_or_create_pool(
                instance_name=instance_name,
                workspace_client=workspace_client,
                **pool_kwargs,
            )

        if create_tables:
            self._ensure_tables()

        self._ensure_session()

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        sessions_sql = self.CREATE_SESSIONS_TABLE_SQL.format(sessions_table=self.sessions_table)
        messages_sql = self.CREATE_MESSAGES_TABLE_SQL.format(
            sessions_table=self.sessions_table, messages_table=self.messages_table
        )

        with self._pool.connection() as conn:
            conn.execute(sessions_sql)
            conn.execute(messages_sql)

        logger.debug(f"Ensured tables {self.sessions_table}, {self.messages_table} exist")

    def _ensure_session(self) -> None:
        """Ensure the session record exists in agent_sessions table."""
        now = datetime.now(timezone.utc)

        with self._pool.connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.sessions_table} (session_id, created_at, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) DO NOTHING
                """,
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
        if limit is not None:
            # Get the last N items, but return in chronological order
            query = f"""
                SELECT message_data FROM (
                    SELECT message_data, id
                    FROM {self.messages_table}
                    WHERE session_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                ) sub
                ORDER BY id ASC
            """
            params = (self.session_id, limit)
        else:
            # Get all items in chronological order
            query = f"""
                SELECT message_data 
                FROM {self.messages_table}
                WHERE session_id = %s
                ORDER BY id ASC
            """
            params = (self.session_id,)

        with self._pool.connection() as conn:
            result = conn.execute(query, params)
            rows = result.fetchall()

        return [row["message_data"] for row in rows]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """
        Add new items to the conversation history.

        Args:
            items: List of input items to add to the history.
        """
        if not items:
            return

        now = datetime.now(timezone.utc)

        with self._pool.connection() as conn:
            # Insert all messages
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {self.messages_table} (session_id, message_data, created_at)
                    VALUES (%s, %s, %s)
                    """,
                    [(self.session_id, json.dumps(item), now) for item in items],
                )

            # Update session timestamp
            conn.execute(
                f"""
                UPDATE {self.sessions_table} 
                SET updated_at = %s 
                WHERE session_id = %s
                """,
                (now, self.session_id),
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
                f"""
                DELETE FROM {self.messages_table}
                WHERE id = (
                    SELECT id 
                    FROM {self.messages_table}
                    WHERE session_id = %s
                    ORDER BY id DESC
                    LIMIT 1
                )
                RETURNING message_data
                """,
                (self.session_id,),
            )
            row = result.fetchone()

            if row:
                # Update session timestamp
                now = datetime.now(timezone.utc)
                conn.execute(
                    f"""
                    UPDATE {self.sessions_table} 
                    SET updated_at = %s 
                    WHERE session_id = %s
                    """,
                    (now, self.session_id),
                )

        if row:
            logger.debug(f"Popped item from session {self.session_id}")
            return row["message_data"]
        return None

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        with self._pool.connection() as conn:
            result = conn.execute(
                f"DELETE FROM {self.messages_table} WHERE session_id = %s", (self.session_id,)
            )
            count = result.rowcount

            # Update session timestamp
            now = datetime.now(timezone.utc)
            conn.execute(
                f"""
                UPDATE {self.sessions_table} 
                SET updated_at = %s 
                WHERE session_id = %s
                """,
                (now, self.session_id),
            )

        logger.info(f"Cleared {count} items from session {self.session_id}")
