from __future__ import annotations

import json
from unittest.mock import MagicMock
from uuid import UUID

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
pytest.importorskip("agents.memory.session")

from databricks_ai_bridge import lakebase
from psycopg import sql

from databricks_openai.agents.session import (
    AsyncMemorySession,
    MemorySession,
    _async_pool_cache,
    _pool_cache,
)

# Use UUID (V7) for performance
TEST_SESSION_ID = UUID("12345678-1234-5678-1234-567812345678")
TEST_SESSION_ID_2 = UUID("22345678-1234-5678-1234-567812345678")
TEST_SESSION_ID_3 = UUID("32345678-1234-5678-1234-567812345678")


def query_to_string(query):
    """Convert a query (string or sql.Composed) to a string for testing."""
    if isinstance(query, str):
        return query
    if isinstance(query, (sql.Composed, sql.SQL, sql.Identifier)):
        return query.as_string(None)
    return str(query)


class MockCursor:
    """Mock cursor for executemany operations."""

    def __init__(self):
        self.executed_queries = []

    def executemany(self, query, params):
        self.executed_queries.append((query, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class MockAsyncCursor:
    """Mock async cursor for executemany operations."""

    def __init__(self):
        self.executed_queries = []

    async def executemany(self, query, params):
        self.executed_queries.append((query, params))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class MockResult:
    """Mock result object for database queries."""

    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount
        self._index = 0

    def fetchall(self):
        return self._rows

    def fetchone(self):
        if self._index < len(self._rows):
            row = self._rows[self._index]
            self._index += 1
            return row
        return None


class MockAsyncResult:
    """Mock async result object for database queries."""

    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount
        self._index = 0

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        if self._index < len(self._rows):
            row = self._rows[self._index]
            self._index += 1
            return row
        return None


class MockConnection:
    """Mock database connection."""

    def __init__(self):
        self.executed_queries = []
        self._cursor = MockCursor()
        self._next_result = MockResult()
        self._results_queue = []

    def execute(self, query, params=None):
        self.executed_queries.append((query, params))
        if self._results_queue:
            return self._results_queue.pop(0)
        return self._next_result

    def cursor(self):
        return self._cursor

    def set_next_result(self, result):
        self._next_result = result

    def queue_result(self, result):
        self._results_queue.append(result)


class MockAsyncConnection:
    """Mock async database connection."""

    def __init__(self):
        self.executed_queries = []
        self._cursor = MockAsyncCursor()
        self._next_result = MockAsyncResult()
        self._results_queue = []

    async def execute(self, query, params=None):
        self.executed_queries.append((query, params))
        if self._results_queue:
            return self._results_queue.pop(0)
        return self._next_result

    def cursor(self):
        return self._cursor

    def set_next_result(self, result):
        self._next_result = result

    def queue_result(self, result):
        self._results_queue.append(result)


class MockConnectionPool:
    """Mock connection pool for testing."""

    def __init__(self, connection_value=None):
        self.connection_value = connection_value or MockConnection()
        self.conninfo = ""

    def __call__(self, *, conninfo, connection_class=None, **kwargs):
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


class MockAsyncConnectionPool:
    """Mock async connection pool for testing."""

    def __init__(self, connection_value=None):
        self.connection_value = connection_value or MockAsyncConnection()
        self.conninfo = ""
        self._opened = False
        self._closed = False

    def __call__(self, *, conninfo, connection_class=None, **kwargs):
        self.conninfo = conninfo
        return self

    async def open(self):
        self._opened = True

    async def close(self):
        self._closed = True

    def connection(self):
        class _AsyncCtx:
            def __init__(self, outer):
                self.outer = outer

            async def __aenter__(self):
                return self.outer.connection_value

            async def __aexit__(self, exc_type, exc, tb):
                pass

        return _AsyncCtx(self)


@pytest.fixture(autouse=True)
def clear_pool_cache():
    """Clear the pool cache before each test."""
    _pool_cache.clear()
    _async_pool_cache.clear()
    yield
    _pool_cache.clear()
    _async_pool_cache.clear()


@pytest.fixture
def mock_workspace():
    """Create a mock workspace client."""
    workspace = MagicMock()
    workspace.database.generate_database_credential.return_value = MagicMock(token="stub-token")
    workspace.database.get_database_instance.return_value.read_write_dns = "db-host"
    workspace.current_service_principal.me.side_effect = RuntimeError("no sp")
    workspace.current_user.me.return_value = MagicMock(user_name="test@databricks.com")
    return workspace


@pytest.fixture
def mock_connection():
    """Create a mock connection."""
    return MockConnection()


@pytest.fixture
def mock_async_connection():
    """Create a mock async connection."""
    return MockAsyncConnection()


@pytest.fixture
def mock_pool(mock_connection):
    """Create a mock connection pool."""
    return MockConnectionPool(connection_value=mock_connection)


@pytest.fixture
def mock_async_pool(mock_async_connection):
    """Create a mock async connection pool."""
    return MockAsyncConnectionPool(connection_value=mock_async_connection)


# =============================================================================
# MemorySession Tests (Sync)
# =============================================================================


def test_session_configures_lakebase(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test that MemorySession correctly configures the Lakebase pool."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Mock: tables already exist
    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))
    mock_connection.queue_result(MockResult())  # INSERT session

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-lakebase-instance",
        workspace_client=mock_workspace,
    )

    assert (
        mock_pool.conninfo
        == "dbname=databricks_postgres user=test@databricks.com host=db-host port=5432 sslmode=require"
    )
    assert session.session_id == TEST_SESSION_ID
    assert session.sessions_table == "agent_sessions"
    assert session.messages_table == "agent_messages"


def test_session_creates_tables_on_init_when_not_exist(
    monkeypatch, mock_workspace, mock_pool, mock_connection
):
    """Test that MemorySession creates tables when they don't exist."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Mock: tables don't exist (count=0)
    mock_connection.queue_result(MockResult(rows=[{"cnt": 0}]))
    # CREATE sessions table
    mock_connection.queue_result(MockResult())
    # CREATE messages table
    mock_connection.queue_result(MockResult())
    # INSERT session
    mock_connection.queue_result(MockResult())

    MemorySession(
        session_id=TEST_SESSION_ID_2,
        instance_name="test-lakebase-instance",
        workspace_client=mock_workspace,
    )

    # Should have executed CREATE TABLE statements
    queries = [query_to_string(q) for q, _ in mock_connection.executed_queries]
    create_sessions_found = any(
        "CREATE TABLE IF NOT EXISTS" in q and "agent_sessions" in q for q in queries
    )
    create_messages_found = any(
        "CREATE TABLE IF NOT EXISTS" in q and "agent_messages" in q for q in queries
    )

    assert create_sessions_found, "Should create sessions table"
    assert create_messages_found, "Should create messages table"


def test_session_skips_table_creation_when_tables_exist(
    monkeypatch, mock_workspace, mock_pool, mock_connection
):
    """Test that MemorySession skips table creation when tables already exist."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Mock: both tables exist (count=2)
    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))
    # INSERT session (no CREATE TABLE calls)
    mock_connection.queue_result(MockResult())

    MemorySession(
        session_id=TEST_SESSION_ID_3,
        instance_name="test-lakebase-instance",
        workspace_client=mock_workspace,
    )

    # Should NOT have executed CREATE TABLE statements
    queries = [query_to_string(q) for q, _ in mock_connection.executed_queries]
    create_table_found = any("CREATE TABLE" in q for q in queries)

    assert not create_table_found, "Should not create tables when they already exist"


def test_session_ensures_session_record(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test that MemorySession ensures the session record exists."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Mock: tables already exist
    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))
    mock_connection.queue_result(MockResult())  # INSERT session

    MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-lakebase-instance",
        workspace_client=mock_workspace,
    )

    # Find the INSERT INTO agent_sessions query
    insert_queries = [
        (q, p)
        for q, p in mock_connection.executed_queries
        if "INSERT INTO" in query_to_string(q) and "agent_sessions" in query_to_string(q)
    ]

    assert len(insert_queries) > 0, "Should insert session record"
    query, params = insert_queries[0]
    assert params[0] == TEST_SESSION_ID, "Should use correct session_id"


@pytest.mark.asyncio
async def test_get_items_empty_session(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test get_items returns empty list for new session."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Mock: tables exist, then INSERT session, then SELECT messages
    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_connection.queue_result(MockResult())  # INSERT session
    mock_connection.queue_result(MockResult(rows=[]))  # SELECT messages

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    items = await session.get_items()
    assert items == []


@pytest.mark.asyncio
async def test_get_items_returns_parsed_json(
    monkeypatch, mock_workspace, mock_pool, mock_connection
):
    """Test get_items correctly parses JSON data."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    test_messages = [
        {"message_data": json.dumps({"role": "user", "content": "Hello"})},
        {"message_data": json.dumps({"role": "assistant", "content": "Hi there!"})},
    ]

    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_connection.queue_result(MockResult())  # INSERT session
    mock_connection.queue_result(MockResult(rows=test_messages))  # SELECT messages

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    items = await session.get_items()

    assert len(items) == 2
    assert items[0]["role"] == "user"
    assert items[0]["content"] == "Hello"
    assert items[1]["role"] == "assistant"
    assert items[1]["content"] == "Hi there!"


@pytest.mark.asyncio
async def test_get_items_with_limit(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test get_items respects limit parameter."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_connection.queue_result(MockResult())  # INSERT session
    mock_connection.queue_result(
        MockResult(rows=[{"message_data": json.dumps({"role": "user", "content": "Latest"})}])
    )

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    items = await session.get_items(limit=1)

    assert len(items) == 1

    # Verify the query used LIMIT
    select_queries = [
        query_to_string(q)
        for q, p in mock_connection.executed_queries
        if "SELECT message_data" in query_to_string(q)
    ]
    assert any("LIMIT" in q for q in select_queries), "Should use LIMIT in query"


@pytest.mark.asyncio
async def test_add_items(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test add_items inserts messages correctly."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Mock: tables exist
    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))
    mock_connection.queue_result(MockResult())  # INSERT session

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    test_items = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    await session.add_items(test_items)

    # Check that executemany was called on cursor
    assert len(mock_connection._cursor.executed_queries) > 0
    query, params = mock_connection._cursor.executed_queries[-1]
    query_str = query_to_string(query)
    assert "INSERT INTO" in query_str and "agent_messages" in query_str
    assert len(params) == 2  # Two items


@pytest.mark.asyncio
async def test_add_items_empty_list(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test add_items handles empty list gracefully."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Mock: tables exist
    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))
    mock_connection.queue_result(MockResult())  # INSERT session

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    initial_query_count = len(mock_connection.executed_queries)

    await session.add_items([])

    # Should not execute any additional queries for empty list
    # (only the queries from init should be present)
    assert len(mock_connection.executed_queries) == initial_query_count


@pytest.mark.asyncio
async def test_pop_item_returns_last_item(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test pop_item removes and returns the most recent item."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_connection.queue_result(MockResult())  # INSERT session
    # DELETE RETURNING result
    mock_connection.queue_result(
        MockResult(
            rows=[{"message_data": json.dumps({"role": "assistant", "content": "Last msg"})}]
        )
    )
    mock_connection.queue_result(MockResult())  # UPDATE session timestamp

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    popped = await session.pop_item()

    assert popped is not None
    assert popped["role"] == "assistant"
    assert popped["content"] == "Last msg"


@pytest.mark.asyncio
async def test_pop_item_returns_none_when_empty(
    monkeypatch, mock_workspace, mock_pool, mock_connection
):
    """Test pop_item returns None for empty session."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_connection.queue_result(MockResult())  # INSERT session
    mock_connection.queue_result(MockResult(rows=[]))  # DELETE RETURNING - empty

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    popped = await session.pop_item()

    assert popped is None


@pytest.mark.asyncio
async def test_clear_session(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test clear_session deletes all messages for the session."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_connection.queue_result(MockResult())  # INSERT session
    mock_connection.queue_result(MockResult(rowcount=5))  # DELETE messages
    mock_connection.queue_result(MockResult())  # UPDATE session timestamp

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    await session.clear_session()

    # Find the DELETE query
    delete_queries = [
        (q, p)
        for q, p in mock_connection.executed_queries
        if "DELETE FROM" in query_to_string(q)
        and "agent_messages" in query_to_string(q)
        and "WHERE session_id" in query_to_string(q)
    ]

    assert len(delete_queries) > 0, "Should execute DELETE query"
    query, params = delete_queries[0]
    assert params == (TEST_SESSION_ID,), "Should use correct session_id"


def test_custom_table_names(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test that custom table names are used correctly."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Mock: tables don't exist (custom names), so they will be created
    mock_connection.queue_result(MockResult(rows=[{"cnt": 0}]))  # tables don't exist
    mock_connection.queue_result(MockResult())  # CREATE sessions
    mock_connection.queue_result(MockResult())  # CREATE messages
    mock_connection.queue_result(MockResult())  # INSERT session

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
        sessions_table="custom_sessions",
        messages_table="custom_messages",
    )

    assert session.sessions_table == "custom_sessions"
    assert session.messages_table == "custom_messages"

    # Check that CREATE TABLE uses custom names
    queries = [query_to_string(q) for q, _ in mock_connection.executed_queries]
    assert any("custom_sessions" in q for q in queries)
    assert any("custom_messages" in q for q in queries)


def test_pool_caching(monkeypatch, mock_workspace, mock_pool, mock_connection):
    """Test that pools are cached and reused."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Mock for both session creations
    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))  # tables exist
    mock_connection.queue_result(MockResult())  # INSERT session 1
    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))  # tables exist
    mock_connection.queue_result(MockResult())  # INSERT session 2

    session1 = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="shared-instance",
        workspace_client=mock_workspace,
    )

    session2 = MemorySession(
        session_id=TEST_SESSION_ID_2,
        instance_name="shared-instance",
        workspace_client=mock_workspace,
    )

    # Both sessions should share the same pool
    assert session1._pool is session2._pool


@pytest.mark.asyncio
async def test_get_items_handles_dict_message_data(
    monkeypatch, mock_workspace, mock_pool, mock_connection
):
    """Test get_items handles message_data that's already a dict (not JSON string)."""
    monkeypatch.setattr(lakebase, "ConnectionPool", mock_pool)

    # Some database drivers return JSONB as dict directly
    test_messages = [
        {"message_data": {"role": "user", "content": "Already parsed"}},
    ]

    mock_connection.queue_result(MockResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_connection.queue_result(MockResult())  # INSERT session
    mock_connection.queue_result(MockResult(rows=test_messages))

    session = MemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    items = await session.get_items()

    assert len(items) == 1
    assert items[0]["role"] == "user"
    assert items[0]["content"] == "Already parsed"


# =============================================================================
# AsyncMemorySession Tests
# =============================================================================


@pytest.mark.asyncio
async def test_async_session_lazy_initialization(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test that AsyncMemorySession initializes lazily on first use."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    # Create session - should NOT trigger any DB operations yet
    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    # No queries should have been executed yet
    assert len(mock_async_connection.executed_queries) == 0
    assert not session._initialized

    # Mock: tables exist, then INSERT session, then SELECT messages
    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    mock_async_connection.queue_result(MockAsyncResult(rows=[]))  # SELECT messages

    # First use triggers initialization
    await session.get_items()

    # Now it should be initialized
    assert session._initialized
    assert mock_async_pool._opened


@pytest.mark.asyncio
async def test_async_session_configures_lakebase(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test that AsyncMemorySession correctly configures the Lakebase pool."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-lakebase-instance",
        workspace_client=mock_workspace,
    )

    # Mock: tables already exist
    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    mock_async_connection.queue_result(MockAsyncResult(rows=[]))  # SELECT for get_items

    # Trigger initialization
    await session.get_items()

    assert (
        mock_async_pool.conninfo
        == "dbname=databricks_postgres user=test@databricks.com host=db-host port=5432 sslmode=require"
    )
    assert session.session_id == TEST_SESSION_ID
    assert session.sessions_table == "agent_sessions"
    assert session.messages_table == "agent_messages"


@pytest.mark.asyncio
async def test_async_session_creates_tables_when_not_exist(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test that AsyncMemorySession creates tables when they don't exist."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    # Mock: tables don't exist (count=0)
    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 0}]))
    mock_async_connection.queue_result(MockAsyncResult())  # CREATE sessions table
    mock_async_connection.queue_result(MockAsyncResult())  # CREATE messages table
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    mock_async_connection.queue_result(MockAsyncResult(rows=[]))  # SELECT for get_items

    await session.get_items()

    # Should have executed CREATE TABLE statements
    queries = [query_to_string(q) for q, _ in mock_async_connection.executed_queries]
    create_sessions_found = any(
        "CREATE TABLE IF NOT EXISTS" in q and "agent_sessions" in q for q in queries
    )
    create_messages_found = any(
        "CREATE TABLE IF NOT EXISTS" in q and "agent_messages" in q for q in queries
    )

    assert create_sessions_found, "Should create sessions table"
    assert create_messages_found, "Should create messages table"


@pytest.mark.asyncio
async def test_async_session_skips_table_creation_when_tables_exist(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test that AsyncMemorySession skips table creation when tables already exist."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    # Mock: both tables exist (count=2)
    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    mock_async_connection.queue_result(MockAsyncResult(rows=[]))  # SELECT for get_items

    await session.get_items()

    # Should NOT have executed CREATE TABLE statements
    queries = [query_to_string(q) for q, _ in mock_async_connection.executed_queries]
    create_table_found = any("CREATE TABLE" in q for q in queries)

    assert not create_table_found, "Should not create tables when they already exist"


@pytest.mark.asyncio
async def test_async_get_items_empty_session(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test async get_items returns empty list for new session."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    mock_async_connection.queue_result(MockAsyncResult(rows=[]))  # SELECT messages

    items = await session.get_items()
    assert items == []


@pytest.mark.asyncio
async def test_async_get_items_returns_parsed_json(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test async get_items correctly parses JSON data."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    test_messages = [
        {"message_data": json.dumps({"role": "user", "content": "Hello"})},
        {"message_data": json.dumps({"role": "assistant", "content": "Hi there!"})},
    ]

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    mock_async_connection.queue_result(MockAsyncResult(rows=test_messages))  # SELECT messages

    items = await session.get_items()

    assert len(items) == 2
    assert items[0]["role"] == "user"
    assert items[0]["content"] == "Hello"
    assert items[1]["role"] == "assistant"
    assert items[1]["content"] == "Hi there!"


@pytest.mark.asyncio
async def test_async_add_items(monkeypatch, mock_workspace, mock_async_pool, mock_async_connection):
    """Test async add_items inserts messages correctly."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    # Mock: tables exist
    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session

    test_items = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    await session.add_items(test_items)

    # Check that executemany was called on cursor
    assert len(mock_async_connection._cursor.executed_queries) > 0
    query, params = mock_async_connection._cursor.executed_queries[-1]
    query_str = query_to_string(query)
    assert "INSERT INTO" in query_str and "agent_messages" in query_str
    assert len(params) == 2  # Two items


@pytest.mark.asyncio
async def test_async_pop_item_returns_last_item(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test async pop_item removes and returns the most recent item."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    # DELETE RETURNING result
    mock_async_connection.queue_result(
        MockAsyncResult(
            rows=[{"message_data": json.dumps({"role": "assistant", "content": "Last msg"})}]
        )
    )
    mock_async_connection.queue_result(MockAsyncResult())  # UPDATE session timestamp

    popped = await session.pop_item()

    assert popped is not None
    assert popped["role"] == "assistant"
    assert popped["content"] == "Last msg"


@pytest.mark.asyncio
async def test_async_pop_item_returns_none_when_empty(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test async pop_item returns None for empty session."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    mock_async_connection.queue_result(MockAsyncResult(rows=[]))  # DELETE RETURNING - empty

    popped = await session.pop_item()

    assert popped is None


@pytest.mark.asyncio
async def test_async_clear_session(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test async clear_session deletes all messages for the session."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
    )

    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))  # tables exist check
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    mock_async_connection.queue_result(MockAsyncResult(rowcount=5))  # DELETE messages
    mock_async_connection.queue_result(MockAsyncResult())  # UPDATE session timestamp

    await session.clear_session()

    # Find the DELETE query
    delete_queries = [
        (q, p)
        for q, p in mock_async_connection.executed_queries
        if "DELETE FROM" in query_to_string(q)
        and "agent_messages" in query_to_string(q)
        and "WHERE session_id" in query_to_string(q)
    ]

    assert len(delete_queries) > 0, "Should execute DELETE query"
    query, params = delete_queries[0]
    assert params == (TEST_SESSION_ID,), "Should use correct session_id"


@pytest.mark.asyncio
async def test_async_custom_table_names(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test that async session uses custom table names correctly."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="test-instance",
        workspace_client=mock_workspace,
        sessions_table="custom_sessions",
        messages_table="custom_messages",
    )

    # Mock: tables don't exist (custom names), so they will be created
    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 0}]))  # tables don't exist
    mock_async_connection.queue_result(MockAsyncResult())  # CREATE sessions
    mock_async_connection.queue_result(MockAsyncResult())  # CREATE messages
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session
    mock_async_connection.queue_result(MockAsyncResult(rows=[]))  # SELECT for get_items

    await session.get_items()

    assert session.sessions_table == "custom_sessions"
    assert session.messages_table == "custom_messages"

    # Check that CREATE TABLE uses custom names
    queries = [query_to_string(q) for q, _ in mock_async_connection.executed_queries]
    assert any("custom_sessions" in q for q in queries)
    assert any("custom_messages" in q for q in queries)


@pytest.mark.asyncio
async def test_async_pool_caching(
    monkeypatch, mock_workspace, mock_async_pool, mock_async_connection
):
    """Test that async pools are cached and reused."""
    monkeypatch.setattr(lakebase, "AsyncConnectionPool", mock_async_pool)

    session1 = AsyncMemorySession(
        session_id=TEST_SESSION_ID,
        instance_name="shared-async-instance",
        workspace_client=mock_workspace,
    )

    session2 = AsyncMemorySession(
        session_id=TEST_SESSION_ID_2,
        instance_name="shared-async-instance",
        workspace_client=mock_workspace,
    )

    # Mock for initialization of first session
    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))  # tables exist
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session 1
    mock_async_connection.queue_result(MockAsyncResult(rows=[]))  # SELECT

    await session1.get_items()

    # Mock for second session (tables already exist check, but pool should be reused)
    mock_async_connection.queue_result(MockAsyncResult(rows=[{"cnt": 2}]))  # tables exist
    mock_async_connection.queue_result(MockAsyncResult())  # INSERT session 2
    mock_async_connection.queue_result(MockAsyncResult(rows=[]))  # SELECT

    await session2.get_items()

    # Both sessions should share the same pool
    assert session1._pool is session2._pool
