"""Integration tests for MemorySession and AsyncMemorySession.

These tests require:
1. A Lakebase instance to be available
2. Valid Databricks authentication (DATABRICKS_HOST + DATABRICKS_TOKEN as env variables)

Set the environment variable:
    LAKEBASE_INSTANCE_NAME: Name or hostname of the Lakebase instance

Example:
    LAKEBASE_INSTANCE_NAME=lakebase pytest tests/integration_tests/test_memory_session.py -v
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

# Skip all tests if LAKEBASE_INSTANCE_NAME is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("LAKEBASE_INSTANCE_NAME"),
    reason="LAKEBASE_INSTANCE_NAME environment variable not set",
)


def get_instance_name() -> str:
    """Get the Lakebase instance name from environment."""
    return os.environ["LAKEBASE_INSTANCE_NAME"]


def get_unique_table_names() -> tuple[str, str]:
    """Generate unique table names for test isolation."""
    suffix = uuid.uuid4().hex[:8]
    return f"test_sessions_{suffix}", f"test_messages_{suffix}"


@pytest.fixture(scope="session", autouse=True)
def cleanup_pool_cache():
    """Session-scoped fixture to close cached pools after all tests complete."""
    yield

    # Close sync pool cache
    from databricks_openai.agents import session as session_module

    for pool in session_module._pool_cache.values():
        try:
            pool.close()
        except Exception:
            pass
    session_module._pool_cache.clear()

    # Close async pool cache - need to handle event loop carefully
    for pool in list(session_module._async_pool_cache.values()):
        try:
            # Access the underlying pool and close it synchronously if possible
            # The pool's _pool attribute is the actual AsyncConnectionPool
            if hasattr(pool, "_pool") and pool._pool is not None:
                # Use wait=False to avoid blocking on workers
                pool._pool.close(timeout=0)
        except Exception:
            pass
    session_module._async_pool_cache.clear()


@pytest.fixture
def cleanup_tables():
    """Fixture to track and clean up test tables after tests."""
    tables_to_cleanup: list[tuple[str, str]] = []

    yield tables_to_cleanup

    # Cleanup after test
    if tables_to_cleanup:
        from databricks_ai_bridge.lakebase import LakebasePool

        pool = LakebasePool(instance_name=get_instance_name())
        with pool.connection() as conn:
            for sessions_table, messages_table in tables_to_cleanup:
                # Drop messages first (foreign key constraint)
                conn.execute(f"DROP TABLE IF EXISTS {messages_table}")
                conn.execute(f"DROP TABLE IF EXISTS {sessions_table}")


# =============================================================================
# Sync MemorySession Tests
# =============================================================================


def test_memory_session_crud_operations(cleanup_tables):
    """
    Comprehensive CRUD test for sync MemorySession.

    Tests the full lifecycle:
    - clear_session() on fresh session
    - get_items() returns empty list for new session
    - add_items() stores messages
    - get_items() retrieves stored messages
    - get_items(limit=N) returns latest N items in order
    - pop_item() removes and returns most recent item
    - clear_session() removes all items
    """
    from databricks_openai.agents.session import MemorySession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session_id = uuid.uuid4()
    session = MemorySession(
        session_id=session_id,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Clear any existing data (should be no-op for new session)
    asyncio.run(session.clear_session())

    # Test get_items on empty session
    items = asyncio.run(session.get_items())
    assert items == [], f"Expected empty list, got {items}"

    # Test add_items
    test_items = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]
    asyncio.run(session.add_items(test_items))

    # Test get_items returns what we added
    items = asyncio.run(session.get_items())
    assert len(items) == 2, f"Expected 2 items, got {len(items)}"
    assert items[0]["role"] == "user"
    assert items[0]["content"] == "Hello, how are you?"
    assert items[1]["role"] == "assistant"
    assert items[1]["content"] == "I'm doing well, thank you!"

    # Test get_items with limit - should return latest N items in chronological order
    items = asyncio.run(session.get_items(limit=1))
    assert len(items) == 1, f"Expected 1 item with limit, got {len(items)}"
    assert items[0]["role"] == "assistant"  # Latest item

    # Test pop_item - removes and returns the last item
    popped = asyncio.run(session.pop_item())
    assert popped is not None
    assert popped["role"] == "assistant"  # Should be the last item

    # Verify only 1 item remains
    items = asyncio.run(session.get_items())
    assert len(items) == 1, f"Expected 1 item after pop, got {len(items)}"
    assert items[0]["role"] == "user"

    # Test clear_session
    asyncio.run(session.clear_session())
    items = asyncio.run(session.get_items())
    assert items == [], f"Expected empty after clear, got {items}"


def test_memory_session_multiple_sessions_isolated(cleanup_tables):
    """Test that different session_ids have isolated data."""
    from databricks_openai.agents.session import MemorySession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session_id_1 = uuid.uuid4()
    session_id_2 = uuid.uuid4()

    session_1 = MemorySession(
        session_id=session_id_1,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    session_2 = MemorySession(
        session_id=session_id_2,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Add different items to each session
    asyncio.run(session_1.add_items([{"role": "user", "content": "Session 1 message"}]))
    asyncio.run(session_2.add_items([{"role": "user", "content": "Session 2 message"}]))

    # Verify isolation
    items_1 = asyncio.run(session_1.get_items())
    items_2 = asyncio.run(session_2.get_items())

    assert len(items_1) == 1
    assert len(items_2) == 1
    assert items_1[0]["content"] == "Session 1 message"
    assert items_2[0]["content"] == "Session 2 message"

    # Clear one session shouldn't affect the other
    asyncio.run(session_1.clear_session())
    items_1 = asyncio.run(session_1.get_items())
    items_2 = asyncio.run(session_2.get_items())
    assert len(items_1) == 0
    assert len(items_2) == 1


def test_memory_session_pop_empty_returns_none(cleanup_tables):
    """Test that pop_item returns None on empty session."""
    from databricks_openai.agents.session import MemorySession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session = MemorySession(
        session_id=uuid.uuid4(),
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Pop on empty session should return None
    popped = asyncio.run(session.pop_item())
    assert popped is None


def test_memory_session_add_empty_items_noop(cleanup_tables):
    """Test that add_items with empty list is a no-op."""
    from databricks_openai.agents.session import MemorySession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session = MemorySession(
        session_id=uuid.uuid4(),
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Add empty list - should not raise
    asyncio.run(session.add_items([]))

    # Session should still be empty
    items = asyncio.run(session.get_items())
    assert items == []


# =============================================================================
# Async AsyncMemorySession Tests
# =============================================================================


@pytest.mark.asyncio
async def test_async_memory_session_crud_operations(cleanup_tables):
    """
    Comprehensive CRUD test for AsyncMemorySession.

    Tests the full lifecycle:
    - clear_session() on fresh session
    - get_items() returns empty list for new session
    - add_items() stores messages
    - get_items() retrieves stored messages
    - get_items(limit=N) returns latest N items in order
    - pop_item() removes and returns most recent item
    - clear_session() removes all items
    """
    from databricks_openai.agents.session import AsyncMemorySession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session_id = uuid.uuid4()
    session = AsyncMemorySession(
        session_id=session_id,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Clear any existing data (should be no-op for new session)
    await session.clear_session()

    # Test get_items on empty session
    items = await session.get_items()
    assert items == [], f"Expected empty list, got {items}"

    # Test add_items
    test_items = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]
    await session.add_items(test_items)

    # Test get_items returns what we added
    items = await session.get_items()
    assert len(items) == 2, f"Expected 2 items, got {len(items)}"
    assert items[0]["role"] == "user"
    assert items[0]["content"] == "Hello, how are you?"
    assert items[1]["role"] == "assistant"
    assert items[1]["content"] == "I'm doing well, thank you!"

    # Test get_items with limit - should return latest N items in chronological order
    items = await session.get_items(limit=1)
    assert len(items) == 1, f"Expected 1 item with limit, got {len(items)}"
    assert items[0]["role"] == "assistant"  # Latest item

    # Test pop_item - removes and returns the last item
    popped = await session.pop_item()
    assert popped is not None
    assert popped["role"] == "assistant"  # Should be the last item

    # Verify only 1 item remains
    items = await session.get_items()
    assert len(items) == 1, f"Expected 1 item after pop, got {len(items)}"
    assert items[0]["role"] == "user"

    # Test clear_session
    await session.clear_session()
    items = await session.get_items()
    assert items == [], f"Expected empty after clear, got {items}"


@pytest.mark.asyncio
async def test_async_memory_session_multiple_sessions_isolated(cleanup_tables):
    """Test that different session_ids have isolated data (async version)."""
    from databricks_openai.agents.session import AsyncMemorySession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session_id_1 = uuid.uuid4()
    session_id_2 = uuid.uuid4()

    session_1 = AsyncMemorySession(
        session_id=session_id_1,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    session_2 = AsyncMemorySession(
        session_id=session_id_2,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Add different items to each session
    await session_1.add_items([{"role": "user", "content": "Async Session 1 message"}])
    await session_2.add_items([{"role": "user", "content": "Async Session 2 message"}])

    # Verify isolation
    items_1 = await session_1.get_items()
    items_2 = await session_2.get_items()

    assert len(items_1) == 1
    assert len(items_2) == 1
    assert items_1[0]["content"] == "Async Session 1 message"
    assert items_2[0]["content"] == "Async Session 2 message"

    # Clear one session shouldn't affect the other
    await session_1.clear_session()
    items_1 = await session_1.get_items()
    items_2 = await session_2.get_items()
    assert len(items_1) == 0
    assert len(items_2) == 1


@pytest.mark.asyncio
async def test_async_memory_session_pop_empty_returns_none(cleanup_tables):
    """Test that pop_item returns None on empty session (async version)."""
    from databricks_openai.agents.session import AsyncMemorySession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session = AsyncMemorySession(
        session_id=uuid.uuid4(),
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Pop on empty session should return None
    popped = await session.pop_item()
    assert popped is None


@pytest.mark.asyncio
async def test_async_memory_session_add_empty_items_noop(cleanup_tables):
    """Test that add_items with empty list is a no-op (async version)."""
    from databricks_openai.agents.session import AsyncMemorySession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session = AsyncMemorySession(
        session_id=uuid.uuid4(),
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Add empty list - should not raise
    await session.add_items([])

    # Session should still be empty
    items = await session.get_items()
    assert items == []
