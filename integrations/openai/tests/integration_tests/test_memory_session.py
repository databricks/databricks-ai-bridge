"""
Integration tests for AsyncDatabricksSession.

These tests require:
1. A Lakebase instance to be available
2. Valid Databricks authentication (DATABRICKS_HOST + DATABRICKS_TOKEN or profile)

Set the environment variable:
    LAKEBASE_INSTANCE_NAME: Name of the Lakebase instance

Example:
    LAKEBASE_INSTANCE_NAME=lakebase pytest tests/integration_tests/test_memory_session.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any, cast

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
        pool.close()


# =============================================================================
# AsyncDatabricksSession Tests
# =============================================================================


@pytest.mark.asyncio
async def test_memory_session_crud_operations(cleanup_tables):
    """
    Comprehensive CRUD test for AsyncDatabricksSession.

    Tests the full lifecycle:
    - clear_session() on fresh session
    - get_items() returns empty list for new session
    - add_items() stores messages
    - get_items() retrieves stored messages
    - get_items(limit=N) returns latest N items in order
    - pop_item() removes and returns most recent item
    - clear_session() removes all items
    """
    from databricks_openai.agents import AsyncDatabricksSession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session_id = str(uuid.uuid4())
    session = AsyncDatabricksSession(
        session_id=session_id,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Clear any existing data (should be no-op for new session)
    await session.clear_session()

    # Test get_items on empty session
    items = cast(list[Any], await session.get_items())
    assert items == [], f"Expected empty list, got {items}"

    # Test add_items
    test_items: list[Any] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]
    await session.add_items(test_items)

    # Test get_items returns what we added
    items = cast(list[Any], await session.get_items())
    assert len(items) == 2, f"Expected 2 items, got {len(items)}"
    assert items[0]["role"] == "user"
    assert items[0]["content"] == "Hello, how are you?"
    assert items[1]["role"] == "assistant"
    assert items[1]["content"] == "I'm doing well, thank you!"

    # Test get_items with limit - should return latest N items in chronological order
    items = cast(list[Any], await session.get_items(limit=1))
    assert len(items) == 1, f"Expected 1 item with limit, got {len(items)}"
    assert items[0]["role"] == "assistant"  # Latest item

    # Test pop_item - removes and returns the last item
    popped = cast(Any, await session.pop_item())
    assert popped is not None
    assert popped["role"] == "assistant"  # Should be the last item

    # Verify only 1 item remains
    items = cast(list[Any], await session.get_items())
    assert len(items) == 1, f"Expected 1 item after pop, got {len(items)}"
    assert items[0]["role"] == "user"

    # Test clear_session
    await session.clear_session()
    items = cast(list[Any], await session.get_items())
    assert items == [], f"Expected empty after clear, got {items}"


@pytest.mark.asyncio
async def test_memory_session_multiple_sessions_isolated(cleanup_tables):
    """Test that different session_ids have isolated data."""
    from databricks_openai.agents import AsyncDatabricksSession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session_id_1 = str(uuid.uuid4())
    session_id_2 = str(uuid.uuid4())

    session_1 = AsyncDatabricksSession(
        session_id=session_id_1,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    session_2 = AsyncDatabricksSession(
        session_id=session_id_2,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
        create_tables=False,  # Tables already created by session_1
    )

    # Add different items to each session
    items_1_data: list[Any] = [{"role": "user", "content": "Session 1 message"}]
    items_2_data: list[Any] = [{"role": "user", "content": "Session 2 message"}]
    await session_1.add_items(items_1_data)
    await session_2.add_items(items_2_data)

    # Verify isolation
    items_1 = cast(list[Any], await session_1.get_items())
    items_2 = cast(list[Any], await session_2.get_items())

    assert len(items_1) == 1
    assert len(items_2) == 1
    assert items_1[0]["content"] == "Session 1 message"
    assert items_2[0]["content"] == "Session 2 message"

    # Clear one session shouldn't affect the other
    await session_1.clear_session()
    items_1 = cast(list[Any], await session_1.get_items())
    items_2 = cast(list[Any], await session_2.get_items())
    assert len(items_1) == 0
    assert len(items_2) == 1

    # Cleanup
    await session_2.clear_session()


@pytest.mark.asyncio
async def test_memory_session_pop_empty_returns_none(cleanup_tables):
    """Test that pop_item returns None on empty session."""
    from databricks_openai.agents import AsyncDatabricksSession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session = AsyncDatabricksSession(
        session_id=str(uuid.uuid4()),
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Pop on empty session should return None
    popped = await session.pop_item()
    assert popped is None


@pytest.mark.asyncio
async def test_memory_session_add_empty_items_noop(cleanup_tables):
    """Test that add_items with empty list is a no-op."""
    from databricks_openai.agents import AsyncDatabricksSession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session = AsyncDatabricksSession(
        session_id=str(uuid.uuid4()),
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Add empty list - should not raise
    await session.add_items([])

    # Session should still be empty
    items = cast(list[Any], await session.get_items())
    assert items == []


@pytest.mark.asyncio
async def test_memory_session_complex_message_data(cleanup_tables):
    """Test storing complex message data with nested structures."""
    from databricks_openai.agents import AsyncDatabricksSession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session = AsyncDatabricksSession(
        session_id=str(uuid.uuid4()),
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Add complex item with nested data
    complex_item = {
        "role": "assistant",
        "content": "Here's your result",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco"}',
                },
            }
        ],
        "metadata": {
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
    }
    await session.add_items([complex_item])  # type: ignore[list-item]

    # Retrieve and verify
    items = cast(list[Any], await session.get_items())
    assert len(items) == 1
    assert items[0]["role"] == "assistant"
    assert items[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert items[0]["metadata"]["usage"]["prompt_tokens"] == 10

    # Cleanup
    await session.clear_session()


@pytest.mark.asyncio
async def test_memory_session_properties(cleanup_tables):
    """Test that session properties return correct values."""
    from databricks_openai.agents import AsyncDatabricksSession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session_id = str(uuid.uuid4())
    session = AsyncDatabricksSession(
        session_id=session_id,
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Verify properties
    assert session.instance_name == get_instance_name()
    assert session.host is not None
    assert len(session.host) > 0
    assert session.username is not None
    assert len(session.username) > 0
    assert session.session_id == session_id


@pytest.mark.asyncio
async def test_memory_session_get_items_ordering(cleanup_tables):
    """Test that get_items returns items in correct chronological order."""
    from databricks_openai.agents import AsyncDatabricksSession

    sessions_table, messages_table = get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session = AsyncDatabricksSession(
        session_id=str(uuid.uuid4()),
        instance_name=get_instance_name(),
        sessions_table=sessions_table,
        messages_table=messages_table,
    )

    # Add multiple items
    test_items: list[Any] = [{"role": "user", "content": f"Message {i}"} for i in range(5)]
    await session.add_items(test_items)

    # Get all items - should be in chronological order
    items = cast(list[Any], await session.get_items())
    assert len(items) == 5
    for i, item in enumerate(items):
        assert item["content"] == f"Message {i}"

    # Get with limit - should return LATEST N items in chronological order
    items = cast(list[Any], await session.get_items(limit=3))
    assert len(items) == 3
    # Should be messages 2, 3, 4 (the latest 3) in order
    assert items[0]["content"] == "Message 2"
    assert items[1]["content"] == "Message 3"
    assert items[2]["content"] == "Message 4"

    # Cleanup
    await session.clear_session()
