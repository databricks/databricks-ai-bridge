"""
Integration tests for LangChain Lakebase wrappers (DatabricksStore, CheckpointSaver).

These tests require:
1. A Lakebase instance to be available
2. Valid Databricks authentication (DATABRICKS_HOST + DATABRICKS_CLIENT_ID/SECRET or profile)

Set the environment variable:
    LAKEBASE_INSTANCE_NAME: Name of the Lakebase instance

Example:
    LAKEBASE_INSTANCE_NAME=my-lakebase pytest tests/integration_tests/test_langchain_lakebase.py -v
"""

from __future__ import annotations

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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def unique_namespace() -> tuple[str, str]:
    """Generate a UUID-based namespace tuple for test isolation."""
    return ("test", f"ns_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def cleanup_store_tables():
    """Fixture to track and clean up LangGraph store tables after tests."""
    tables_to_cleanup: list[str] = []

    yield tables_to_cleanup

    if tables_to_cleanup:
        from databricks_ai_bridge.lakebase import LakebasePool

        pool = LakebasePool(instance_name=get_instance_name())
        with pool.connection() as conn:
            for table in tables_to_cleanup:
                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        pool.close()


@pytest.fixture
def cleanup_checkpoint_tables():
    """Fixture to track and clean up LangGraph checkpoint tables after tests."""
    tables: list[str] = []

    yield tables

    if tables:
        from databricks_ai_bridge.lakebase import LakebasePool

        pool = LakebasePool(instance_name=get_instance_name())
        with pool.connection() as conn:
            for table in tables:
                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        pool.close()


# =============================================================================
# DatabricksStore (Sync) Tests
# =============================================================================


class TestDatabricksStore:
    """Test synchronous DatabricksStore against a live Lakebase instance."""

    def test_store_setup_put_and_get(self, unique_namespace, cleanup_store_tables):
        """Test core bridge path: pool creation -> _with_store -> setup + batch (put/get)."""
        from databricks_langchain import DatabricksStore

        cleanup_store_tables.append("store")

        store = DatabricksStore(instance_name=get_instance_name())
        store.setup()

        ns = unique_namespace
        store.put(ns, "key1", {"data": "hello world"})

        item = store.get(ns, "key1")
        assert item is not None
        assert item.value == {"data": "hello world"}
        assert item.key == "key1"
        assert item.namespace == ns

    def test_store_search(self, unique_namespace, cleanup_store_tables):
        """Test search operation through bridge."""
        from databricks_langchain import DatabricksStore

        cleanup_store_tables.append("store")

        store = DatabricksStore(instance_name=get_instance_name())
        store.setup()

        ns = unique_namespace
        store.put(ns, "item_a", {"topic": "python"})
        store.put(ns, "item_b", {"topic": "rust"})

        results = store.search(ns)
        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"item_a", "item_b"}

    def test_store_delete(self, unique_namespace, cleanup_store_tables):
        """Test delete operation through bridge."""
        from databricks_langchain import DatabricksStore

        cleanup_store_tables.append("store")

        store = DatabricksStore(instance_name=get_instance_name())
        store.setup()

        ns = unique_namespace
        store.put(ns, "to_delete", {"temp": True})
        assert store.get(ns, "to_delete") is not None

        store.delete(ns, "to_delete")
        assert store.get(ns, "to_delete") is None


# =============================================================================
# AsyncDatabricksStore Tests
# =============================================================================


class TestAsyncDatabricksStore:
    """Test asynchronous AsyncDatabricksStore against a live Lakebase instance."""

    @pytest.mark.asyncio
    async def test_async_store_put_and_get(self, unique_namespace, cleanup_store_tables):
        """Test async pool open/close/borrow cycle with put and get."""
        from databricks_langchain import AsyncDatabricksStore

        cleanup_store_tables.append("store")

        store = AsyncDatabricksStore(instance_name=get_instance_name())
        await store._lakebase.open()
        try:
            await store.setup()

            ns = unique_namespace
            await store.aput(ns, "async_key", {"data": "async hello"})

            item = await store.aget(ns, "async_key")
            assert item is not None
            assert item.value == {"data": "async hello"}
            assert item.key == "async_key"
        finally:
            await store._lakebase.close()

    @pytest.mark.asyncio
    async def test_async_store_context_manager(self, unique_namespace, cleanup_store_tables):
        """Test async with lifecycle (open/close via context manager)."""
        from databricks_langchain import AsyncDatabricksStore

        cleanup_store_tables.append("store")

        async with AsyncDatabricksStore(instance_name=get_instance_name()) as store:
            await store.setup()

            ns = unique_namespace
            await store.aput(ns, "ctx_key", {"data": "context managed"})

            item = await store.aget(ns, "ctx_key")
            assert item is not None
            assert item.value == {"data": "context managed"}


# =============================================================================
# CheckpointSaver (Sync) Tests
# =============================================================================

CHECKPOINT_TABLES = [
    "checkpoint_migrations",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoints",
]


class TestCheckpointSaver:
    """Test synchronous CheckpointSaver against a live Lakebase instance."""

    def test_checkpoint_write_and_read(self, cleanup_checkpoint_tables):
        """Test pool handoff to PostgresSaver: setup, put, get_tuple."""
        from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

        from databricks_langchain import CheckpointSaver

        cleanup_checkpoint_tables.extend(CHECKPOINT_TABLES)

        thread_id = uuid.uuid4().hex

        with CheckpointSaver(instance_name=get_instance_name()) as saver:
            saver.setup()

            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = Checkpoint(
                v=1,
                id=uuid.uuid4().hex,
                ts="2025-01-01T00:00:00+00:00",
                channel_values={},
                channel_versions={},
                versions_seen={},
                pending_sends=[],
            )
            metadata = CheckpointMetadata()

            saver.put(config, checkpoint, metadata, {})

            result = saver.get_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]

    def test_checkpoint_list(self, cleanup_checkpoint_tables):
        """Test listing checkpoints."""
        from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

        from databricks_langchain import CheckpointSaver

        cleanup_checkpoint_tables.extend(CHECKPOINT_TABLES)

        thread_id = uuid.uuid4().hex

        with CheckpointSaver(instance_name=get_instance_name()) as saver:
            saver.setup()

            config = {"configurable": {"thread_id": thread_id}}

            for i in range(3):
                checkpoint = Checkpoint(
                    v=1,
                    id=uuid.uuid4().hex,
                    ts=f"2025-01-01T00:0{i}:00+00:00",
                    channel_values={},
                    channel_versions={},
                    versions_seen={},
                    pending_sends=[],
                )
                saver.put(config, checkpoint, CheckpointMetadata(), {})

            checkpoints = list(saver.list(config))
            assert len(checkpoints) == 3


# =============================================================================
# AsyncCheckpointSaver Tests
# =============================================================================


class TestAsyncCheckpointSaver:
    """Test asynchronous AsyncCheckpointSaver against a live Lakebase instance."""

    @pytest.mark.asyncio
    async def test_async_checkpoint_write_and_read(self, cleanup_checkpoint_tables):
        """Test async pool lifecycle: setup, put, get_tuple."""
        from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

        from databricks_langchain import AsyncCheckpointSaver

        cleanup_checkpoint_tables.extend(CHECKPOINT_TABLES)

        thread_id = uuid.uuid4().hex

        async with AsyncCheckpointSaver(instance_name=get_instance_name()) as saver:
            await saver.setup()

            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = Checkpoint(
                v=1,
                id=uuid.uuid4().hex,
                ts="2025-01-01T00:00:00+00:00",
                channel_values={},
                channel_versions={},
                versions_seen={},
                pending_sends=[],
            )
            metadata = CheckpointMetadata()

            await saver.aput(config, checkpoint, metadata, {})

            result = await saver.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]

    @pytest.mark.asyncio
    async def test_async_checkpoint_context_manager(self, cleanup_checkpoint_tables):
        """Test async with lifecycle: open/close via context manager."""
        from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

        from databricks_langchain import AsyncCheckpointSaver

        cleanup_checkpoint_tables.extend(CHECKPOINT_TABLES)

        thread_id = uuid.uuid4().hex

        async with AsyncCheckpointSaver(instance_name=get_instance_name()) as saver:
            await saver.setup()

            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = Checkpoint(
                v=1,
                id=uuid.uuid4().hex,
                ts="2025-01-01T00:00:00+00:00",
                channel_values={},
                channel_versions={},
                versions_seen={},
                pending_sends=[],
            )

            await saver.aput(config, checkpoint, CheckpointMetadata(), {})

            result = await saver.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]
