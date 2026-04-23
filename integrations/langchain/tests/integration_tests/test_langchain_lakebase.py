"""
Integration tests for LangChain Lakebase wrappers (DatabricksStore, CheckpointSaver).

These tests require:
1. A Lakebase instance to be available (provisioned or autoscaling)
2. Valid Databricks authentication (DATABRICKS_HOST + DATABRICKS_CLIENT_ID/SECRET or profile)

Set at least one of these environment variables:
    LAKEBASE_INSTANCE_NAME: Name of the Lakebase provisioned instance
    LAKEBASE_PROJECT + LAKEBASE_BRANCH: Autoscaling project and branch names
    LAKEBASE_AUTOSCALING_ENDPOINT: Full autoscaling endpoint resource path

Example (provisioned):
    LAKEBASE_INSTANCE_NAME=my-lakebase pytest tests/integration_tests/test_langchain_lakebase.py -v

Example (autoscaling):
    LAKEBASE_PROJECT=my-project LAKEBASE_BRANCH=main \
        pytest tests/integration_tests/test_langchain_lakebase.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

# Skip the entire module if the memory extra (langgraph) is not installed
pytest.importorskip("langgraph", reason="langgraph not installed (requires memory extra)")

from databricks_ai_bridge.lakebase import LakebaseClient
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from databricks_langchain import (
    AsyncCheckpointSaver,
    AsyncDatabricksStore,
    CheckpointSaver,
    DatabricksStore,
)

# Skip all tests if no Lakebase env vars are set
pytestmark = pytest.mark.skipif(
    not os.environ.get("LAKEBASE_INSTANCE_NAME")
    and not os.environ.get("LAKEBASE_PROJECT")
    and not os.environ.get("LAKEBASE_AUTOSCALING_ENDPOINT"),
    reason="No Lakebase env vars set "
    "(need LAKEBASE_INSTANCE_NAME, LAKEBASE_PROJECT, or LAKEBASE_AUTOSCALING_ENDPOINT)",
)

_skip_no_instance = pytest.mark.skipif(
    not os.environ.get("LAKEBASE_INSTANCE_NAME"),
    reason="LAKEBASE_INSTANCE_NAME not set",
)

_skip_no_project_branch = pytest.mark.skipif(
    not os.environ.get("LAKEBASE_PROJECT") or not os.environ.get("LAKEBASE_BRANCH"),
    reason="LAKEBASE_PROJECT and LAKEBASE_BRANCH not set",
)

_skip_no_endpoint = pytest.mark.skipif(
    not os.environ.get("LAKEBASE_AUTOSCALING_ENDPOINT"),
    reason="LAKEBASE_AUTOSCALING_ENDPOINT not set",
)


def get_instance_name() -> str:
    """Get the Lakebase instance name from environment."""
    return os.environ["LAKEBASE_INSTANCE_NAME"]


def get_project() -> str:
    return os.environ["LAKEBASE_PROJECT"]


def get_branch() -> str:
    return os.environ["LAKEBASE_BRANCH"]


def get_autoscaling_endpoint() -> str:
    return os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"]


# =============================================================================
# Tables managed by LangGraph that must be cleaned up between test runs.
# Includes both data tables and migration-tracking tables; PostgresStore's
# setup() is a no-op when the migration table already marks a version as
# applied, so we must always drop both together.
# =============================================================================

STORE_TABLES = ["store_vectors", "vector_migrations", "store", "store_migrations"]
CHECKPOINT_TABLES = [
    "checkpoint_migrations",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoints",
]
ALL_TABLES = STORE_TABLES + CHECKPOINT_TABLES


def _drop_tables(tables: list[str], **client_kwargs) -> None:
    """Drop the given tables from a Lakebase instance."""
    if not client_kwargs:
        client_kwargs = {"instance_name": get_instance_name()}
    with LakebaseClient(**client_kwargs) as client:
        for table in tables:
            client.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def unique_namespace() -> tuple[str, str]:
    """Generate a UUID-based namespace tuple for test isolation."""
    return ("test", f"ns_{uuid.uuid4().hex[:8]}")


@pytest.fixture(scope="module")
def cleanup_store_tables():
    """Drop store tables before and after all provisioned store tests.

    scope="module" means tables are dropped once at the start of the module and
    once at the end — NOT before/after each individual test. This keeps tests
    fast while still cleaning up stale migration-tracking tables left by
    previous CI runs (PostgresStore.setup() silently skips table creation when
    its migration tracker says the schema is already at the latest version).
    """
    _drop_tables(STORE_TABLES)
    yield
    _drop_tables(STORE_TABLES)


@pytest.fixture(scope="module")
def cleanup_checkpoint_tables():
    """Drop checkpoint tables before and after all provisioned checkpoint tests.

    scope="module" means tables are dropped once at the start of the module and
    once at the end — NOT before/after each individual test.
    """
    _drop_tables(CHECKPOINT_TABLES)
    yield
    _drop_tables(CHECKPOINT_TABLES)


@pytest.fixture(scope="module")
def cleanup_all_tables_project_branch():
    """Drop all LangGraph tables on the project/branch autoscaling database."""
    if not os.environ.get("LAKEBASE_PROJECT") or not os.environ.get("LAKEBASE_BRANCH"):
        yield
        return
    kwargs = {"project": get_project(), "branch": get_branch()}
    _drop_tables(ALL_TABLES, **kwargs)
    yield
    _drop_tables(ALL_TABLES, **kwargs)


@pytest.fixture(scope="module")
def cleanup_all_tables_endpoint():
    """Drop all LangGraph tables on the endpoint autoscaling database."""
    if not os.environ.get("LAKEBASE_AUTOSCALING_ENDPOINT"):
        yield
        return
    kwargs = {"autoscaling_endpoint": get_autoscaling_endpoint()}
    _drop_tables(ALL_TABLES, **kwargs)
    yield
    _drop_tables(ALL_TABLES, **kwargs)


def _make_checkpoint(ts: str = "2025-01-01T00:00:00+00:00") -> Checkpoint:
    """Build a Checkpoint with a random ID and the given timestamp."""
    return Checkpoint(
        v=1,
        id=uuid.uuid4().hex,
        ts=ts,
        channel_values={},
        channel_versions={},
        versions_seen={},
        pending_sends=[],
    )


# =============================================================================
# DatabricksStore (Sync) Tests — Provisioned
# =============================================================================


@_skip_no_instance
class TestDatabricksStore:
    """Test synchronous DatabricksStore against a live Lakebase instance."""

    def test_store_setup_put_and_get(self, unique_namespace, cleanup_store_tables):
        """Test core bridge path: pool creation -> _with_store -> setup + batch (put/get)."""
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
        store = DatabricksStore(instance_name=get_instance_name())
        store.setup()

        ns = unique_namespace
        store.put(ns, "to_delete", {"temp": True})
        assert store.get(ns, "to_delete") is not None

        store.delete(ns, "to_delete")
        assert store.get(ns, "to_delete") is None

    def test_store_vector_search(self, unique_namespace, cleanup_store_tables):
        """Test vector-based search with embedding endpoint via PostgresIndexConfig."""
        store = DatabricksStore(
            instance_name=get_instance_name(),
            embedding_endpoint="databricks-bge-large-en",
            embedding_dims=1024,
        )
        store.setup()

        ns = unique_namespace
        store.put(ns, "doc_python", {"text": "Python is a programming language"})
        store.put(ns, "doc_coffee", {"text": "Coffee is a popular beverage"})

        results = store.search(ns, query="programming languages")
        assert len(results) > 0
        # The programming-related doc should rank higher than the coffee doc
        assert results[0].key == "doc_python"


# =============================================================================
# AsyncDatabricksStore Tests — Provisioned
# =============================================================================


@_skip_no_instance
class TestAsyncDatabricksStore:
    """Test asynchronous AsyncDatabricksStore against a live Lakebase instance."""

    @pytest.mark.asyncio
    async def test_async_store_setup_put_and_get(self, unique_namespace, cleanup_store_tables):
        """Test async put and get operations through bridge."""
        async with AsyncDatabricksStore(instance_name=get_instance_name()) as store:
            await store.setup()

            ns = unique_namespace
            await store.aput(ns, "async_key", {"data": "async hello"})

            item = await store.aget(ns, "async_key")
            assert item is not None
            assert item.value == {"data": "async hello"}
            assert item.key == "async_key"

        # Pool should be closed after exiting the context manager
        assert store._lakebase.pool.closed

    @pytest.mark.asyncio
    async def test_async_store_search(self, unique_namespace, cleanup_store_tables):
        """Test async search operation through bridge."""
        async with AsyncDatabricksStore(instance_name=get_instance_name()) as store:
            await store.setup()

            ns = unique_namespace
            await store.aput(ns, "item_a", {"topic": "python"})
            await store.aput(ns, "item_b", {"topic": "rust"})

            results = await store.asearch(ns)
            assert len(results) == 2
            keys = {r.key for r in results}
            assert keys == {"item_a", "item_b"}

    @pytest.mark.asyncio
    async def test_async_store_delete(self, unique_namespace, cleanup_store_tables):
        """Test async delete operation through bridge."""
        async with AsyncDatabricksStore(instance_name=get_instance_name()) as store:
            await store.setup()

            ns = unique_namespace
            await store.aput(ns, "to_delete", {"temp": True})
            assert (await store.aget(ns, "to_delete")) is not None

            await store.adelete(ns, "to_delete")
            assert (await store.aget(ns, "to_delete")) is None

    @pytest.mark.asyncio
    async def test_async_store_vector_search(self, unique_namespace, cleanup_store_tables):
        """Test async vector-based search with embedding endpoint via PostgresIndexConfig."""
        async with AsyncDatabricksStore(
            instance_name=get_instance_name(),
            embedding_endpoint="databricks-bge-large-en",
            embedding_dims=1024,
        ) as store:
            await store.setup()

            ns = unique_namespace
            await store.aput(ns, "doc_python", {"text": "Python is a programming language"})
            await store.aput(ns, "doc_coffee", {"text": "Coffee is a popular beverage"})

            results = await store.asearch(ns, query="programming languages")
            assert len(results) > 0
            # The programming-related doc should rank higher than the coffee doc
            assert results[0].key == "doc_python"


# =============================================================================
# CheckpointSaver (Sync) Tests — Provisioned
# =============================================================================


@_skip_no_instance
class TestCheckpointSaver:
    """Test synchronous CheckpointSaver against a live Lakebase instance."""

    def test_checkpoint_write_and_read(self, cleanup_checkpoint_tables):
        """Test pool handoff to PostgresSaver: setup, put, get_tuple, and pool cleanup."""
        thread_id = uuid.uuid4().hex

        with CheckpointSaver(instance_name=get_instance_name()) as saver:
            saver.setup()

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            checkpoint = _make_checkpoint()
            metadata = CheckpointMetadata()

            saver.put(config, checkpoint, metadata, {})

            result = saver.get_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]

        # Pool should be closed after exiting the context manager
        assert saver._lakebase.pool.closed

    def test_checkpoint_list(self, cleanup_checkpoint_tables):
        """Test listing checkpoints."""
        thread_id = uuid.uuid4().hex

        with CheckpointSaver(instance_name=get_instance_name()) as saver:
            saver.setup()

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            for i in range(3):
                checkpoint = _make_checkpoint(ts=f"2025-01-01T00:0{i}:00+00:00")
                saver.put(config, checkpoint, CheckpointMetadata(), {})

            checkpoints = list(saver.list(config))
            assert len(checkpoints) == 3


# =============================================================================
# AsyncCheckpointSaver Tests — Provisioned
# =============================================================================


@_skip_no_instance
class TestAsyncCheckpointSaver:
    """Test asynchronous AsyncCheckpointSaver against a live Lakebase instance."""

    @pytest.mark.asyncio
    async def test_async_checkpoint_write_and_read(self, cleanup_checkpoint_tables):
        """Test async pool lifecycle: setup, put, get_tuple, and pool cleanup."""
        thread_id = uuid.uuid4().hex

        async with AsyncCheckpointSaver(instance_name=get_instance_name()) as saver:
            await saver.setup()

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            checkpoint = _make_checkpoint()
            metadata = CheckpointMetadata()

            await saver.aput(config, checkpoint, metadata, {})

            result = await saver.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]

        # Pool should be closed after exiting the context manager
        assert saver._lakebase.pool.closed

    @pytest.mark.asyncio
    async def test_async_checkpoint_list(self, cleanup_checkpoint_tables):
        """Test async listing checkpoints."""
        thread_id = uuid.uuid4().hex

        async with AsyncCheckpointSaver(instance_name=get_instance_name()) as saver:
            await saver.setup()

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            for i in range(3):
                checkpoint = _make_checkpoint(ts=f"2025-01-01T00:0{i}:00+00:00")
                await saver.aput(config, checkpoint, CheckpointMetadata(), {})

            checkpoints = [c async for c in saver.alist(config)]
            assert len(checkpoints) == 3


# =============================================================================
# Autoscaling — Project/Branch
# =============================================================================


@_skip_no_project_branch
class TestAutoscalingProjectBranch:
    """Test all LangChain Lakebase wrappers with autoscaling project/branch mode."""

    def test_store_put_and_get(self, unique_namespace, cleanup_all_tables_project_branch):
        """Test DatabricksStore: autoscaling params forwarded to LakebasePool."""
        store = DatabricksStore(project=get_project(), branch=get_branch())
        store.setup()

        ns = unique_namespace
        store.put(ns, "key1", {"data": "autoscaling hello"})

        item = store.get(ns, "key1")
        assert item is not None
        assert item.value == {"data": "autoscaling hello"}
        assert item.key == "key1"

    @pytest.mark.asyncio
    async def test_async_store_put_and_get(
        self, unique_namespace, cleanup_all_tables_project_branch
    ):
        """Test AsyncDatabricksStore: async pool open/close + put + get."""
        async with AsyncDatabricksStore(project=get_project(), branch=get_branch()) as store:
            await store.setup()

            ns = unique_namespace
            await store.aput(ns, "async_key", {"data": "async autoscaling"})

            item = await store.aget(ns, "async_key")
            assert item is not None
            assert item.value == {"data": "async autoscaling"}

        assert store._lakebase.pool.closed

    def test_checkpoint_write_and_read(self, cleanup_all_tables_project_branch):
        """Test CheckpointSaver: setup + put + get_tuple + pool cleanup."""
        thread_id = uuid.uuid4().hex

        with CheckpointSaver(project=get_project(), branch=get_branch()) as saver:
            saver.setup()

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            checkpoint = _make_checkpoint()
            saver.put(config, checkpoint, CheckpointMetadata(), {})

            result = saver.get_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]

        assert saver._lakebase.pool.closed

    @pytest.mark.asyncio
    async def test_async_checkpoint_write_and_read(self, cleanup_all_tables_project_branch):
        """Test AsyncCheckpointSaver: setup + put + get_tuple + pool cleanup."""
        thread_id = uuid.uuid4().hex

        async with AsyncCheckpointSaver(project=get_project(), branch=get_branch()) as saver:
            await saver.setup()

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            checkpoint = _make_checkpoint()
            await saver.aput(config, checkpoint, CheckpointMetadata(), {})

            result = await saver.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]

        assert saver._lakebase.pool.closed


# =============================================================================
# Autoscaling — Endpoint
# =============================================================================


@_skip_no_endpoint
class TestAutoscalingEndpoint:
    """Test all LangChain Lakebase wrappers with autoscaling endpoint mode."""

    def test_store_put_and_get(self, unique_namespace, cleanup_all_tables_endpoint):
        """Test DatabricksStore: endpoint params forwarded to LakebasePool."""
        store = DatabricksStore(autoscaling_endpoint=get_autoscaling_endpoint())
        store.setup()

        ns = unique_namespace
        store.put(ns, "key1", {"data": "endpoint hello"})

        item = store.get(ns, "key1")
        assert item is not None
        assert item.value == {"data": "endpoint hello"}
        assert item.key == "key1"

    @pytest.mark.asyncio
    async def test_async_store_put_and_get(self, unique_namespace, cleanup_all_tables_endpoint):
        """Test AsyncDatabricksStore: async endpoint pool open/close + put + get."""
        async with AsyncDatabricksStore(autoscaling_endpoint=get_autoscaling_endpoint()) as store:
            await store.setup()

            ns = unique_namespace
            await store.aput(ns, "async_key", {"data": "async endpoint"})

            item = await store.aget(ns, "async_key")
            assert item is not None
            assert item.value == {"data": "async endpoint"}

        assert store._lakebase.pool.closed

    def test_checkpoint_write_and_read(self, cleanup_all_tables_endpoint):
        """Test CheckpointSaver: setup + put + get_tuple + pool cleanup."""
        thread_id = uuid.uuid4().hex

        with CheckpointSaver(autoscaling_endpoint=get_autoscaling_endpoint()) as saver:
            saver.setup()

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            checkpoint = _make_checkpoint()
            saver.put(config, checkpoint, CheckpointMetadata(), {})

            result = saver.get_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]

        assert saver._lakebase.pool.closed

    @pytest.mark.asyncio
    async def test_async_checkpoint_write_and_read(self, cleanup_all_tables_endpoint):
        """Test AsyncCheckpointSaver: setup + put + get_tuple + pool cleanup."""
        thread_id = uuid.uuid4().hex

        async with AsyncCheckpointSaver(autoscaling_endpoint=get_autoscaling_endpoint()) as saver:
            await saver.setup()

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            checkpoint = _make_checkpoint()
            await saver.aput(config, checkpoint, CheckpointMetadata(), {})

            result = await saver.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == checkpoint["id"]

        assert saver._lakebase.pool.closed
