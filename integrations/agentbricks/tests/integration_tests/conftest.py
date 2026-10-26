"""Isolated PostgreSQL schemas for opt-in runtime integration tests."""

import os
from uuid import uuid4

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def postgres_runtime_store():
    url = os.environ.get("AGENTKIT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set AGENTKIT_TEST_POSTGRES_URL to a disposable PostgreSQL database")

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from databricks_agentkit.runtime.durability.lakebase_runtime_store import (
        LakebaseDurableRuntimeStore,
    )

    schema = f"runtime_affinity_{uuid4().hex}"
    engine = create_async_engine(url)

    class PostgresLakebase:
        def __init__(self):
            self.engine = create_async_engine(url)

        async def create_schema(self):
            async with self.engine.begin() as connection:
                await connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

    class StoreFactory:
        def __init__(self):
            self.engine = engine
            self.schema = schema
            self.table = f"{schema}.invocations"
            self.stores = []

        async def create(self, recovery_affinity=None):
            store = LakebaseDurableRuntimeStore(
                lakebase=PostgresLakebase(), schema=schema, recovery_affinity=recovery_affinity
            )
            self.stores.append(store)
            await store.initialize()
            return store

        async def age(self, invocation_id):
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        f"UPDATE {self.table} "
                        "SET heartbeat_at=clock_timestamp() - INTERVAL '1 hour' "
                        "WHERE invocation_id=:invocation_id"
                    ),
                    {"invocation_id": invocation_id},
                )

    factory = StoreFactory()
    try:
        yield factory
    finally:
        for store in factory.stores:
            await store.close()
        async with engine.begin() as connection:
            await connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        await engine.dispose()
