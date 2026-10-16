"""Real PostgreSQL claims, FIFO promotion, fencing, and upgrade coverage.

Set AGENTKIT_TEST_POSTGRES_URL to a disposable database using a
postgresql+psycopg:// URL. Each test creates and removes its own schema.
"""

import asyncio

import pytest

pytestmark = pytest.mark.asyncio


async def test_admission_reserves_owner_and_retry_preserves_lease(postgres_runtime_store):
    from sqlalchemy import text

    from databricks_agentkit.runtime.types import InvocationStatus

    owner = await postgres_runtime_store.create("blue")
    other = await postgres_runtime_store.create("blue")
    accepted = await owner.accept("invocation", {"input": "hello"})
    assert accepted.status == InvocationStatus.QUEUED
    assert accepted.worker_id == owner.worker_id
    assert accepted.recovery_affinity == "blue"
    await postgres_runtime_store.age("invocation")
    query = text(f"SELECT heartbeat_at FROM {postgres_runtime_store.table}")
    async with postgres_runtime_store.engine.connect() as connection:
        original_heartbeat = (await connection.execute(query)).scalar_one()

    assert await other.accept("invocation", {"input": "hello"}) == accepted
    async with postgres_runtime_store.engine.connect() as connection:
        assert (await connection.execute(query)).scalar_one() == original_heartbeat

    # Exercise contention against the SQL predicate, bypassing scanner filtering.
    claims = await asyncio.gather(other.claim("invocation"), owner.claim("invocation"))
    assert claims[0] is None
    assert claims[1] is not None
    assert claims[1].worker_id == owner.worker_id
    assert claims[1].attempt == 1
    assert await other.claim("invocation") is None
    assert [event.event for event in await owner.events("invocation")] == [{"type": "run.started"}]


@pytest.mark.parametrize("affinity", [None, "blue"])
async def test_stale_queued_transfer_requires_matching_group(postgres_runtime_store, affinity):
    owner = await postgres_runtime_store.create(affinity)
    replacement = await postgres_runtime_store.create(affinity)
    wrong_group = await postgres_runtime_store.create("other" if affinity is None else None)
    await owner.accept("abandoned", {})

    assert await replacement.claim_recoverable("abandoned", 10) is None
    await postgres_runtime_store.age("abandoned")
    assert "abandoned" in await replacement.stale_invocation_ids(10)
    assert "abandoned" not in await wrong_group.stale_invocation_ids(10)
    assert await wrong_group.claim_recoverable("abandoned", 10) is None
    assert await replacement.claim("abandoned") is None

    recovered = await replacement.claim_recoverable("abandoned", 10)
    assert recovered is not None
    assert recovered.worker_id == replacement.worker_id
    assert recovered.recovery_affinity == affinity
    assert recovered.attempt == 2  # OBO must recognize takeover before first execution as recovery.
    assert await owner.claim("abandoned") is None
    assert await replacement.complete("abandoned", recovered.attempt, "recovered")


@pytest.mark.parametrize("terminal", ["complete", "fail"])
async def test_fifo_promotion_refreshes_waiting_owners_grace(postgres_runtime_store, terminal):
    owner = await postgres_runtime_store.create()
    next_owner = await postgres_runtime_store.create()
    replacement = await postgres_runtime_store.create()
    await owner.accept("first", {}, session_id="session")
    await next_owner.accept("second", {}, session_id="session")
    await replacement.accept("third", {}, session_id="session")
    active = await owner.claim("first")
    assert active is not None
    await postgres_runtime_store.age("second")
    await postgres_runtime_store.age("third")
    assert await next_owner.claim("second") is None
    assert "second" not in await replacement.stale_invocation_ids(10)
    assert await replacement.claim_recoverable("second", 10) is None

    assert await getattr(owner, terminal)("first", active.attempt, None)
    assert "second" not in await replacement.stale_invocation_ids(10)
    assert await replacement.claim_recoverable("second", 10) is None
    assert await owner.claim("second") is None
    assert await replacement.claim("third") is None
    assert await replacement.claim_recoverable("third", 10) is None
    second = await next_owner.claim("second")
    assert second is not None
    assert second.attempt == 1
    assert second.session_sequence_number == 2


async def test_active_takeover_fences_old_writes_and_serializes_recoverers(postgres_runtime_store):
    owner = await postgres_runtime_store.create("blue")
    replacements = [await postgres_runtime_store.create("blue") for _ in range(2)]
    wrong_group = await postgres_runtime_store.create("green")
    await owner.accept("active", {})
    first = await owner.claim("active")
    assert first is not None
    await postgres_runtime_store.age("active")
    assert await wrong_group.claim_recoverable("active", 10) is None
    claims = await asyncio.gather(
        *(store.claim_recoverable("active", 10) for store in replacements)
    )
    winners = [
        (store, claim)
        for store, claim in zip(replacements, claims, strict=True)
        if claim is not None
    ]
    assert len(winners) == 1
    winner, recovered = winners[0]
    assert recovered.attempt == first.attempt + 1
    assert await owner.heartbeat("active", first.attempt) is False
    assert await owner.complete("active", first.attempt, "old") is False
    assert await owner.fail("active", first.attempt) is False
    assert await owner.append_event("active", first.attempt, {"type": "late"}) is None
    assert await winner.append_event("active", recovered.attempt, {"type": "resumed"})
    assert await winner.complete("active", recovered.attempt, "new")
    assert (await owner.get("active")).response == "new"
    assert [(event.attempt, event.event["type"]) for event in await owner.events("active")] == [
        (1, "run.started"),
        (2, "run.started"),
        (2, "resumed"),
        (2, "run.completed"),
    ]


async def test_initialization_upgrades_legacy_schema_without_reassigning_rows(
    postgres_runtime_store,
):
    from sqlalchemy import text

    schema = postgres_runtime_store.schema
    async with postgres_runtime_store.engine.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA {schema}"))
        await connection.execute(
            text(
                f"CREATE TABLE {schema}.invocations ("
                "invocation_id TEXT PRIMARY KEY, status TEXT NOT NULL, "
                "attempt INTEGER NOT NULL DEFAULT 0, heartbeat_at TIMESTAMPTZ, "
                "request JSONB NOT NULL, response JSONB)"
            )
        )
        await connection.execute(
            text(
                f"INSERT INTO {schema}.invocations (invocation_id, status, request) "
                "VALUES ('legacy', 'QUEUED', '{}')"
            )
        )
    default = await postgres_runtime_store.create()
    named = await postgres_runtime_store.create("named")
    await default.initialize()  # The migration is idempotent.
    legacy = await default.get("legacy")
    assert legacy is not None
    assert legacy.worker_id is None
    assert legacy.recovery_affinity is None
    assert await named.claim("legacy") is None
    assert await named.claim_recoverable("legacy", 10) is None
    claimed = await default.claim("legacy")
    assert claimed is not None
    assert claimed.worker_id == default.worker_id
    assert claimed.attempt == 1
