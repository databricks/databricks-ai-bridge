"""Session queueing tests for the process-local Runtime Store."""

import pytest

from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from databricks_agentkit.runtime.types import InvocationConflictError, InvocationStatus


@pytest.mark.asyncio
async def test_accept_assigns_queue_order_within_each_session() -> None:
    store = InMemoryRuntimeStore()

    first = await store.accept("invocation-1", {"message": "one"}, session_id="session-a")
    other = await store.accept("invocation-2", {"message": "other"}, session_id="session-b")
    second = await store.accept("invocation-3", {"message": "two"}, session_id="session-a")
    sessionless = await store.accept("invocation-4", {"message": "standalone"})

    assert (first.session_id, first.queue_order) == ("session-a", 1)
    assert (second.session_id, second.queue_order) == ("session-a", 2)
    assert (other.session_id, other.queue_order) == ("session-b", 1)
    assert (sessionless.session_id, sessionless.queue_order) == (None, None)


@pytest.mark.asyncio
async def test_accept_is_idempotent_only_for_the_same_request_and_session() -> None:
    store = InMemoryRuntimeStore()
    accepted = await store.accept(
        "invocation-1",
        {"message": "hello"},
        session_id="session-a",
    )

    replay = await store.accept(
        "invocation-1",
        {"message": "hello"},
        session_id="session-a",
    )

    assert replay == accepted
    with pytest.raises(InvocationConflictError):
        await store.accept(
            "invocation-1",
            {"message": "different"},
            session_id="session-a",
        )
    with pytest.raises(InvocationConflictError):
        await store.accept(
            "invocation-1",
            {"message": "hello"},
            session_id="session-b",
        )


@pytest.mark.asyncio
async def test_session_claims_are_fifo_and_session_reads_return_current_work() -> None:
    store = InMemoryRuntimeStore()
    await store.accept("invocation-1", {"message": "one"}, session_id="session-a")
    await store.accept("invocation-2", {"message": "two"}, session_id="session-a")

    queued = await store.get(session_id="session-a")
    assert queued is not None
    assert (queued.invocation_id, queued.status) == ("invocation-1", InvocationStatus.QUEUED)
    assert await store.claim("invocation-2") is None

    active = await store.claim("invocation-1")
    assert active is not None
    assert (active.session_id, active.queue_order) == ("session-a", 1)
    assert await store.get(session_id="session-a") == active
    assert await store.claim("invocation-2") is None

    assert await store.complete("invocation-1", active.attempt, {"answer": "one"})
    completed = await store.get("invocation-1")
    assert completed is not None
    assert (completed.session_id, completed.queue_order) == ("session-a", 1)
    next_queued = await store.get(session_id="session-a")
    assert next_queued is not None
    assert (next_queued.invocation_id, next_queued.queue_order) == ("invocation-2", 2)

    next_active = await store.claim("invocation-2")
    assert next_active is not None
    assert await store.fail("invocation-2", next_active.attempt)
    failed = await store.get("invocation-2")
    assert failed is not None
    assert (failed.session_id, failed.queue_order) == ("session-a", 2)
    assert await store.get(session_id="session-a") is None


@pytest.mark.asyncio
async def test_different_sessions_can_be_active_independently() -> None:
    store = InMemoryRuntimeStore()
    await store.accept("a-1", {}, session_id="session-a")
    await store.accept("a-2", {}, session_id="session-a")
    await store.accept("b-1", {}, session_id="session-b")

    active_a = await store.claim("a-1")
    active_b = await store.claim("b-1")

    assert active_a is not None
    assert active_b is not None
    assert await store.claim("a-2") is None
    assert await store.complete("a-1", active_a.attempt, {})
    assert await store.claim("a-2") is not None


@pytest.mark.asyncio
async def test_session_events_span_invocations_in_global_sequence_order() -> None:
    store = InMemoryRuntimeStore()
    await store.accept("a-1", {}, session_id="session-a")
    await store.accept("a-2", {}, session_id="session-a")
    await store.accept("b-1", {}, session_id="session-b")

    active_a1 = await store.claim("a-1")
    assert active_a1 is not None
    assert await store.append_event("a-1", active_a1.attempt, {"type": "a-1.delta"}) == 2
    active_b1 = await store.claim("b-1")
    assert active_b1 is not None
    assert await store.append_event("b-1", active_b1.attempt, {"type": "b-1.delta"}) == 4
    assert await store.complete("a-1", active_a1.attempt, {})
    active_a2 = await store.claim("a-2")
    assert active_a2 is not None
    assert await store.append_event("a-2", active_a2.attempt, {"type": "a-2.delta"}) == 7
    assert await store.complete("a-2", active_a2.attempt, {})

    events = await store.events(session_id="session-a")

    assert [event.sequence_number for event in events] == [1, 2, 5, 6, 7, 8]
    assert [event.invocation_id for event in events] == [
        "a-1",
        "a-1",
        "a-1",
        "a-2",
        "a-2",
        "a-2",
    ]
    assert [
        event.sequence_number
        for event in await store.events(session_id="session-a", after_sequence=5)
    ] == [6, 7, 8]


@pytest.mark.asyncio
async def test_session_reads_require_exactly_one_selector() -> None:
    store = InMemoryRuntimeStore()

    with pytest.raises(ValueError, match="exactly one"):
        await store.get()
    with pytest.raises(ValueError, match="exactly one"):
        await store.get("invocation-1", session_id="session-a")
    with pytest.raises(ValueError, match="exactly one"):
        await store.events()
    with pytest.raises(ValueError, match="exactly one"):
        await store.events("invocation-1", session_id="session-a")


@pytest.mark.asyncio
async def test_empty_session_id_is_rejected() -> None:
    store = InMemoryRuntimeStore()

    with pytest.raises(ValueError, match="session_id must not be empty"):
        await store.accept("invocation-1", {}, session_id="")
    with pytest.raises(ValueError, match="session_id must not be empty"):
        await store.get(session_id="")
