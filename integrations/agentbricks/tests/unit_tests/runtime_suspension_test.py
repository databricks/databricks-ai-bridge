"""Atomic pause/resolution and continuation behavior without a live human wait."""

import asyncio
from typing import cast

import pytest

from databricks_agentkit.runtime.durability.store import DurableRuntimeStore
from databricks_agentkit.runtime.execution import AttemptExecution
from databricks_agentkit.runtime.runtime import Runtime
from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from databricks_agentkit.runtime.types import (
    InvocationAwaitingResolutionError,
    InvocationStatus,
    InvocationSuspension,
    JsonObject,
)


@pytest.mark.asyncio
async def test_suspension_resumes_same_invocation_with_saved_state():
    store = InMemoryRuntimeStore()
    contexts = []

    async def execute(request, context):
        contexts.append(context)
        if context.continuation is None:
            return InvocationSuspension({"pending": ["call-1"], "resolutions": {}})
        return {"finished": context.continuation["resolutions"]["call-1"]}

    execution = AttemptExecution(execute, runtime_store=store)
    await store.accept("invocation", {"input": "hello"}, session_id="session")
    first = await store.claim("invocation")
    assert first is not None
    await execution.run(first)
    waiting = await store.get("invocation")
    assert waiting is not None
    assert waiting.continuation is not None
    assert waiting.status == InvocationStatus.AWAITING_RESOLUTION
    assert not waiting.is_terminal
    assert waiting.response is None
    resolved: JsonObject = {
        "pending": [],
        "resolutions": {"call-1": {"result": "done"}},
    }
    queued = await store.resolve("invocation", waiting.continuation, resolved, resume=True)
    assert queued is not None
    assert queued.status == InvocationStatus.QUEUED
    second = await store.claim("invocation")
    assert second is not None
    await execution.run(second)
    final = await store.get("invocation")
    assert final is not None
    assert final.status == InvocationStatus.COMPLETED
    assert final.response == {"finished": {"result": "done"}}
    assert final.continuation == resolved
    assert [(c.invocation_id, c.attempt) for c in contexts] == [
        ("invocation", 1),
        ("invocation", 2),
    ]


@pytest.mark.asyncio
async def test_only_one_resolution_can_update_a_waiting_snapshot():
    store = InMemoryRuntimeStore()
    await store.accept("invocation", None)
    claimed = await store.claim("invocation")
    assert claimed is not None
    original: JsonObject = {"pending": ["one", "two"]}
    assert await store.suspend("invocation", claimed.attempt, original)
    results = await asyncio.gather(
        store.resolve("invocation", original, {"pending": ["two"]}, resume=False),
        store.resolve("invocation", original, {"pending": ["one"]}, resume=False),
    )
    assert sum(result is not None for result in results) == 1
    waiting = await store.get("invocation")
    assert waiting is not None
    assert waiting.status == InvocationStatus.AWAITING_RESOLUTION


@pytest.mark.asyncio
async def test_active_resolution_is_rejected_and_waiting_blocks_later_session_work():
    store = InMemoryRuntimeStore()
    await store.accept("first", None, session_id="session")
    await store.accept("second", None, session_id="session")
    first = await store.claim("first")
    assert first is not None
    assert await store.resolve("first", {}, {"accepted": True}, resume=True) is None
    assert await store.suspend("first", first.attempt, {"pending": ["one"]})
    assert await store.claim("second") is None
    waiting = await store.get(session_id="session")
    assert waiting is not None
    assert waiting.invocation_id == "first"
    assert not await store.complete("first", first.attempt, "late result")
    assert not await store.suspend("first", first.attempt + 1, {})


@pytest.mark.asyncio
async def test_resolve_validates_and_detaches_json_inputs():
    store = InMemoryRuntimeStore()
    runtime = Runtime.local(lambda request, context: None, runtime_store=store)
    await runtime.start()
    try:
        await store.accept("invocation", {})
        claimed = await store.claim("invocation")
        assert claimed is not None
        await store.suspend("invocation", claimed.attempt, {"pending": ["one"]})
        with pytest.raises(TypeError, match="expected and continuation must be JSON objects"):
            await runtime.resolve("invocation", [], {}, resume=False)  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match="continuation must be JSON serializable"):
            await runtime.resolve(
                "invocation",
                {"pending": ["one"]},
                {"bad": object()},  # ty: ignore[invalid-argument-type]
                resume=False,
            )
        expected: JsonObject = {"pending": ["one"]}
        continuation: JsonObject = {"pending": ["two"]}
        resolved = await runtime.resolve("invocation", expected, continuation, resume=False)
        assert resolved is not None
        expected["pending"].append("changed")
        continuation["pending"].append("changed")
        assert resolved.continuation == {"pending": ["two"]}
        waiting = await store.get("invocation")
        assert waiting is not None
        assert waiting.continuation == {"pending": ["two"]}
        with pytest.raises(InvocationAwaitingResolutionError):
            await runtime.wait("invocation")
    finally:
        await runtime.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_before_error", [False, True])
async def test_ambiguous_suspend_error_does_not_fail_attempt(commit_before_error):
    class UncertainStore(InMemoryRuntimeStore):
        async def suspend(self, invocation_id, attempt, continuation):
            if commit_before_error:
                await super().suspend(invocation_id, attempt, continuation)
            raise ConnectionError("suspension acknowledgment was lost")

    store = UncertainStore()

    async def execute(request, context):
        return InvocationSuspension({"pending": ["one"]})

    execution = AttemptExecution(execute, runtime_store=store)
    await store.accept("invocation", {})
    claimed = await store.claim("invocation")
    assert claimed is not None
    await execution.run(claimed)

    state = await store.get("invocation")
    assert state is not None
    assert state.status == (
        InvocationStatus.AWAITING_RESOLUTION if commit_before_error else InvocationStatus.ACTIVE
    )
    assert [event.event["type"] for event in await store.events("invocation")] == (
        ["run.started", "run.awaiting_resolution"] if commit_before_error else ["run.started"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_resolution_during_prior_task_cleanup_starts_one_new_attempt(durable):
    suspended = asyncio.Event()
    finish_cleanup = asyncio.Event()
    finished = asyncio.Event()

    class PausingStore(InMemoryRuntimeStore):
        async def suspend(self, invocation_id, attempt, continuation):
            result = await super().suspend(invocation_id, attempt, continuation)
            suspended.set()
            await finish_cleanup.wait()
            return result

    class DurablePausingStore(PausingStore):
        async def queued_invocation_ids(self):
            return []

        async def stale_invocation_ids(self, stale_seconds):
            return []

        async def claim_recoverable(self, invocation_id, stale_seconds):
            return None

        async def heartbeat(self, invocation_id, attempt):
            state = await self.get(invocation_id)
            assert state is not None
            return state.status == InvocationStatus.ACTIVE and state.attempt == attempt

    store = DurablePausingStore() if durable else PausingStore()
    attempts = []

    async def execute(request, context):
        attempts.append(context.attempt)
        if context.continuation is None:
            return InvocationSuspension({"pending": ["one"]})
        finished.set()
        return context.continuation

    if durable:
        runtime = Runtime.durable(
            execute,
            runtime_store=cast(DurableRuntimeStore, store),
            scan_seconds=100,
        )
    else:
        runtime = Runtime.local(execute, runtime_store=store)
    await runtime.start()
    try:
        await runtime.submit("invocation", {}, session_id="session")
        await asyncio.wait_for(suspended.wait(), 2)
        waiting = await store.get("invocation")
        assert waiting is not None
        assert waiting.continuation is not None
        assert waiting.status == InvocationStatus.AWAITING_RESOLUTION
        await runtime.resolve("invocation", waiting.continuation, {"resolved": True}, resume=True)
        assert attempts == [1]
        finish_cleanup.set()
        await asyncio.wait_for(finished.wait(), 2)
        assert await runtime.wait("invocation", timeout=2) == {"resolved": True}
        assert attempts == [1, 2]
    finally:
        finish_cleanup.set()
        await runtime.stop()
