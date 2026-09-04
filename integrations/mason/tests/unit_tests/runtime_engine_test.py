"""Tests for DurableRuntime orchestration."""

import asyncio
import copy
from datetime import datetime, timedelta, timezone

import pytest

from databricks_mason.runtime.runtime import DurableRuntime
from databricks_mason.runtime.types import (
    DurableEvent,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableRequestConflictError,
)


class MemoryDurabilityStore:
    def __init__(self) -> None:
        self.states: dict[str, DurableExecution] = {}
        self.initialized = False
        self.closed = False
        self.heartbeats: list[tuple[str, int]] = []
        self.persisted_events: list[DurableEvent] = []

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def accept(self, execution_id: str, request: dict) -> DurableExecution:
        existing = self.states.get(execution_id)
        if existing is not None:
            if existing.request != request:
                raise DurableRequestConflictError(execution_id)
            return existing
        state = DurableExecution(
            execution_id=execution_id,
            status=DurableExecutionStatus.QUEUED,
            attempt=0,
            heartbeat_at=None,
            request=copy.deepcopy(request),
            response=None,
        )
        self.states[execution_id] = state
        return state

    async def get(self, execution_id: str) -> DurableExecution | None:
        return self.states.get(execution_id)

    async def recoverable_execution_ids(self, stale_seconds: float) -> list[str]:
        return [
            execution_id
            for execution_id, state in self.states.items()
            if state.status == DurableExecutionStatus.QUEUED
            or (
                state.status == DurableExecutionStatus.ACTIVE
                and (
                    state.heartbeat_at is None
                    or datetime.now(timezone.utc) - state.heartbeat_at
                    >= timedelta(seconds=stale_seconds)
                )
            )
        ]

    async def claim(
        self,
        execution_id: str,
        stale_seconds: float,
    ) -> DurableExecution | None:
        state = self.states[execution_id]
        recoverable = state.status == DurableExecutionStatus.QUEUED or (
            state.status == DurableExecutionStatus.ACTIVE
            and (
                state.heartbeat_at is None
                or datetime.now(timezone.utc) - state.heartbeat_at
                >= timedelta(seconds=stale_seconds)
            )
        )
        if not recoverable:
            return None
        state = DurableExecution(
            execution_id=state.execution_id,
            status=DurableExecutionStatus.ACTIVE,
            attempt=state.attempt + 1,
            heartbeat_at=datetime.now(timezone.utc),
            request=state.request,
            response=None,
        )
        self.states[execution_id] = state
        return state

    async def heartbeat(self, execution_id: str, attempt: int) -> bool:
        state = self.states[execution_id]
        if state.status != DurableExecutionStatus.ACTIVE or state.attempt != attempt:
            return False
        self.heartbeats.append((execution_id, attempt))
        self.states[execution_id] = DurableExecution(
            execution_id=state.execution_id,
            status=state.status,
            attempt=state.attempt,
            heartbeat_at=datetime.now(timezone.utc),
            request=state.request,
            response=state.response,
        )
        return True

    async def complete(self, execution_id: str, attempt: int, response: dict) -> bool:
        state = self.states[execution_id]
        if state.status != DurableExecutionStatus.ACTIVE or state.attempt != attempt:
            return False
        self.states[execution_id] = DurableExecution(
            execution_id=state.execution_id,
            status=DurableExecutionStatus.COMPLETED,
            attempt=state.attempt,
            heartbeat_at=state.heartbeat_at,
            request=state.request,
            response=copy.deepcopy(response),
        )
        return True

    async def fail(self, execution_id: str, attempt: int) -> bool:
        state = self.states[execution_id]
        if state.status != DurableExecutionStatus.ACTIVE or state.attempt != attempt:
            return False
        self.states[execution_id] = DurableExecution(
            execution_id=state.execution_id,
            status=DurableExecutionStatus.FAILED,
            attempt=state.attempt,
            heartbeat_at=state.heartbeat_at,
            request=state.request,
            response=None,
        )
        return True

    async def append_event(
        self,
        execution_id: str,
        attempt: int,
        event: dict,
    ) -> int | None:
        state = self.states[execution_id]
        if state.status != DurableExecutionStatus.ACTIVE or state.attempt != attempt:
            return None
        persisted = DurableEvent(
            sequence_number=len(self.persisted_events) + 1,
            execution_id=execution_id,
            attempt=attempt,
            event=copy.deepcopy(event),
        )
        self.persisted_events.append(persisted)
        return persisted.sequence_number

    async def events(
        self,
        execution_id: str,
        after_sequence: int | None = None,
    ) -> list[DurableEvent]:
        return [
            copy.deepcopy(event)
            for event in self.persisted_events
            if event.execution_id == execution_id
            and (after_sequence is None or event.sequence_number > after_sequence)
        ]


def make_runtime(executor, store=None, **kwargs):
    return DurableRuntime(
        executor,
        durability_store=store or MemoryDurabilityStore(),
        heartbeat_seconds=0.01,
        stale_seconds=0.05,
        scan_seconds=0.01,
        poll_seconds=0.005,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_invoke_persists_request_and_response():
    calls = []

    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        calls.append((request, context))
        return {"output": request["input"]}

    store = MemoryDurabilityStore()
    runtime = make_runtime(execute, store)
    await runtime.start()
    try:
        response = await runtime.invoke("session-1", {"input": "hello"})
        state = await runtime.get("session-1")
    finally:
        await runtime.stop()

    assert response == {"output": "hello"}
    assert state is not None
    assert state.request == {"input": "hello"}
    assert state.response == {"output": "hello"}
    assert calls[0][1].attempt == 1
    assert calls[0][1].is_recovery is False


@pytest.mark.asyncio
async def test_completed_request_returns_cached_response_without_reexecution():
    call_count = 0

    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        nonlocal call_count
        call_count += 1
        return {"output": "done"}

    runtime = make_runtime(execute)
    await runtime.start()
    try:
        first = await runtime.invoke("session-1", {"input": "hello"})
        second = await runtime.invoke("session-1", {"input": "hello"})
    finally:
        await runtime.stop()

    assert first == second == {"output": "done"}
    assert call_count == 1


@pytest.mark.asyncio
async def test_same_id_with_different_request_is_rejected():
    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        return {"output": "done"}

    runtime = make_runtime(execute)
    await runtime.start()
    try:
        await runtime.invoke("session-1", {"input": "first"})
        with pytest.raises(DurableRequestConflictError):
            await runtime.submit("session-1", {"input": "second"})
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_stale_attempt_reuses_request_and_marks_recovery():
    contexts = []

    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        contexts.append((request, context))
        return {"output": "recovered"}

    store = MemoryDurabilityStore()
    store.states["session-1"] = DurableExecution(
        execution_id="session-1",
        status=DurableExecutionStatus.ACTIVE,
        attempt=1,
        heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        request={"input": "original"},
        response=None,
    )
    runtime = make_runtime(execute, store)
    await runtime.start()
    try:
        response = await runtime.wait("session-1")
    finally:
        await runtime.stop()

    assert response == {"output": "recovered"}
    assert contexts == [
        (
            {"input": "original"},
            DurableExecutionContext(execution_id="session-1", attempt=2),
        )
    ]
    assert contexts[0][1].is_recovery is True


@pytest.mark.asyncio
async def test_executor_failure_is_persisted_as_terminal_state():
    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        raise RuntimeError("boom")

    runtime = make_runtime(execute)
    await runtime.start()
    try:
        with pytest.raises(DurableExecutionFailedError):
            await runtime.invoke("session-1", {"input": "hello"})
        state = await runtime.get("session-1")
    finally:
        await runtime.stop()

    assert state is not None
    assert state.status == DurableExecutionStatus.FAILED
    assert state.response is None


@pytest.mark.asyncio
async def test_submit_returns_before_background_execution_finishes():
    release = asyncio.Event()

    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        await release.wait()
        return {"output": "done"}

    runtime = make_runtime(execute)
    await runtime.start()
    try:
        state = await runtime.submit("session-1", {"input": "hello"})
        assert state.status == DurableExecutionStatus.QUEUED
        release.set()
        assert await runtime.wait("session-1") == {"output": "done"}
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_wait_observes_response_completed_by_another_process():
    executor_called = False

    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        nonlocal executor_called
        executor_called = True
        return {}

    store = MemoryDurabilityStore()
    store.states["session-1"] = DurableExecution(
        execution_id="session-1",
        status=DurableExecutionStatus.ACTIVE,
        attempt=1,
        heartbeat_at=datetime.now(timezone.utc),
        request={"input": "hello"},
        response=None,
    )
    runtime = make_runtime(execute, store)
    await runtime.start()

    async def complete_elsewhere():
        await asyncio.sleep(0.02)
        await store.complete("session-1", 1, {"output": "remote"})

    completion = asyncio.create_task(complete_elsewhere())
    try:
        assert await runtime.wait("session-1") == {"output": "remote"}
    finally:
        await completion
        await runtime.stop()

    assert executor_called is False


@pytest.mark.asyncio
async def test_blocking_timeout_does_not_cancel_execution():
    release = asyncio.Event()

    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        await release.wait()
        return {"output": "done"}

    runtime = make_runtime(execute)
    await runtime.start()
    try:
        with pytest.raises(TimeoutError):
            await runtime.invoke("session-1", {"input": "hello"}, timeout=0.02)
        state = await runtime.get("session-1")
        assert state is not None
        assert state.status == DurableExecutionStatus.ACTIVE
        release.set()
        assert await runtime.wait("session-1") == {"output": "done"}
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_executor_can_persist_replayable_events():
    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        sequence_number = await context.emit({"type": "progress", "step": 1})
        return {"last_sequence_number": sequence_number}

    runtime = make_runtime(execute)
    await runtime.start()
    try:
        assert await runtime.invoke("session-1", {"input": "hello"}) == {"last_sequence_number": 1}
        events = await runtime.events("session-1")
        assert [(event.sequence_number, event.event) for event in events] == [
            (1, {"type": "progress", "step": 1})
        ]
        assert await runtime.events("session-1", after_sequence=1) == []
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_request_must_be_json_object():
    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        return {}

    runtime = make_runtime(execute)
    await runtime.start()
    try:
        with pytest.raises(TypeError, match="request must be a JSON object"):
            await runtime.submit("session-1", ["not", "an", "object"])
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_subclass_can_own_execution_wiring():
    class Runtime(DurableRuntime):
        async def execute(
            self,
            request: dict,
            context: DurableExecutionContext,
        ) -> dict:
            return {"attempt": context.attempt, "input": request["input"]}

    runtime = Runtime(
        durability_store=MemoryDurabilityStore(),
        heartbeat_seconds=0.01,
        stale_seconds=0.05,
        scan_seconds=0.01,
        poll_seconds=0.005,
    )
    await runtime.start()
    try:
        assert await runtime.invoke("session-1", {"input": "hello"}) == {
            "attempt": 1,
            "input": "hello",
        }
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_start_and_stop_manage_store_lifecycle():
    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        return {}

    store = MemoryDurabilityStore()
    runtime = make_runtime(execute, store)

    await runtime.start()
    assert store.initialized is True
    await runtime.stop()
    assert store.closed is True


@pytest.mark.asyncio
async def test_runtime_requires_start_before_use():
    async def execute(request: dict, context: DurableExecutionContext) -> dict:
        return {}

    runtime = make_runtime(execute)
    with pytest.raises(RuntimeError, match=r"start\(\)"):
        await runtime.submit("session-1", {})
