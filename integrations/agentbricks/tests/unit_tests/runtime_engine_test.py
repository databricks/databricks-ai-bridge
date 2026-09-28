"""Tests for Runtime orchestration and local/durable executors."""

import asyncio
import copy
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

import databricks_agentkit.runtime.runtime as runtime_module
from databricks_agentkit.runtime.durability.execution import DurableInvocationExecutor
from databricks_agentkit.runtime.execution import LocalInvocationExecutor
from databricks_agentkit.runtime.runtime import Runtime
from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from databricks_agentkit.runtime.types import (
    Invocation,
    InvocationAttemptContext,
    InvocationConflictError,
    InvocationEvent,
    InvocationFailedError,
    InvocationStatus,
    JsonObject,
    JsonValue,
)


class MemoryDurableRuntimeStore:
    """Durable-store fake that keeps heartbeat leases separate from invocations."""

    def __init__(self) -> None:
        self.states: dict[str, Invocation] = {}
        self.initialized = False
        self.closed = False
        self.heartbeats: list[tuple[str, int]] = []
        self.persisted_events: list[InvocationEvent] = []
        self._heartbeat_at: dict[str, datetime] = {}

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def accept(
        self,
        invocation_id: str,
        request: JsonValue,
        session_id: str | None = None,
    ) -> Invocation:
        existing = self.states.get(invocation_id)
        if existing is not None:
            if existing.request != request or existing.session_id != session_id:
                raise InvocationConflictError(invocation_id)
            return copy.deepcopy(existing)
        queue_order = None
        if session_id is not None:
            queue_order = (
                max(
                    (
                        state.queue_order or 0
                        for state in self.states.values()
                        if state.session_id == session_id
                    ),
                    default=0,
                )
                + 1
            )
        state = Invocation(
            invocation_id=invocation_id,
            status=InvocationStatus.QUEUED,
            attempt=0,
            request=copy.deepcopy(request),
            response=None,
            session_id=session_id,
            queue_order=queue_order,
        )
        self.states[invocation_id] = state
        return copy.deepcopy(state)

    async def get(
        self,
        invocation_id: str | None = None,
        session_id: str | None = None,
    ) -> Invocation | None:
        if (invocation_id is None) == (session_id is None):
            raise ValueError("exactly one of invocation_id or session_id must be provided")
        if invocation_id is not None:
            state = self.states.get(invocation_id)
        else:
            candidates = [
                state
                for state in self.states.values()
                if state.session_id == session_id
                and state.status in {InvocationStatus.ACTIVE, InvocationStatus.QUEUED}
            ]
            state = min(
                candidates,
                key=lambda candidate: (
                    0 if candidate.status == InvocationStatus.ACTIVE else 1,
                    candidate.queue_order or 0,
                ),
                default=None,
            )
        return copy.deepcopy(state) if state is not None else None

    async def queued_invocation_ids(self) -> list[str]:
        runnable = []
        for invocation_id, state in self.states.items():
            if state.status != InvocationStatus.QUEUED:
                continue
            if state.session_id is None:
                runnable.append(invocation_id)
                continue
            session_states = [
                candidate
                for candidate in self.states.values()
                if candidate.session_id == state.session_id
            ]
            if any(candidate.status == InvocationStatus.ACTIVE for candidate in session_states):
                continue
            head = min(
                (
                    candidate
                    for candidate in session_states
                    if candidate.status == InvocationStatus.QUEUED
                ),
                key=lambda candidate: candidate.queue_order or 0,
            )
            if head.invocation_id == invocation_id:
                runnable.append(invocation_id)
        return runnable

    async def stale_invocation_ids(self, stale_seconds: float) -> list[str]:
        return [
            invocation_id
            for invocation_id, state in self.states.items()
            if self._is_stale(state, stale_seconds)
        ]

    async def claim(self, invocation_id: str) -> Invocation | None:
        state = self.states.get(invocation_id)
        if state is None or state.status != InvocationStatus.QUEUED:
            return None
        if state.session_id is not None:
            session_states = [
                candidate
                for candidate in self.states.values()
                if candidate.session_id == state.session_id
            ]
            if any(candidate.status == InvocationStatus.ACTIVE for candidate in session_states):
                return None
            head = min(
                (
                    candidate
                    for candidate in session_states
                    if candidate.status == InvocationStatus.QUEUED
                ),
                key=lambda candidate: candidate.queue_order or 0,
            )
            if head.invocation_id != invocation_id:
                return None
        return self._claim(state)

    async def claim_recoverable(
        self,
        invocation_id: str,
        stale_seconds: float,
    ) -> Invocation | None:
        state = self.states.get(invocation_id)
        if state is None or not self._is_stale(state, stale_seconds):
            return None
        return self._claim(state)

    async def heartbeat(self, invocation_id: str, attempt: int) -> bool:
        state = self.states.get(invocation_id)
        if state is None or state.status != InvocationStatus.ACTIVE or state.attempt != attempt:
            return False
        self.heartbeats.append((invocation_id, attempt))
        self._heartbeat_at[invocation_id] = datetime.now(timezone.utc)
        return True

    async def complete(self, invocation_id: str, attempt: int, response: JsonValue) -> bool:
        state = self.states.get(invocation_id)
        if state is None or not self._owns_attempt(state, attempt):
            return False
        self.states[invocation_id] = Invocation(
            invocation_id=state.invocation_id,
            status=InvocationStatus.COMPLETED,
            attempt=state.attempt,
            request=state.request,
            response=copy.deepcopy(response),
            session_id=state.session_id,
            queue_order=state.queue_order,
        )
        self._append_event(invocation_id, attempt, {"type": "run.completed"})
        return True

    async def fail(self, invocation_id: str, attempt: int) -> bool:
        state = self.states.get(invocation_id)
        if state is None or not self._owns_attempt(state, attempt):
            return False
        self.states[invocation_id] = Invocation(
            invocation_id=state.invocation_id,
            status=InvocationStatus.FAILED,
            attempt=state.attempt,
            request=state.request,
            response=None,
            session_id=state.session_id,
            queue_order=state.queue_order,
        )
        self._append_event(invocation_id, attempt, {"type": "run.failed"})
        return True

    async def append_event(
        self,
        invocation_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None:
        state = self.states.get(invocation_id)
        if state is None or not self._owns_attempt(state, attempt):
            return None
        return self._append_event(invocation_id, attempt, event).sequence_number

    async def events(
        self,
        invocation_id: str | None = None,
        after_sequence: int | None = None,
        session_id: str | None = None,
    ) -> list[InvocationEvent]:
        if (invocation_id is None) == (session_id is None):
            raise ValueError("exactly one of invocation_id or session_id must be provided")
        return [
            copy.deepcopy(event)
            for event in self.persisted_events
            if (
                event.invocation_id == invocation_id
                if invocation_id is not None
                else self.states[event.invocation_id].session_id == session_id
            )
            and (after_sequence is None or event.sequence_number > after_sequence)
        ]

    def seed_active(
        self,
        invocation_id: str,
        request: JsonValue,
        *,
        attempt: int = 1,
        heartbeat_at: datetime | None = None,
        session_id: str | None = None,
        queue_order: int | None = None,
    ) -> None:
        self.states[invocation_id] = Invocation(
            invocation_id=invocation_id,
            status=InvocationStatus.ACTIVE,
            attempt=attempt,
            request=copy.deepcopy(request),
            response=None,
            session_id=session_id,
            queue_order=queue_order,
        )
        self._heartbeat_at[invocation_id] = heartbeat_at or datetime.now(timezone.utc)

    def _claim(self, state: Invocation) -> Invocation:
        claimed = Invocation(
            invocation_id=state.invocation_id,
            status=InvocationStatus.ACTIVE,
            attempt=state.attempt + 1,
            request=copy.deepcopy(state.request),
            response=None,
            session_id=state.session_id,
            queue_order=state.queue_order,
        )
        self.states[claimed.invocation_id] = claimed
        self._heartbeat_at[claimed.invocation_id] = datetime.now(timezone.utc)
        self._append_event(claimed.invocation_id, claimed.attempt, {"type": "run.started"})
        return copy.deepcopy(claimed)

    def _is_stale(self, state: Invocation, stale_seconds: float) -> bool:
        if state.status != InvocationStatus.ACTIVE:
            return False
        heartbeat_at = self._heartbeat_at.get(state.invocation_id)
        return heartbeat_at is None or datetime.now(timezone.utc) - heartbeat_at >= timedelta(
            seconds=stale_seconds
        )

    @staticmethod
    def _owns_attempt(state: Invocation, attempt: int) -> bool:
        return state.status == InvocationStatus.ACTIVE and state.attempt == attempt

    def _append_event(
        self,
        invocation_id: str,
        attempt: int,
        event: JsonObject,
    ) -> InvocationEvent:
        persisted = InvocationEvent(
            sequence_number=len(self.persisted_events) + 1,
            invocation_id=invocation_id,
            attempt=attempt,
            event=copy.deepcopy(event),
        )
        self.persisted_events.append(persisted)
        return persisted


class TrackingInMemoryRuntimeStore(InMemoryRuntimeStore):
    def __init__(self) -> None:
        super().__init__()
        self.initialized = False
        self.closed = False
        self.heartbeat_calls = 0

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def heartbeat(self, invocation_id: str, attempt: int) -> bool:
        self.heartbeat_calls += 1
        return False


def make_local_runtime(execute_fn, store=None) -> Runtime:
    return Runtime.local(
        execute_fn,
        runtime_store=store,
        poll_seconds=0.005,
    )


def make_durable_runtime(
    execute_fn,
    store=None,
    *,
    recover=False,
    scan_seconds=0.01,
) -> Runtime:
    runtime_store = MemoryDurableRuntimeStore() if store is None else store
    return Runtime.durable(
        execute_fn,
        runtime_store=runtime_store,
        recovery_enabled=lambda: recover,
        heartbeat_seconds=0.01,
        stale_seconds=0.05,
        scan_seconds=scan_seconds,
        poll_seconds=0.005,
    )


def test_runtime_factories_select_execution_capabilities() -> None:
    async def execute(request: JsonValue, context: InvocationAttemptContext) -> JsonValue:
        return request

    local = Runtime.from_store(execute, runtime_store=InMemoryRuntimeStore())
    durable = Runtime.from_store(execute, runtime_store=MemoryDurableRuntimeStore())

    assert isinstance(local.executor, LocalInvocationExecutor)
    assert local.is_durable is False
    assert isinstance(durable.executor, DurableInvocationExecutor)
    assert durable.is_durable is True


def test_runtime_from_environment_resolves_store_before_factory_selection(monkeypatch) -> None:
    async def execute(request: JsonValue, context: InvocationAttemptContext) -> JsonValue:
        return request

    store = InMemoryRuntimeStore()
    resolve = lambda: store
    monkeypatch.setattr(runtime_module, "runtime_store_from_environment", resolve)

    runtime = Runtime.from_environment(execute)

    assert runtime.runtime_store is store
    assert isinstance(runtime.executor, LocalInvocationExecutor)


def test_local_factory_rejects_durable_store() -> None:
    async def execute(request: JsonValue, context: InvocationAttemptContext) -> JsonValue:
        return request

    with pytest.raises(TypeError, match="non-durable"):
        Runtime.local(execute, runtime_store=MemoryDurableRuntimeStore())


def test_durable_factory_rejects_local_store() -> None:
    async def execute(request: JsonValue, context: InvocationAttemptContext) -> JsonValue:
        return request

    with pytest.raises(TypeError, match="DurableRuntimeStore"):
        Runtime.durable(execute, runtime_store=cast(Any, InMemoryRuntimeStore()))


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", [None, "conversation-1"])
async def test_invoke_persists_request_and_response(session_id: str | None) -> None:
    calls = []

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        calls.append((request, context))
        return {"output": request["input"]}

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        response = await runtime.invoke("session-1", {"input": "hello"}, session_id=session_id)
        state = await runtime.get_invocation("session-1")
    finally:
        await runtime.stop()

    assert response == {"output": "hello"}
    assert state is not None
    assert state.request == {"input": "hello"}
    assert state.response == {"output": "hello"}
    assert state.session_id == session_id
    assert state.queue_order == (1 if session_id is not None else None)
    assert calls[0][1].session_id == session_id
    assert calls[0][1].attempt == 1
    assert calls[0][1].is_recovery is False


@pytest.mark.asyncio
@pytest.mark.parametrize("session_kwargs", [{}, {"session_id": None}], ids=["omitted", "none"])
async def test_invoke_preserves_legacy_submit_override(session_kwargs: dict[str, Any]) -> None:
    class LegacyRuntime(Runtime):
        # Existing subclasses may implement the signature from before session-aware submission.
        async def submit(  # ty: ignore[invalid-method-override]
            self, invocation_id: str, request: JsonValue
        ) -> Invocation:
            return await super().submit(invocation_id, {"wrapped_input": request})

    async def execute(request: JsonValue, context: InvocationAttemptContext) -> JsonValue:
        assert context.session_id is None
        return request

    runtime = LegacyRuntime.local(execute, poll_seconds=0.005)
    await runtime.start()
    try:
        response = await runtime.invoke(
            "invocation-1", {"message": "hello"}, timeout=1, **session_kwargs
        )
        state = await runtime.get_invocation("invocation-1")
    finally:
        await runtime.stop()

    assert response == {"wrapped_input": {"message": "hello"}}
    assert state is not None
    assert state.status == InvocationStatus.COMPLETED
    assert state.request == state.response == response
    assert state.session_id is None
    assert state.queue_order is None


@pytest.mark.asyncio
async def test_local_runtime_drains_a_session_queue_in_order() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    execution_order = []
    contexts = []
    running = 0
    max_running = 0

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        execution_order.append(request["message"])
        contexts.append(context)
        if request["message"] == "one":
            first_started.set()
            await release_first.wait()
        running -= 1
        return {"output": request["message"]}

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        first = await runtime.submit(
            "invocation-1",
            {"message": "one"},
            session_id="session-a",
        )
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = await runtime.submit(
            "invocation-2",
            {"message": "two"},
            session_id="session-a",
        )
        await asyncio.sleep(0.02)

        session_state = await runtime.get_invocation(session_id="session-a")
        assert session_state is not None
        assert session_state.invocation_id == "invocation-1"
        assert execution_order == ["one"]

        release_first.set()
        assert await runtime.wait("invocation-1") == {"output": "one"}
        assert await runtime.wait("invocation-2") == {"output": "two"}
        events = await runtime.get_events(session_id="session-a")
        assert await runtime.get_invocation(session_id="session-a") is None
    finally:
        release_first.set()
        await runtime.stop()

    assert (first.queue_order, second.queue_order) == (1, 2)
    assert execution_order == ["one", "two"]
    assert max_running == 1
    assert [context.session_id for context in contexts] == ["session-a", "session-a"]
    assert [event.invocation_id for event in events] == [
        "invocation-1",
        "invocation-1",
        "invocation-2",
        "invocation-2",
    ]


@pytest.mark.asyncio
async def test_local_runtime_runs_different_sessions_independently() -> None:
    both_started = asyncio.Event()
    release = asyncio.Event()
    started_sessions = set()

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        assert context.session_id is not None
        started_sessions.add(context.session_id)
        if len(started_sessions) == 2:
            both_started.set()
        await release.wait()
        return request

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        await runtime.submit("invocation-a", {}, session_id="session-a")
        await runtime.submit("invocation-b", {}, session_id="session-b")
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        await asyncio.gather(
            runtime.wait("invocation-a"),
            runtime.wait("invocation-b"),
        )
    finally:
        release.set()
        await runtime.stop()

    assert started_sessions == {"session-a", "session-b"}


@pytest.mark.asyncio
async def test_durable_runtime_drains_the_next_session_invocation_without_a_scan() -> None:
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    execution_order = []

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        execution_order.append(request["message"])
        if request["message"] == "one":
            first_started.set()
            await release_first.wait()
        else:
            second_started.set()
        return request

    store = MemoryDurableRuntimeStore()
    runtime = make_durable_runtime(execute, store, scan_seconds=60)
    await runtime.start()
    try:
        await runtime.submit("invocation-1", {"message": "one"}, session_id="session-a")
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await runtime.submit("invocation-2", {"message": "two"}, session_id="session-a")
        await asyncio.sleep(0.02)
        assert execution_order == ["one"]

        release_first.set()
        await asyncio.wait_for(second_started.wait(), timeout=1)
        assert await runtime.wait("invocation-2") == {"message": "two"}
    finally:
        release_first.set()
        await runtime.stop()

    assert execution_order == ["one", "two"]


@pytest.mark.asyncio
async def test_runtime_session_reads_require_exactly_one_selector() -> None:
    async def execute(request: JsonValue, context: InvocationAttemptContext) -> JsonValue:
        return request

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        with pytest.raises(ValueError, match="exactly one"):
            await runtime.get_invocation()
        with pytest.raises(ValueError, match="exactly one"):
            await runtime.get_invocation("invocation-1", session_id="session-a")
        with pytest.raises(ValueError, match="exactly one"):
            await runtime.get_events()
        with pytest.raises(ValueError, match="exactly one"):
            await runtime.get_events("invocation-1", session_id="session-a")
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_local_execution_does_not_use_heartbeats() -> None:
    release = asyncio.Event()

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        await release.wait()
        return {"output": "done"}

    store = TrackingInMemoryRuntimeStore()
    runtime = make_local_runtime(execute, store)
    await runtime.start()
    try:
        await runtime.submit("session-1", {"input": "hello"})
        await asyncio.sleep(0.03)
        assert store.heartbeat_calls == 0
        release.set()
        assert await runtime.wait("session-1") == {"output": "done"}
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_durable_execution_heartbeats_while_agent_is_running() -> None:
    release = asyncio.Event()

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        await release.wait()
        return {"output": "done"}

    store = MemoryDurableRuntimeStore()
    runtime = make_durable_runtime(execute, store)
    await runtime.start()
    try:
        await runtime.submit("session-1", {"input": "hello"})
        for _ in range(20):
            if store.heartbeats:
                break
            await asyncio.sleep(0.005)
        assert store.heartbeats
        assert set(store.heartbeats) == {("session-1", 1)}
        release.set()
        assert await runtime.wait("session-1") == {"output": "done"}
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_completed_request_returns_cached_response_without_reexecution() -> None:
    call_count = 0

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        nonlocal call_count
        call_count += 1
        return {"output": "done"}

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        first = await runtime.invoke("session-1", {"input": "hello"})
        second = await runtime.invoke("session-1", {"input": "hello"})
    finally:
        await runtime.stop()

    assert first == second == {"output": "done"}
    assert call_count == 1


@pytest.mark.asyncio
async def test_same_id_with_different_request_is_rejected() -> None:
    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        return {"output": "done"}

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        await runtime.invoke("session-1", {"input": "first"})
        with pytest.raises(InvocationConflictError):
            await runtime.submit("session-1", {"input": "second"})
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_stale_invocation_reuses_request_and_marks_recovery() -> None:
    contexts = []

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        contexts.append((request, context))
        return {"output": "recovered"}

    store = MemoryDurableRuntimeStore()
    store.seed_active(
        "session-1",
        {"input": "original"},
        heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        session_id="conversation-1",
        queue_order=4,
    )
    runtime = make_durable_runtime(execute, store, recover=True)
    await runtime.start()
    try:
        response = await runtime.wait("session-1")
    finally:
        await runtime.stop()

    assert response == {"output": "recovered"}
    assert contexts[0][0] == {"input": "original"}
    assert contexts[0][1].invocation_id == "session-1"
    assert contexts[0][1].attempt == 2
    assert contexts[0][1].is_recovery is True
    assert contexts[0][1].session_id == "conversation-1"
    recovered = store.states["session-1"]
    assert (recovered.session_id, recovered.queue_order) == ("conversation-1", 4)


@pytest.mark.asyncio
async def test_durable_runtime_does_not_recover_without_a_recovery_hook() -> None:
    called = False

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        nonlocal called
        called = True
        return {"output": "unexpected"}

    store = MemoryDurableRuntimeStore()
    store.seed_active(
        "session-1",
        {"input": "original"},
        heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    runtime = make_durable_runtime(execute, store, recover=False)
    await runtime.start()
    try:
        await asyncio.sleep(0.03)
        state = await runtime.get_invocation("session-1")
    finally:
        await runtime.stop()

    assert called is False
    assert state is not None
    assert state.status == InvocationStatus.ACTIVE
    assert state.attempt == 1


@pytest.mark.asyncio
async def test_durable_runtime_starts_persisted_queued_work_without_recovery_hook() -> None:
    contexts = []

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        contexts.append(context)
        return {"output": request["input"]}

    store = MemoryDurableRuntimeStore()
    await store.accept("session-1", {"input": "persisted before restart"})
    runtime = make_durable_runtime(execute, store, recover=False)

    await runtime.start()
    try:
        assert await runtime.wait("session-1") == {"output": "persisted before restart"}
    finally:
        await runtime.stop()

    assert len(contexts) == 1
    assert contexts[0].attempt == 1
    assert contexts[0].is_recovery is False


@pytest.mark.asyncio
async def test_executor_failure_is_persisted_as_terminal_state() -> None:
    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        raise RuntimeError("boom")

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        with pytest.raises(InvocationFailedError):
            await runtime.invoke("session-1", {"input": "hello"})
        state = await runtime.get_invocation("session-1")
    finally:
        await runtime.stop()

    assert state is not None
    assert state.status == InvocationStatus.FAILED
    assert state.response is None


@pytest.mark.asyncio
async def test_submit_returns_before_background_execution_finishes() -> None:
    release = asyncio.Event()

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        await release.wait()
        return {"output": "done"}

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        state = await runtime.submit("session-1", {"input": "hello"})
        assert state.status == InvocationStatus.QUEUED
        release.set()
        assert await runtime.wait("session-1") == {"output": "done"}
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_get_invocation_schedules_queued_work_accepted_by_another_process() -> None:
    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        return {"output": request["input"]}

    store = InMemoryRuntimeStore()
    runtime = make_local_runtime(execute, store)
    await runtime.start()
    try:
        await store.accept("session-1", {"input": "accepted elsewhere"})
        state = await runtime.get_invocation("session-1")
        assert state is not None
        assert state.status == InvocationStatus.QUEUED
        assert await runtime.wait("session-1") == {"output": "accepted elsewhere"}
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_wait_observes_response_completed_by_another_process() -> None:
    executor_called = False

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        nonlocal executor_called
        executor_called = True
        return {}

    store = InMemoryRuntimeStore()
    await store.accept("session-1", {"input": "hello"})
    await store.claim("session-1")
    runtime = make_local_runtime(execute, store)
    await runtime.start()

    async def complete_elsewhere() -> None:
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
async def test_blocking_timeout_does_not_cancel_execution() -> None:
    release = asyncio.Event()

    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        await release.wait()
        return {"output": "done"}

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        with pytest.raises(TimeoutError):
            await runtime.invoke("session-1", {"input": "hello"}, timeout=0.02)
        state = await runtime.get_invocation("session-1")
        assert state is not None
        assert state.status == InvocationStatus.ACTIVE
        release.set()
        assert await runtime.wait("session-1") == {"output": "done"}
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_executor_can_persist_replayable_events() -> None:
    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        sequence_number = await context.emit({"type": "progress", "step": 1})
        return {"last_sequence_number": sequence_number}

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        assert await runtime.invoke("session-1", {"input": "hello"}) == {"last_sequence_number": 2}
        events = await runtime.get_events("session-1")
        assert [(event.sequence_number, event.event) for event in events] == [
            (1, {"type": "run.started"}),
            (2, {"type": "progress", "step": 1}),
            (3, {"type": "run.completed"}),
        ]
        assert [
            event.event for event in await runtime.get_events("session-1", after_sequence=1)
        ] == [
            {"type": "progress", "step": 1},
            {"type": "run.completed"},
        ]
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_request_and_response_may_be_any_json_value() -> None:
    async def execute(request: JsonValue, context: InvocationAttemptContext) -> JsonValue:
        return None

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        assert await runtime.invoke("session-1", ["not", "an", "object"]) is None
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_runtime_uses_injected_execution_wiring() -> None:
    async def execute(request: JsonValue, context: InvocationAttemptContext) -> JsonValue:
        assert isinstance(request, dict)
        return {"attempt": context.attempt, "input": request["input"]}

    runtime = make_local_runtime(execute)
    await runtime.start()
    try:
        assert await runtime.invoke("session-1", {"input": "hello"}) == {
            "attempt": 1,
            "input": "hello",
        }
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_start_and_stop_manage_store_lifecycle() -> None:
    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        return {}

    store = TrackingInMemoryRuntimeStore()
    runtime = make_local_runtime(execute, store)

    await runtime.start()
    assert store.initialized is True
    await runtime.stop()
    assert store.closed is True


@pytest.mark.asyncio
async def test_runtime_requires_start_before_use() -> None:
    async def execute(request: dict, context: InvocationAttemptContext) -> dict:
        return {}

    runtime = make_local_runtime(execute)
    with pytest.raises(RuntimeError, match=r"start\(\)"):
        await runtime.submit("session-1", {})
