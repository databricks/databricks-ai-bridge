"""Transport-neutral facade for durable request execution."""

from __future__ import annotations

import asyncio
import copy

from databricks_mason.runtime.durability.attempt import AttemptRunner, copy_json_value
from databricks_mason.runtime.durability.recovery import RecoveryScheduler
from databricks_mason.runtime.durability.types import (
    DurabilityStore,
    DurableEvent,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionNotFoundError,
    DurableExecutionStatus,
    DurableExecutorFn,
    JsonValue,
)


class DurableRuntime:
    """Coordinate idempotent execution, leases, recovery, and event replay.

    ``submit`` first records an immutable request in the configured ``DurabilityStore``. The
    ``RecoveryScheduler`` schedules eligible work, and ``AttemptRunner`` atomically claims one
    attempt, runs ``execute_fn``, and refreshes its heartbeat until output or failure is committed.
    When recovery is enabled, the scheduler also reclaims active work whose heartbeat has become
    stale. Attempt numbers fence late writes from replaced workers, while persisted events let
    clients replay progress across processes.

    Durability depends on the configured store: Mason uses process-local memory during development
    and Lakebase in deployed Apps. This runtime persists execution state only; the executor remains
    responsible for agent checkpoints and idempotent external side effects.
    """

    def __init__(
        self,
        execute_fn: DurableExecutorFn | None = None,
        *,
        durability_store: DurabilityStore,
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 1.0,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

        self._execute_fn = execute_fn
        self.durability_store = durability_store
        self.heartbeat_seconds = heartbeat_seconds
        self.stale_seconds = stale_seconds
        self.scan_seconds = scan_seconds
        self.poll_seconds = poll_seconds
        self._attempt_runner = AttemptRunner(
            self.execute,
            durability_store=durability_store,
            heartbeat_seconds=heartbeat_seconds,
            stale_seconds=stale_seconds,
        )
        self._recovery_scheduler = RecoveryScheduler(
            self._attempt_runner,
            durability_store=durability_store,
            stale_seconds=stale_seconds,
            scan_seconds=scan_seconds,
        )
        self._started = False

    async def execute(
        self,
        request: JsonValue,
        context: DurableExecutionContext,
    ) -> JsonValue:
        """Run one attempt; subclasses may override this method."""
        if self._execute_fn is None:
            raise NotImplementedError("provide an executor or override execute()")
        return await self._execute_fn(request, context)

    async def start(self, *, recover: bool = True) -> None:
        """Initialize storage and optionally start proactive recovery scanning."""
        if self._started:
            return
        await self.durability_store.initialize()
        self._started = True
        self._recovery_scheduler.start(recover=recover)

    async def stop(self) -> None:
        """Stop local work, leaving active rows recoverable by another process."""
        if not self._started:
            return
        await self._recovery_scheduler.stop()
        self._started = False
        await self.durability_store.close()

    async def submit(self, execution_id: str, request: JsonValue) -> DurableExecution:
        """Accept an idempotent request and ensure recoverable work is scheduled."""
        self._require_started()
        if not execution_id:
            raise ValueError("execution_id must not be empty")
        state = await self.durability_store.accept(
            execution_id,
            copy_json_value(request, "request"),
        )
        self._recovery_scheduler.ensure_scheduled(state)
        return state

    async def invoke(
        self,
        execution_id: str,
        request: JsonValue,
        *,
        timeout: float | None = None,
    ) -> JsonValue:
        """Accept a request and wait for its persisted terminal response."""
        await self.submit(execution_id, request)
        return await self.wait(execution_id, timeout=timeout)

    async def get_execution(self, execution_id: str) -> DurableExecution | None:
        """Return persisted state and schedule recovery if it is currently eligible."""
        self._require_started()
        state = await self.durability_store.get(execution_id)
        if state is not None:
            self._recovery_scheduler.ensure_scheduled(state)
        return state

    async def wait(
        self,
        execution_id: str,
        *,
        timeout: float | None = None,
    ) -> JsonValue:
        """Wait for a completed response, including work owned by another process."""
        self._require_started()

        async def poll() -> JsonValue:
            while True:
                state = await self.get_execution(execution_id)
                if state is None:
                    raise DurableExecutionNotFoundError(execution_id)
                if state.status == DurableExecutionStatus.COMPLETED:
                    return copy.deepcopy(state.response)
                if state.status == DurableExecutionStatus.FAILED:
                    raise DurableExecutionFailedError(execution_id)
                await asyncio.sleep(self.poll_seconds)

        if timeout is None:
            return await poll()
        try:
            return await asyncio.wait_for(poll(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError from exc

    async def get_events(
        self,
        execution_id: str,
        *,
        after_sequence: int | None = None,
    ) -> list[DurableEvent]:
        """Return persisted events after an optional replay cursor."""
        self._require_started()
        return await self.durability_store.events(execution_id, after_sequence)

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("DurableRuntime.start() must be called first")
