"""Scheduling of initial and recovered durable attempts."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from databricks_mason.runtime.durability.attempt import AttemptRunner
from databricks_mason.runtime.durability.types import (
    DurabilityStore,
    DurableExecution,
    DurableExecutionStatus,
)

logger = logging.getLogger(__name__)


class RecoveryScheduler:
    """Schedule queued work and reclaim stale attempts when recovery is enabled."""

    def __init__(
        self,
        attempt_runner: AttemptRunner,
        *,
        durability_store: DurabilityStore,
        stale_seconds: float,
        scan_seconds: float,
    ) -> None:
        if scan_seconds <= 0:
            raise ValueError("scan_seconds must be positive")
        self._attempt_runner = attempt_runner
        self._durability_store = durability_store
        self._stale_seconds = stale_seconds
        self._scan_seconds = scan_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._scanner: asyncio.Task[None] | None = None
        self._recover = False

    def start(self, *, recover: bool) -> None:
        """Enable scheduling and optionally start proactive stale-work scanning."""
        self._recover = recover
        if recover:
            self._scanner = asyncio.create_task(
                self._scan_loop(),
                name="databricks-durable-runtime-scanner",
            )

    async def stop(self) -> None:
        """Cancel local attempts and scanning, leaving active rows recoverable elsewhere."""
        tasks = list(self._tasks.values())
        if self._scanner is not None:
            self._scanner.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(
            *tasks,
            *([self._scanner] if self._scanner is not None else []),
            return_exceptions=True,
        )
        self._tasks.clear()
        self._scanner = None
        self._recover = False

    def ensure_scheduled(self, state: DurableExecution) -> None:
        """Schedule execution when the persisted state is eligible on this worker."""
        if not self._is_recoverable(state):
            return
        current = self._tasks.get(state.execution_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._attempt_runner.run(state.execution_id),
            name=f"durable-execution-{state.execution_id}",
        )
        self._tasks[state.execution_id] = task
        task.add_done_callback(lambda completed: self._discard_task(state.execution_id, completed))

    def _is_recoverable(self, state: DurableExecution) -> bool:
        if state.status == DurableExecutionStatus.QUEUED:
            return True
        if state.status != DurableExecutionStatus.ACTIVE or not self._recover:
            return False
        if state.heartbeat_at is None:
            return True
        heartbeat_at = state.heartbeat_at
        if heartbeat_at.tzinfo is None:
            heartbeat_at = heartbeat_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - heartbeat_at).total_seconds()
        return age >= self._stale_seconds

    async def _scan_loop(self) -> None:
        while True:
            try:
                execution_ids = await self._durability_store.recoverable_execution_ids(
                    self._stale_seconds
                )
                for execution_id in execution_ids:
                    state = await self._durability_store.get(execution_id)
                    if state is not None:
                        self.ensure_scheduled(state)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Databricks durable runtime recovery scan failed")
            await asyncio.sleep(self._scan_seconds)

    def _discard_task(self, execution_id: str, completed: asyncio.Task[None]) -> None:
        if self._tasks.get(execution_id) is completed:
            self._tasks.pop(execution_id, None)
        if not completed.cancelled():
            completed.exception()
