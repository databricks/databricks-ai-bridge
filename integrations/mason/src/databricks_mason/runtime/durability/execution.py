"""Durable scheduling and execution of fenced invocation attempts."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from databricks_mason.runtime.durability.heartbeat import Heartbeat
from databricks_mason.runtime.durability.recovery import RecoveryScheduler
from databricks_mason.runtime.durability.store import DurableRuntimeStore
from databricks_mason.runtime.execution import AttemptExecution, InvocationExecutor
from databricks_mason.runtime.types import Invocation, InvocationStatus

logger = logging.getLogger(__name__)


class DurableInvocationExecutor(InvocationExecutor):
    """Claim durable attempts and wrap shared execution with heartbeats."""

    def __init__(
        self,
        execution: AttemptExecution,
        *,
        runtime_store: DurableRuntimeStore,
        recovery_enabled: Callable[[], bool],
        heartbeat_seconds: float,
        stale_seconds: float,
        scan_seconds: float,
    ) -> None:
        if stale_seconds <= heartbeat_seconds:
            raise ValueError("stale_seconds must be greater than heartbeat_seconds")
        self._execution = execution
        self._runtime_store = runtime_store
        self._recovery_enabled = recovery_enabled
        self._heartbeat_seconds = heartbeat_seconds
        self._stale_seconds = stale_seconds
        self._scan_seconds = scan_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._queued_scanner: asyncio.Task[None] | None = None
        self._recovery_scheduler: RecoveryScheduler | None = None

    async def start(self) -> None:
        self._queued_scanner = asyncio.create_task(
            self._scan_queued_loop(),
            name="databricks-durable-runtime-queued-scanner",
        )
        self._recovery_scheduler = RecoveryScheduler(
            scheduler=self,
            runtime_store=self._runtime_store,
            stale_seconds=self._stale_seconds,
            scan_seconds=self._scan_seconds,
        )
        self._recovery_scheduler.start(recover=self._recovery_enabled())

    async def stop(self) -> None:
        if self._queued_scanner is not None:
            self._queued_scanner.cancel()
        if self._recovery_scheduler is not None:
            await self._recovery_scheduler.stop()
            self._recovery_scheduler = None
        await asyncio.gather(
            *([self._queued_scanner] if self._queued_scanner is not None else []),
            return_exceptions=True,
        )
        self._queued_scanner = None
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def ensure_scheduled(self, state: Invocation) -> None:
        """Schedule a newly queued invocation on this worker."""
        if state.status != InvocationStatus.QUEUED:
            return
        self._schedule(state.invocation_id, recovery=False)

    def ensure_recovery_scheduled(self, invocation_id: str) -> None:
        """Schedule stale active work discovered by the recovery scanner."""
        self._schedule(invocation_id, recovery=True)

    async def _scan_queued_loop(self) -> None:
        """Continuously schedule persisted first attempts, including after process restart."""
        while True:
            try:
                for invocation_id in await self._runtime_store.queued_invocation_ids():
                    self._schedule(invocation_id, recovery=False)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Databricks durable runtime queued invocation scan failed")
            await asyncio.sleep(self._scan_seconds)

    def _schedule(self, invocation_id: str, *, recovery: bool) -> None:
        current = self._tasks.get(invocation_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._run(invocation_id, recovery=recovery),
            name=f"durable-invocation-{invocation_id}",
        )
        self._tasks[invocation_id] = task
        task.add_done_callback(lambda completed: self._discard_task(invocation_id, completed))

    async def _run(self, invocation_id: str, *, recovery: bool) -> None:
        try:
            if recovery:
                claimed = await self._runtime_store.claim_recoverable(
                    invocation_id,
                    self._stale_seconds,
                )
            else:
                claimed = await self._runtime_store.claim(invocation_id)
        except Exception:
            logger.exception("Failed to claim durable invocation: %s", invocation_id)
            return
        if claimed is None:
            return

        async with Heartbeat(
            runtime_store=self._runtime_store,
            invocation_id=invocation_id,
            attempt=claimed.attempt,
            heartbeat_seconds=self._heartbeat_seconds,
        ):
            await self._execution.run(claimed)

    def _discard_task(self, invocation_id: str, completed: asyncio.Task[None]) -> None:
        if self._tasks.get(invocation_id) is completed:
            self._tasks.pop(invocation_id, None)
        if not completed.cancelled():
            completed.exception()
