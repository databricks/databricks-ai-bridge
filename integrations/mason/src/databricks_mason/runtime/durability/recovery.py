"""Detection and scheduling of recoverable durable attempts."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from databricks_mason.runtime.durability.attempt import DurableInvocationExecutor
from databricks_mason.runtime.durability.store import DurableRuntimeStore
from databricks_mason.runtime.types import (
    DurableExecution,
    DurableExecutionStatus,
)

logger = logging.getLogger(__name__)


class RecoveryScheduler:
    """Find queued or stale durable attempts when recovery is enabled."""

    def __init__(
        self,
        executor: DurableInvocationExecutor,
        *,
        runtime_store: DurableRuntimeStore,
        stale_seconds: float,
        scan_seconds: float,
    ) -> None:
        if scan_seconds <= 0:
            raise ValueError("scan_seconds must be positive")
        self._executor = executor
        self._runtime_store = runtime_store
        self._stale_seconds = stale_seconds
        self._scan_seconds = scan_seconds
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
        """Stop scanning, leaving active rows recoverable elsewhere."""
        if self._scanner is not None:
            self._scanner.cancel()
        await asyncio.gather(
            *([self._scanner] if self._scanner is not None else []),
            return_exceptions=True,
        )
        self._scanner = None
        self._recover = False

    def ensure_scheduled(self, state: DurableExecution) -> None:
        """Schedule execution when the persisted state is eligible on this worker."""
        if not self._is_recoverable(state):
            return
        self._executor.ensure_recovery_scheduled(state)

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
                execution_ids = await self._runtime_store.recoverable_execution_ids(
                    self._stale_seconds
                )
                for execution_id in execution_ids:
                    state = await self._runtime_store.get(execution_id)
                    if state is not None:
                        self.ensure_scheduled(state)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Databricks durable runtime recovery scan failed")
            await asyncio.sleep(self._scan_seconds)
