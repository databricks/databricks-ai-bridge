"""Detection and scheduling of recoverable durable attempts."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from databricks_mason.runtime.durability.store import DurableRuntimeStore

logger = logging.getLogger(__name__)


class RecoverySchedulerTarget(Protocol):
    """Schedule one invocation after recovery finds it eligible."""

    def ensure_recovery_scheduled(self, invocation_id: str) -> None:
        """Schedule a queued or stale invocation for a recovery attempt."""
        ...


class RecoveryScheduler:
    """Find queued or stale durable attempts when recovery is enabled."""

    def __init__(
        self,
        scheduler: RecoverySchedulerTarget,
        *,
        runtime_store: "DurableRuntimeStore",
        stale_seconds: float,
        scan_seconds: float,
    ) -> None:
        if scan_seconds <= 0:
            raise ValueError("scan_seconds must be positive")
        self._scheduler = scheduler
        self._runtime_store = runtime_store
        self._stale_seconds = stale_seconds
        self._scan_seconds = scan_seconds
        self._scanner: asyncio.Task[None] | None = None

    def start(self, *, recover: bool) -> None:
        """Enable scheduling and optionally start proactive stale-work scanning."""
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

    async def _scan_loop(self) -> None:
        while True:
            try:
                invocation_ids = await self._runtime_store.recoverable_invocation_ids(
                    self._stale_seconds
                )
                for invocation_id in invocation_ids:
                    self._scheduler.ensure_recovery_scheduled(invocation_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Databricks durable runtime recovery scan failed")
            await asyncio.sleep(self._scan_seconds)
