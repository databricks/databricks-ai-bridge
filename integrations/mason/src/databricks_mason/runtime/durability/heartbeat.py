"""Heartbeat lifecycle for a durable invocation attempt."""

from __future__ import annotations

import asyncio
import logging

from databricks_mason.runtime.durability.store import DurableRuntimeStore

logger = logging.getLogger(__name__)


class Heartbeat:
    """Refresh attempt ownership until execution finishes or ownership changes."""

    def __init__(
        self,
        *,
        runtime_store: DurableRuntimeStore,
        invocation_id: str,
        attempt: int,
        heartbeat_seconds: float,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self._runtime_store = runtime_store
        self._invocation_id = invocation_id
        self._attempt = attempt
        self._heartbeat_seconds = heartbeat_seconds
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Heartbeat:
        self._task = asyncio.create_task(
            self._run(),
            name=f"durable-heartbeat-{self._invocation_id}-{self._attempt}",
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                owns_attempt = await self._runtime_store.heartbeat(
                    self._invocation_id,
                    self._attempt,
                )
            except Exception:
                logger.warning(
                    "Durable heartbeat failed: %s attempt=%d",
                    self._invocation_id,
                    self._attempt,
                    exc_info=True,
                )
                await asyncio.sleep(self._heartbeat_seconds)
                continue
            if not owns_attempt:
                return
            await asyncio.sleep(self._heartbeat_seconds)
