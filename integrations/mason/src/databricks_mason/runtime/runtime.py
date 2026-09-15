"""Transport-neutral facade for Mason invocation execution."""

from __future__ import annotations

import asyncio
import copy

from databricks_mason.runtime.execution import InvocationExecutor, copy_json_value
from databricks_mason.runtime.store import RuntimeStore
from databricks_mason.runtime.types import (
    DurableEvent,
    DurableExecution,
    DurableExecutionFailedError,
    DurableExecutionNotFoundError,
    DurableExecutionStatus,
    JsonValue,
)


class Runtime:
    """Coordinate invocation state, execution scheduling, polling, and event replay."""

    def __init__(
        self,
        *,
        runtime_store: RuntimeStore,
        executor: InvocationExecutor,
        poll_seconds: float = 1.0,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.runtime_store = runtime_store
        self.durability_store = runtime_store
        self.executor = executor
        self.poll_seconds = poll_seconds
        self._started = False

    async def start(self) -> None:
        """Initialize the Runtime Store and invocation executor."""
        if self._started:
            return
        await self.runtime_store.initialize()
        try:
            await self.executor.start()
        except Exception:
            await self.runtime_store.close()
            raise
        self._started = True

    async def stop(self) -> None:
        """Stop local work and close the Runtime Store."""
        if not self._started:
            return
        await self.executor.stop()
        self._started = False
        await self.runtime_store.close()

    async def submit(self, execution_id: str, request: JsonValue) -> DurableExecution:
        """Accept an idempotent request and schedule it on this worker."""
        self._require_started()
        if not execution_id:
            raise ValueError("execution_id must not be empty")
        state = await self.runtime_store.accept(
            execution_id,
            copy_json_value(request, "request"),
        )
        self.executor.ensure_scheduled(state)
        return state

    async def invoke(
        self,
        execution_id: str,
        request: JsonValue,
        *,
        timeout: float | None = None,
    ) -> JsonValue:
        """Accept a request and wait for its terminal response."""
        await self.submit(execution_id, request)
        return await self.wait(execution_id, timeout=timeout)

    async def get_execution(self, execution_id: str) -> DurableExecution | None:
        """Return the latest invocation state."""
        self._require_started()
        return await self.runtime_store.get(execution_id)

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
        """Return events after an optional exclusive replay cursor."""
        self._require_started()
        return await self.runtime_store.events(execution_id, after_sequence)

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Runtime.start() must be called first")
