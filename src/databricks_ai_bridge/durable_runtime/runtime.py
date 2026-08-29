"""Transport-neutral durable request runtime."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from databricks_ai_bridge.durable_runtime.store import LakebaseDurabilityStore
from databricks_ai_bridge.durable_runtime.types import (
    DurabilityStore,
    DurableEvent,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionNotFoundError,
    DurableExecutionStatus,
    DurableExecutor,
    JsonObject,
)

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


def _copy_json_object(value: JsonObject, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object, got {type(value).__name__}")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON serializable") from exc


class DatabricksDurableRuntime:
    """Execute idempotent JSON requests with Lakebase-backed crash recovery.

    The runtime persists only request execution state. The executor owns agent
    sessions, checkpoints, tools, and any recovery-specific prompt or behavior.
    """

    def __init__(
        self,
        executor: DurableExecutor | None = None,
        *,
        durability_store: DurabilityStore | None = None,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str = "databricks_durable_runtime",
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 1.0,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if stale_seconds <= heartbeat_seconds:
            raise ValueError("stale_seconds must be greater than heartbeat_seconds")
        if scan_seconds <= 0:
            raise ValueError("scan_seconds must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

        self._executor = executor
        self.durability_store = (
            durability_store
            if durability_store is not None
            else LakebaseDurabilityStore(
                autoscaling_endpoint=autoscaling_endpoint,
                project=project,
                branch=branch,
                workspace_client=workspace_client,
                schema=schema,
            )
        )
        self.heartbeat_seconds = heartbeat_seconds
        self.stale_seconds = stale_seconds
        self.scan_seconds = scan_seconds
        self.poll_seconds = poll_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._scanner: asyncio.Task[None] | None = None
        self._started = False

    async def execute(
        self,
        request: JsonObject,
        context: DurableExecutionContext,
    ) -> JsonObject:
        """Run one attempt; subclasses may override this method."""
        if self._executor is None:
            raise NotImplementedError("provide an executor or override execute()")
        return await self._executor(request, context)

    async def start(self) -> None:
        """Initialize storage and start proactive recovery scanning."""
        if self._started:
            return
        await self.durability_store.initialize()
        self._started = True
        self._scanner = asyncio.create_task(
            self._scan_loop(),
            name="databricks-durable-runtime-scanner",
        )

    async def stop(self) -> None:
        """Stop local work, leaving active rows recoverable by another process."""
        if not self._started:
            return
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
        self._started = False
        await self.durability_store.close()

    async def submit(self, execution_id: str, request: JsonObject) -> DurableExecution:
        """Accept an idempotent request and ensure recoverable work is scheduled."""
        self._require_started()
        if not execution_id:
            raise ValueError("execution_id must not be empty")
        state = await self.durability_store.accept(
            execution_id,
            _copy_json_object(request, "request"),
        )
        self._ensure_scheduled(state)
        return state

    async def invoke(
        self,
        execution_id: str,
        request: JsonObject,
        *,
        timeout: float | None = None,
    ) -> JsonObject:
        """Accept a request and wait for its persisted terminal response."""
        await self.submit(execution_id, request)
        return await self.wait(execution_id, timeout=timeout)

    async def get(self, execution_id: str) -> DurableExecution | None:
        """Return persisted state and schedule recovery if it is currently eligible."""
        self._require_started()
        state = await self.durability_store.get(execution_id)
        if state is not None:
            self._ensure_scheduled(state)
        return state

    async def wait(
        self,
        execution_id: str,
        *,
        timeout: float | None = None,
    ) -> JsonObject:
        """Wait for a completed response, including work owned by another process."""
        self._require_started()

        async def poll() -> JsonObject:
            while True:
                state = await self.get(execution_id)
                if state is None:
                    raise DurableExecutionNotFoundError(execution_id)
                if state.status == DurableExecutionStatus.COMPLETED:
                    if state.response is None:
                        raise RuntimeError(
                            f"execution {execution_id!r} completed without a response"
                        )
                    return copy.deepcopy(state.response)
                if state.status == DurableExecutionStatus.FAILED:
                    raise DurableExecutionFailedError(execution_id)
                await asyncio.sleep(self.poll_seconds)

        if timeout is None:
            return await poll()
        return await asyncio.wait_for(poll(), timeout=timeout)

    async def events(
        self,
        execution_id: str,
        *,
        after_sequence: int | None = None,
    ) -> list[DurableEvent]:
        """Return persisted events after an optional replay cursor."""
        self._require_started()
        return await self.durability_store.events(execution_id, after_sequence)

    def _ensure_scheduled(self, state: DurableExecution) -> None:
        if not self._is_recoverable(state):
            return
        current = self._tasks.get(state.execution_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._execute_attempt(state.execution_id),
            name=f"durable-execution-{state.execution_id}",
        )
        self._tasks[state.execution_id] = task
        task.add_done_callback(lambda completed: self._discard_task(state.execution_id, completed))

    def _is_recoverable(self, state: DurableExecution) -> bool:
        if state.status == DurableExecutionStatus.QUEUED:
            return True
        if state.status != DurableExecutionStatus.ACTIVE:
            return False
        if state.heartbeat_at is None:
            return True
        heartbeat_at = state.heartbeat_at
        if heartbeat_at.tzinfo is None:
            heartbeat_at = heartbeat_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - heartbeat_at).total_seconds()
        return age >= self.stale_seconds

    async def _execute_attempt(self, execution_id: str) -> None:
        try:
            claimed = await self.durability_store.claim(execution_id, self.stale_seconds)
        except Exception:
            logger.exception("Failed to claim durable execution: %s", execution_id)
            return
        if claimed is None:
            return

        heartbeat = asyncio.create_task(
            self._heartbeat_loop(execution_id, claimed.attempt),
            name=f"durable-heartbeat-{execution_id}-{claimed.attempt}",
        )
        try:

            async def emit(event: JsonObject) -> int:
                sequence_number = await self.durability_store.append_event(
                    execution_id,
                    claimed.attempt,
                    _copy_json_object(event, "event"),
                )
                if sequence_number is None:
                    raise RuntimeError(
                        f"execution {execution_id!r} no longer owns attempt {claimed.attempt}"
                    )
                return sequence_number

            response = await self.execute(
                copy.deepcopy(claimed.request),
                DurableExecutionContext(
                    execution_id=execution_id,
                    attempt=claimed.attempt,
                    _emit=emit,
                ),
            )
            response = _copy_json_object(response, "executor response")
            completed = await self.durability_store.complete(
                execution_id,
                claimed.attempt,
                response,
            )
            if not completed:
                logger.info(
                    "Skipped completion after durability ownership changed: %s attempt=%d",
                    execution_id,
                    claimed.attempt,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Durable execution failed: %s attempt=%d",
                execution_id,
                claimed.attempt,
            )
            try:
                await self.durability_store.fail(execution_id, claimed.attempt)
            except Exception:
                logger.exception(
                    "Failed to persist durable failure: %s attempt=%d",
                    execution_id,
                    claimed.attempt,
                )
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _heartbeat_loop(self, execution_id: str, attempt: int) -> None:
        while True:
            try:
                owns_attempt = await self.durability_store.heartbeat(execution_id, attempt)
            except Exception:
                logger.warning(
                    "Durable heartbeat failed: %s attempt=%d",
                    execution_id,
                    attempt,
                    exc_info=True,
                )
                await asyncio.sleep(self.heartbeat_seconds)
                continue
            if not owns_attempt:
                return
            await asyncio.sleep(self.heartbeat_seconds)

    async def _scan_loop(self) -> None:
        while True:
            try:
                execution_ids = await self.durability_store.recoverable_execution_ids(
                    self.stale_seconds
                )
                for execution_id in execution_ids:
                    state = await self.durability_store.get(execution_id)
                    if state is not None:
                        self._ensure_scheduled(state)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Databricks durable runtime recovery scan failed")
            await asyncio.sleep(self.scan_seconds)

    def _discard_task(self, execution_id: str, completed: asyncio.Task[None]) -> None:
        if self._tasks.get(execution_id) is completed:
            self._tasks.pop(execution_id, None)
        if not completed.cancelled():
            completed.exception()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("DatabricksDurableRuntime.start() must be called first")
