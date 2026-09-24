"""Process-local scheduling and shared agent attempt execution."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import Protocol, cast

from databricks_agentbricks.runtime.store import RuntimeStore
from databricks_agentbricks.runtime.types import (
    Invocation,
    InvocationAttemptContext,
    InvocationExecutorFn,
    InvocationStatus,
    JsonObject,
    JsonValue,
)

logger = logging.getLogger(__name__)


def copy_json_value(value: JsonValue, name: str) -> JsonValue:
    """Return a detached JSON value or reject a value that cannot be persisted."""
    try:
        return cast(JsonValue, json.loads(json.dumps(value, allow_nan=False)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON serializable") from exc


def _copy_json_object(value: JsonObject, name: str) -> JsonObject:
    copied = copy_json_value(value, name)
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must be a JSON object")
    return copied


class InvocationExecutor(Protocol):
    """Schedule invocation attempts for one runtime process."""

    async def start(self) -> None:
        """Start any background scheduling owned by this executor."""
        ...

    async def stop(self) -> None:
        """Stop scheduled work and release executor resources."""
        ...

    def ensure_scheduled(self, state: Invocation) -> None:
        """Arrange for a newly accepted invocation to run on this process."""
        ...


class AttemptExecution:
    """Call the agent hook and commit one already-claimed attempt."""

    def __init__(self, execute_fn: InvocationExecutorFn, *, runtime_store: RuntimeStore) -> None:
        self._execute_fn = execute_fn
        self._runtime_store = runtime_store

    async def run(self, claimed: Invocation) -> None:
        invocation_id = claimed.invocation_id
        try:

            async def emit(event: JsonObject) -> int:
                sequence_number = await self._runtime_store.append_event(
                    invocation_id,
                    claimed.attempt,
                    _copy_json_object(event, "event"),
                )
                if sequence_number is None:
                    raise RuntimeError(
                        f"invocation {invocation_id!r} no longer owns attempt {claimed.attempt}"
                    )
                return sequence_number

            response = await self._execute_fn(
                copy.deepcopy(claimed.request),
                InvocationAttemptContext(
                    invocation_id=invocation_id,
                    attempt=claimed.attempt,
                    _emit=emit,
                ),
            )
            response = copy_json_value(response, "executor response")
            completed = await self._runtime_store.complete(
                invocation_id,
                claimed.attempt,
                response,
            )
            if not completed:
                logger.info(
                    "Skipped completion after Runtime Store ownership changed: %s attempt=%d",
                    invocation_id,
                    claimed.attempt,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Invocation execution failed: %s attempt=%d",
                invocation_id,
                claimed.attempt,
            )
            try:
                await self._runtime_store.fail(invocation_id, claimed.attempt)
            except Exception:
                logger.exception(
                    "Failed to persist invocation failure: %s attempt=%d",
                    invocation_id,
                    claimed.attempt,
                )


class LocalInvocationExecutor(InvocationExecutor):
    """Run background invocations inside the current process without heartbeats."""

    def __init__(self, execution: AttemptExecution, *, runtime_store: RuntimeStore) -> None:
        self._execution = execution
        self._runtime_store = runtime_store
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def ensure_scheduled(self, state: Invocation) -> None:
        if state.status != InvocationStatus.QUEUED:
            return
        current = self._tasks.get(state.invocation_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._run(state.invocation_id),
            name=f"invocation-{state.invocation_id}",
        )
        self._tasks[state.invocation_id] = task
        task.add_done_callback(lambda completed: self._discard_task(state.invocation_id, completed))

    async def _run(self, invocation_id: str) -> None:
        try:
            claimed = await self._runtime_store.claim(invocation_id)
        except Exception:
            logger.exception("Failed to claim invocation: %s", invocation_id)
            return
        if claimed is not None:
            await self._execution.run(claimed)

    def _discard_task(self, invocation_id: str, completed: asyncio.Task[None]) -> None:
        if self._tasks.get(invocation_id) is completed:
            self._tasks.pop(invocation_id, None)
        if not completed.cancelled():
            completed.exception()
