"""Process-local scheduling and shared agent attempt execution."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol, cast

from databricks_agentkit.runtime.auth import AuthError
from databricks_agentkit.runtime.store import RuntimeStore
from databricks_agentkit.runtime.types import (
    Invocation,
    InvocationAttemptContext,
    InvocationExecutorFn,
    InvocationStatus,
    JsonObject,
    JsonValue,
)

logger = logging.getLogger(__name__)


class InvocationUpdates:
    """Wake local readers after persisted changes; other workers still poll the store.

    Subscriptions are registered before reading, so a commit during a read cannot be missed.
    Only active readers occupy memory, and disconnecting a stream releases its subscription.
    """

    def __init__(self) -> None:
        self._readers: dict[str, set[asyncio.Event]] = {}

    @contextmanager
    def subscribe(self, invocation_id: str) -> Iterator[asyncio.Event]:
        changed = asyncio.Event()
        readers = self._readers.setdefault(invocation_id, set())
        readers.add(changed)
        try:
            yield changed
        finally:
            readers.discard(changed)
            if not readers:
                self._readers.pop(invocation_id, None)

    def notify(self, invocation_id: str) -> None:
        for reader in self._readers.get(invocation_id, ()):
            reader.set()


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

    def __init__(
        self,
        execute_fn: InvocationExecutorFn,
        *,
        runtime_store: RuntimeStore,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        self._execute_fn = execute_fn
        self._runtime_store = runtime_store
        self._on_change = on_change or (lambda _invocation_id: None)

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
                self._on_change(invocation_id)
                return sequence_number

            response = await self._execute_fn(
                copy.deepcopy(claimed.request),
                InvocationAttemptContext(
                    invocation_id=invocation_id,
                    attempt=claimed.attempt,
                    session_id=claimed.session_id,
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
        except Exception as error:
            logger.exception(
                "Invocation execution failed: %s attempt=%d",
                invocation_id,
                claimed.attempt,
            )
            try:
                failure_response: JsonValue = None
                if isinstance(error, AuthError):
                    failure_response = {
                        "error": error.payload(),
                        "status_code": error.status_code,
                    }
                await self._runtime_store.fail(
                    invocation_id,
                    claimed.attempt,
                    failure_response,
                )
            except Exception:
                logger.exception(
                    "Failed to persist invocation failure: %s attempt=%d",
                    invocation_id,
                    claimed.attempt,
                )
        finally:
            self._on_change(invocation_id)


class LocalInvocationExecutor(InvocationExecutor):
    """Run background invocations inside the current process without heartbeats."""

    def __init__(self, execution: AttemptExecution, *, runtime_store: RuntimeStore) -> None:
        self._execution = execution
        self._runtime_store = runtime_store
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._claim_retries: set[str] = set()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self._claim_retries.clear()
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def ensure_scheduled(self, state: Invocation) -> None:
        if state.status != InvocationStatus.QUEUED:
            return
        self._schedule(state.invocation_id, retry_if_running=False)

    def _schedule(self, invocation_id: str, *, retry_if_running: bool) -> None:
        current = self._tasks.get(invocation_id)
        if current is not None and not current.done():
            if retry_if_running:
                self._claim_retries.add(invocation_id)
            return
        task = asyncio.create_task(
            self._run(invocation_id),
            name=f"invocation-{invocation_id}",
        )
        self._tasks[invocation_id] = task
        task.add_done_callback(lambda completed: self._discard_task(invocation_id, completed))

    async def _run(self, invocation_id: str) -> None:
        while True:
            try:
                claimed = await self._runtime_store.claim(invocation_id)
            except Exception:
                logger.exception("Failed to claim invocation: %s", invocation_id)
                claimed = None
            if claimed is not None:
                self._claim_retries.discard(invocation_id)
                break
            if invocation_id not in self._claim_retries:
                return
            self._claim_retries.remove(invocation_id)

        await self._execution.run(claimed)
        if claimed.session_id is not None:
            try:
                next_state = await self._runtime_store.get(session_id=claimed.session_id)
            except Exception:
                logger.exception("Failed session handoff after invocation: %s", invocation_id)
            else:
                if next_state is not None and next_state.status == InvocationStatus.QUEUED:
                    self._schedule(next_state.invocation_id, retry_if_running=True)

    def _discard_task(self, invocation_id: str, completed: asyncio.Task[None]) -> None:
        if self._tasks.get(invocation_id) is completed:
            self._tasks.pop(invocation_id, None)
            self._claim_retries.discard(invocation_id)
        if not completed.cancelled():
            completed.exception()
