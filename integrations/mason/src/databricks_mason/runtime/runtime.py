"""Transport-neutral facade for Mason invocation execution."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable

from databricks_mason.runtime.execution import (
    AttemptExecution,
    InvocationExecutor,
    LocalInvocationExecutor,
    copy_json_value,
)
from databricks_mason.runtime.store import (
    InMemoryRuntimeStore,
    RuntimeStore,
    default_runtime_store,
)
from databricks_mason.runtime.types import (
    Invocation,
    InvocationEvent,
    InvocationExecutorFn,
    InvocationFailedError,
    InvocationNotFoundError,
    InvocationStatus,
    JsonValue,
)


class Runtime:
    """Coordinate one Mason process's invocation lifecycle.

    A Runtime Store owns invocation state: requests, status, events, and results. An invocation
    executor turns queued state into agent attempts. This facade accepts idempotent requests,
    delegates scheduling to the executor, and polls the store for foreground results or event replay.
    Use :meth:`local` for process-local execution, :meth:`durable` for a Lakebase-backed store, or
    :meth:`from_store` when the environment chooses the store implementation.
    """

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
        self.executor = executor
        self.poll_seconds = poll_seconds
        self._started = False

    @classmethod
    def local(
        cls,
        execute_fn: InvocationExecutorFn,
        *,
        runtime_store: RuntimeStore | None = None,
        poll_seconds: float = 1.0,
    ) -> Runtime:
        """Construct a Runtime that schedules work only in the current process.

        The default :class:`InMemoryRuntimeStore` supports foreground, background, streaming,
        polling, and event replay until the process exits. It does not heartbeat or recover work.
        """
        store = InMemoryRuntimeStore() if runtime_store is None else runtime_store
        from databricks_mason.runtime.durability.store import DurableRuntimeStore

        if isinstance(store, DurableRuntimeStore):
            raise TypeError("Runtime.local requires a non-durable RuntimeStore")
        execution = AttemptExecution(execute_fn, runtime_store=store)
        return cls(
            runtime_store=store,
            executor=LocalInvocationExecutor(execution, runtime_store=store),
            poll_seconds=poll_seconds,
        )

    @classmethod
    def durable(
        cls,
        execute_fn: InvocationExecutorFn,
        *,
        runtime_store: RuntimeStore | None = None,
        recovery_enabled: Callable[[], bool] | None = None,
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 1.0,
    ) -> Runtime:
        """Construct a Runtime that heartbeats and can recover work from a durable store.

        A durable runtime requires a :class:`DurableRuntimeStore`. Its results and events persist
        across process restarts. ``recovery_enabled`` controls only stale-work recovery; it does not
        change whether normal invocation state is persisted.
        """
        from databricks_mason.runtime.durability.execution import DurableInvocationExecutor
        from databricks_mason.runtime.durability.store import DurableRuntimeStore

        store = default_runtime_store() if runtime_store is None else runtime_store
        if not isinstance(store, DurableRuntimeStore):
            raise TypeError("Runtime.durable requires a DurableRuntimeStore")
        execution = AttemptExecution(execute_fn, runtime_store=store)
        return cls(
            runtime_store=store,
            executor=DurableInvocationExecutor(
                execution,
                runtime_store=store,
                recovery_enabled=recovery_enabled or (lambda: True),
                heartbeat_seconds=heartbeat_seconds,
                stale_seconds=stale_seconds,
                scan_seconds=scan_seconds,
            ),
            poll_seconds=poll_seconds,
        )

    @classmethod
    def from_store(
        cls,
        execute_fn: InvocationExecutorFn,
        *,
        runtime_store: RuntimeStore | None = None,
        recovery_enabled: Callable[[], bool] | None = None,
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 1.0,
    ) -> Runtime:
        """Construct the local or durable Runtime required by the selected Runtime Store."""
        store = default_runtime_store() if runtime_store is None else runtime_store
        from databricks_mason.runtime.durability.store import DurableRuntimeStore

        if isinstance(store, DurableRuntimeStore):
            return cls.durable(
                execute_fn,
                runtime_store=store,
                recovery_enabled=recovery_enabled,
                heartbeat_seconds=heartbeat_seconds,
                stale_seconds=stale_seconds,
                scan_seconds=scan_seconds,
                poll_seconds=poll_seconds,
            )
        return cls.local(execute_fn, runtime_store=store, poll_seconds=poll_seconds)

    @property
    def is_durable(self) -> bool:
        """Whether this Runtime has heartbeat and recovery capabilities."""
        from databricks_mason.runtime.durability.store import DurableRuntimeStore

        return isinstance(self.runtime_store, DurableRuntimeStore)

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

    async def submit(self, invocation_id: str, request: JsonValue) -> Invocation:
        """Accept an idempotent invocation and schedule it on this worker.

        The Runtime Store makes ``invocation_id`` the idempotency key. Retrying the same request
        returns its existing state; a different request for the same ID raises
        :class:`InvocationConflictError`.
        """
        self._require_started()
        if not invocation_id:
            raise ValueError("invocation_id must not be empty")
        state = await self.runtime_store.accept(
            invocation_id,
            copy_json_value(request, "request"),
        )
        self.executor.ensure_scheduled(state)
        return state

    async def invoke(
        self,
        invocation_id: str,
        request: JsonValue,
        *,
        timeout: float | None = None,
    ) -> JsonValue:
        """Accept an invocation and wait for its terminal result."""
        await self.submit(invocation_id, request)
        return await self.wait(invocation_id, timeout=timeout)

    async def get_invocation(self, invocation_id: str) -> Invocation | None:
        """Return the current state for an invocation ID."""
        self._require_started()
        return await self.runtime_store.get(invocation_id)

    async def wait(
        self,
        invocation_id: str,
        *,
        timeout: float | None = None,
    ) -> JsonValue:
        """Poll until an invocation completes, fails, or reaches the optional timeout."""
        self._require_started()

        async def poll() -> JsonValue:
            while True:
                state = await self.get_invocation(invocation_id)
                if state is None:
                    raise InvocationNotFoundError(invocation_id)
                if state.status == InvocationStatus.COMPLETED:
                    return copy.deepcopy(state.response)
                if state.status == InvocationStatus.FAILED:
                    raise InvocationFailedError(invocation_id)
                await asyncio.sleep(self.poll_seconds)

        if timeout is None:
            return await poll()
        try:
            return await asyncio.wait_for(poll(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError from exc

    async def get_events(
        self,
        invocation_id: str,
        *,
        after_sequence: int | None = None,
    ) -> list[InvocationEvent]:
        """Return persisted invocation events after an optional exclusive replay cursor."""
        self._require_started()
        return await self.runtime_store.events(invocation_id, after_sequence)

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Runtime.start() must be called first")
