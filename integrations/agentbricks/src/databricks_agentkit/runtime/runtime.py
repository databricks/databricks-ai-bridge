"""Transport-neutral facade for Agent Bricks invocation execution."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from typing import TYPE_CHECKING

from databricks_agentkit.runtime.execution import (
    AttemptExecution,
    InvocationExecutor,
    LocalInvocationExecutor,
    copy_json_value,
)
from databricks_agentkit.runtime.store import (
    InMemoryRuntimeStore,
    RuntimeStore,
    runtime_store_from_environment,
)
from databricks_agentkit.runtime.types import (
    Invocation,
    InvocationEvent,
    InvocationExecutorFn,
    InvocationFailedError,
    InvocationNotFoundError,
    InvocationStatus,
    JsonValue,
)

if TYPE_CHECKING:
    from databricks_agentkit.runtime.durability.store import DurableRuntimeStore


class Runtime:
    """Coordinate one Agent Bricks process's invocation lifecycle.

    A Runtime Store owns invocation state: requests, status, events, and results. An invocation
    executor turns queued state into agent attempts. This facade accepts idempotent requests,
    delegates scheduling to the executor, and polls the store for foreground results or event replay.
    Use :meth:`local` for process-local execution, :meth:`durable` for a durable store,
    :meth:`from_store` for an explicit store, or :meth:`from_environment` when Agent Bricks configures the
    store through the process environment.
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
        from databricks_agentkit.runtime.durability.store import DurableRuntimeStore

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
        runtime_store: DurableRuntimeStore,
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
        from databricks_agentkit.runtime.durability.execution import DurableInvocationExecutor
        from databricks_agentkit.runtime.durability.store import DurableRuntimeStore

        if not isinstance(runtime_store, DurableRuntimeStore):
            raise TypeError("Runtime.durable requires a DurableRuntimeStore")
        execution = AttemptExecution(execute_fn, runtime_store=runtime_store)
        return cls(
            runtime_store=runtime_store,
            executor=DurableInvocationExecutor(
                execution,
                runtime_store=runtime_store,
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
        runtime_store: RuntimeStore,
        recovery_enabled: Callable[[], bool] | None = None,
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 1.0,
    ) -> Runtime:
        """Construct the local or durable Runtime required by the selected Runtime Store."""
        from databricks_agentkit.runtime.durability.store import DurableRuntimeStore

        if isinstance(runtime_store, DurableRuntimeStore):
            return cls.durable(
                execute_fn,
                runtime_store=runtime_store,
                recovery_enabled=recovery_enabled,
                heartbeat_seconds=heartbeat_seconds,
                stale_seconds=stale_seconds,
                scan_seconds=scan_seconds,
                poll_seconds=poll_seconds,
            )
        return cls.local(execute_fn, runtime_store=runtime_store, poll_seconds=poll_seconds)

    @classmethod
    def from_environment(
        cls,
        execute_fn: InvocationExecutorFn,
        *,
        recovery_enabled: Callable[[], bool] | None = None,
        heartbeat_seconds: float = 3.0,
        stale_seconds: float = 10.0,
        scan_seconds: float = 3.0,
        poll_seconds: float = 1.0,
    ) -> Runtime:
        """Construct the Runtime selected by the Agent Bricks process environment."""
        return cls.from_store(
            execute_fn,
            runtime_store=runtime_store_from_environment(),
            recovery_enabled=recovery_enabled,
            heartbeat_seconds=heartbeat_seconds,
            stale_seconds=stale_seconds,
            scan_seconds=scan_seconds,
            poll_seconds=poll_seconds,
        )

    @property
    def is_durable(self) -> bool:
        """Whether this Runtime has heartbeat and recovery capabilities."""
        from databricks_agentkit.runtime.durability.store import DurableRuntimeStore

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

    async def submit(
        self,
        invocation_id: str,
        request: JsonValue,
        *,
        session_id: str | None = None,
    ) -> Invocation:
        """Accept an idempotent invocation and schedule it on this worker.

        The Runtime Store makes ``invocation_id`` the idempotency key. Retrying the same request
        and session returns its existing state; a different request or session for the same ID
        raises :class:`InvocationConflictError`.
        """
        self._require_started()
        if not invocation_id:
            raise ValueError("invocation_id must not be empty")
        copied_request = copy_json_value(request, "request")
        # Preserve compatibility with store adapters that predate session-aware submission.
        if session_id is None:
            state = await self.runtime_store.accept(invocation_id, copied_request)
        else:
            state = await self.runtime_store.accept(
                invocation_id,
                copied_request,
                session_id=session_id,
            )
        self.executor.ensure_scheduled(state)
        return state

    async def invoke(
        self,
        invocation_id: str,
        request: JsonValue,
        *,
        session_id: str | None = None,
        timeout: float | None = None,
    ) -> JsonValue:
        """Accept an invocation and wait for its terminal result."""
        await self.submit(invocation_id, request, session_id=session_id)
        return await self.wait(invocation_id, timeout=timeout)

    async def get_invocation(
        self,
        invocation_id: str | None = None,
        *,
        session_id: str | None = None,
    ) -> Invocation | None:
        """Return invocation or session state and schedule queued first-attempt work."""
        self._require_started()
        self._validate_read_scope(invocation_id, session_id)
        if session_id is None:
            state = await self.runtime_store.get(invocation_id)
        else:
            state = await self.runtime_store.get(session_id=session_id)
        if state is not None:
            self.executor.ensure_scheduled(state)
        return state

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
        invocation_id: str | None = None,
        *,
        session_id: str | None = None,
        after_sequence: int | None = None,
    ) -> list[InvocationEvent]:
        """Return invocation or session events after an optional exclusive replay cursor."""
        self._require_started()
        self._validate_read_scope(invocation_id, session_id)
        if session_id is None:
            return await self.runtime_store.events(invocation_id, after_sequence)
        return await self.runtime_store.events(
            session_id=session_id,
            after_sequence=after_sequence,
        )

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Runtime.start() must be called first")

    @staticmethod
    def _validate_read_scope(invocation_id: str | None, session_id: str | None) -> None:
        if (invocation_id is None) == (session_id is None):
            raise ValueError("exactly one of invocation_id or session_id must be provided")
