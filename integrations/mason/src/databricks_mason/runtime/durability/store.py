"""Runtime Store capabilities required for heartbeat-based recovery."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from databricks_mason.runtime.store import (
    InMemoryRuntimeStore,
    RuntimeStore,
    default_runtime_store,
)
from databricks_mason.runtime.types import DurableExecution, DurableExecutionStatus


@runtime_checkable
class DurableRuntimeStore(RuntimeStore, Protocol):
    """Extend Runtime Store operations with durable recovery coordination."""

    async def recoverable_execution_ids(self, stale_seconds: float) -> list[str]:
        """List queued executions and active executions whose heartbeat is stale."""
        ...

    async def claim_recoverable(
        self,
        execution_id: str,
        stale_seconds: float,
    ) -> DurableExecution | None:
        """Claim queued or stale work and return its new active attempt."""
        ...

    async def heartbeat(self, execution_id: str, attempt: int) -> bool:
        """Refresh an active attempt's lease while the caller owns it."""
        ...


class InMemoryDurabilityStore(InMemoryRuntimeStore, DurableRuntimeStore):
    """Process-local durable store retained by the existing AgentApp interface."""

    async def recoverable_execution_ids(self, stale_seconds: float) -> list[str]:
        async with self._lock:
            return [
                execution_id
                for execution_id, state in self.states.items()
                if self._is_recoverable(state, stale_seconds)
            ]

    async def claim_recoverable(
        self,
        execution_id: str,
        stale_seconds: float,
    ) -> DurableExecution | None:
        async with self._lock:
            state = self.states.get(execution_id)
            if state is None or not self._is_recoverable(state, stale_seconds):
                return None
            claimed = DurableExecution(
                execution_id=execution_id,
                status=DurableExecutionStatus.ACTIVE,
                attempt=state.attempt + 1,
                heartbeat_at=datetime.now(timezone.utc),
                request=copy.deepcopy(state.request),
                response=None,
            )
            self.states[execution_id] = claimed
            self._append_event(execution_id, claimed.attempt, {"type": "run.started"})
            return copy.deepcopy(claimed)

    async def heartbeat(self, execution_id: str, attempt: int) -> bool:
        async with self._lock:
            state = self.states.get(execution_id)
            if state is None or not self._owns_attempt(state, attempt):
                return False
            self.states[execution_id] = DurableExecution(
                execution_id=state.execution_id,
                status=state.status,
                attempt=state.attempt,
                heartbeat_at=datetime.now(timezone.utc),
                request=state.request,
                response=state.response,
            )
            return True

    @staticmethod
    def _is_recoverable(state: DurableExecution, stale_seconds: float) -> bool:
        if state.status == DurableExecutionStatus.QUEUED:
            return True
        if state.status != DurableExecutionStatus.ACTIVE or state.heartbeat_at is None:
            return state.status == DurableExecutionStatus.ACTIVE
        heartbeat_at = state.heartbeat_at
        if heartbeat_at.tzinfo is None:
            heartbeat_at = heartbeat_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - heartbeat_at >= timedelta(seconds=stale_seconds)


def default_durability_store() -> RuntimeStore:
    """Select the deployed durable store or the existing process-local durable test store."""
    store = default_runtime_store()
    if isinstance(store, InMemoryRuntimeStore):
        return InMemoryDurabilityStore()
    return store
