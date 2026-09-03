"""In-memory durability store for local development and tests."""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timedelta, timezone

from databricks_mason.runtime.types import (
    DurableEvent,
    DurableExecution,
    DurableExecutionStatus,
    DurableRequestConflictError,
    JsonObject,
)


class InMemoryDurabilityStore:
    """Process-local implementation of the durability store contract."""

    persistent = False

    def __init__(self) -> None:
        self.states: dict[str, DurableExecution] = {}
        self.persisted_events: list[DurableEvent] = []
        self.initialized = False
        self.closed = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.initialized = True
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def accept(self, execution_id: str, request: JsonObject) -> DurableExecution:
        self._validate_execution_id(execution_id)
        async with self._lock:
            existing = self.states.get(execution_id)
            if existing is not None:
                if existing.request != request:
                    raise DurableRequestConflictError(execution_id)
                return copy.deepcopy(existing)
            state = DurableExecution(
                execution_id=execution_id,
                status=DurableExecutionStatus.QUEUED,
                attempt=0,
                heartbeat_at=None,
                request=copy.deepcopy(request),
                response=None,
            )
            self.states[execution_id] = state
            return copy.deepcopy(state)

    async def get(self, execution_id: str) -> DurableExecution | None:
        self._validate_execution_id(execution_id)
        async with self._lock:
            state = self.states.get(execution_id)
            return copy.deepcopy(state) if state is not None else None

    async def recoverable_execution_ids(self, stale_seconds: float) -> list[str]:
        async with self._lock:
            return [
                execution_id
                for execution_id, state in self.states.items()
                if self._is_recoverable(state, stale_seconds)
            ]

    async def claim(
        self,
        execution_id: str,
        stale_seconds: float,
    ) -> DurableExecution | None:
        self._validate_execution_id(execution_id)
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
            return copy.deepcopy(claimed)

    async def heartbeat(self, execution_id: str, attempt: int) -> bool:
        async with self._lock:
            state = self.states.get(execution_id)
            if (
                state is None
                or state.status != DurableExecutionStatus.ACTIVE
                or state.attempt != attempt
            ):
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

    async def complete(self, execution_id: str, attempt: int, response: JsonObject) -> bool:
        async with self._lock:
            state = self.states.get(execution_id)
            if (
                state is None
                or state.status != DurableExecutionStatus.ACTIVE
                or state.attempt != attempt
            ):
                return False
            self.states[execution_id] = DurableExecution(
                execution_id=state.execution_id,
                status=DurableExecutionStatus.COMPLETED,
                attempt=state.attempt,
                heartbeat_at=state.heartbeat_at,
                request=state.request,
                response=copy.deepcopy(response),
            )
            return True

    async def fail(self, execution_id: str, attempt: int) -> bool:
        async with self._lock:
            state = self.states.get(execution_id)
            if (
                state is None
                or state.status != DurableExecutionStatus.ACTIVE
                or state.attempt != attempt
            ):
                return False
            self.states[execution_id] = DurableExecution(
                execution_id=state.execution_id,
                status=DurableExecutionStatus.FAILED,
                attempt=state.attempt,
                heartbeat_at=state.heartbeat_at,
                request=state.request,
                response=None,
            )
            return True

    async def append_event(
        self,
        execution_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None:
        async with self._lock:
            state = self.states.get(execution_id)
            if (
                state is None
                or state.status != DurableExecutionStatus.ACTIVE
                or state.attempt != attempt
            ):
                return None
            persisted = DurableEvent(
                sequence_number=len(self.persisted_events) + 1,
                execution_id=execution_id,
                attempt=attempt,
                event=copy.deepcopy(event),
            )
            self.persisted_events.append(persisted)
            return persisted.sequence_number

    async def events(
        self,
        execution_id: str,
        after_sequence: int | None = None,
    ) -> list[DurableEvent]:
        self._validate_execution_id(execution_id)
        async with self._lock:
            return [
                copy.deepcopy(event)
                for event in self.persisted_events
                if event.execution_id == execution_id
                and (after_sequence is None or event.sequence_number > after_sequence)
            ]

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

    @staticmethod
    def _validate_execution_id(execution_id: str) -> None:
        if not execution_id:
            raise ValueError("execution_id must not be empty")
