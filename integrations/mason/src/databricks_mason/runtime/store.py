"""Runtime Store contracts and process-local implementation."""

from __future__ import annotations

import asyncio
import copy
import os
from typing import Protocol

from databricks_mason.runtime.types import (
    DurableEvent,
    DurableExecution,
    DurableExecutionStatus,
    DurableRequestConflictError,
    JsonObject,
    JsonValue,
)

DEFAULT_RUNTIME_SCHEMA = "databricks_mason_runtime"
RUNTIME_ENDPOINT_ENV = "DATABRICKS_MASON_RUNTIME_ENDPOINT"
RUNTIME_SCHEMA_ENV = "DATABRICKS_MASON_RUNTIME_SCHEMA"
RUNTIME_LOCAL_ENV = "DATABRICKS_MASON_RUNTIME_LOCAL"


def _validate_execution_id(execution_id: str) -> None:
    if not execution_id:
        raise ValueError("execution_id must not be empty")


class RuntimeStore(Protocol):
    """Invocation state operations shared by local and durable Runtime Stores."""

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def accept(self, execution_id: str, request: JsonValue) -> DurableExecution: ...

    async def get(self, execution_id: str) -> DurableExecution | None: ...

    async def claim(self, execution_id: str) -> DurableExecution | None:
        """Claim a newly queued invocation for its first attempt."""
        ...

    async def complete(
        self,
        execution_id: str,
        attempt: int,
        response: JsonValue,
    ) -> bool: ...

    async def fail(self, execution_id: str, attempt: int) -> bool: ...

    async def append_event(
        self,
        execution_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None: ...

    async def events(
        self,
        execution_id: str,
        after_sequence: int | None = None,
    ) -> list[DurableEvent]: ...


class InMemoryRuntimeStore(RuntimeStore):
    """Process-local Runtime Store for development and tests."""

    def __init__(self) -> None:
        self.states: dict[str, DurableExecution] = {}
        self.persisted_events: list[DurableEvent] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def accept(self, execution_id: str, request: JsonValue) -> DurableExecution:
        _validate_execution_id(execution_id)
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
        _validate_execution_id(execution_id)
        async with self._lock:
            state = self.states.get(execution_id)
            return copy.deepcopy(state) if state is not None else None

    async def claim(self, execution_id: str) -> DurableExecution | None:
        _validate_execution_id(execution_id)
        async with self._lock:
            state = self.states.get(execution_id)
            if state is None or state.status != DurableExecutionStatus.QUEUED:
                return None
            claimed = DurableExecution(
                execution_id=execution_id,
                status=DurableExecutionStatus.ACTIVE,
                attempt=state.attempt + 1,
                heartbeat_at=None,
                request=copy.deepcopy(state.request),
                response=None,
            )
            self.states[execution_id] = claimed
            self._append_event(execution_id, claimed.attempt, {"type": "run.started"})
            return copy.deepcopy(claimed)

    async def complete(self, execution_id: str, attempt: int, response: JsonValue) -> bool:
        async with self._lock:
            state = self.states.get(execution_id)
            if state is None or not self._owns_attempt(state, attempt):
                return False
            self.states[execution_id] = DurableExecution(
                execution_id=state.execution_id,
                status=DurableExecutionStatus.COMPLETED,
                attempt=state.attempt,
                heartbeat_at=None,
                request=state.request,
                response=copy.deepcopy(response),
            )
            self._append_event(execution_id, attempt, {"type": "run.completed"})
            return True

    async def fail(self, execution_id: str, attempt: int) -> bool:
        async with self._lock:
            state = self.states.get(execution_id)
            if state is None or not self._owns_attempt(state, attempt):
                return False
            self.states[execution_id] = DurableExecution(
                execution_id=state.execution_id,
                status=DurableExecutionStatus.FAILED,
                attempt=state.attempt,
                heartbeat_at=None,
                request=state.request,
                response=None,
            )
            self._append_event(execution_id, attempt, {"type": "run.failed"})
            return True

    async def append_event(
        self,
        execution_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None:
        async with self._lock:
            state = self.states.get(execution_id)
            if not self._owns_attempt(state, attempt):
                return None
            persisted = self._append_event(execution_id, attempt, event)
            return persisted.sequence_number

    async def events(
        self,
        execution_id: str,
        after_sequence: int | None = None,
    ) -> list[DurableEvent]:
        _validate_execution_id(execution_id)
        async with self._lock:
            return [
                copy.deepcopy(event)
                for event in self.persisted_events
                if event.execution_id == execution_id
                and (after_sequence is None or event.sequence_number > after_sequence)
            ]

    @staticmethod
    def _owns_attempt(state: DurableExecution | None, attempt: int) -> bool:
        return bool(
            state and state.status == DurableExecutionStatus.ACTIVE and state.attempt == attempt
        )

    def _append_event(self, execution_id: str, attempt: int, event: JsonObject) -> DurableEvent:
        persisted = DurableEvent(
            sequence_number=len(self.persisted_events) + 1,
            execution_id=execution_id,
            attempt=attempt,
            event=copy.deepcopy(event),
        )
        self.persisted_events.append(persisted)
        return persisted


def default_runtime_store() -> RuntimeStore:
    """Use the attached Lakebase resource when deployed, otherwise process-local state."""
    if os.getenv(RUNTIME_LOCAL_ENV, "").lower() == "true":
        return InMemoryRuntimeStore()
    app_name = os.getenv("DATABRICKS_APP_NAME")
    if not app_name:
        return InMemoryRuntimeStore()
    if endpoint := os.getenv(RUNTIME_ENDPOINT_ENV):
        from databricks_mason.lakebase_durability_store import get_lakebase_schema
        from databricks_mason.runtime.durability.lakebase_runtime_store import (
            LakebaseDurableRuntimeStore,
        )

        schema = os.getenv(RUNTIME_SCHEMA_ENV) or get_lakebase_schema(app_name)
        return LakebaseDurableRuntimeStore.from_app_resource(endpoint=endpoint, schema=schema)
    raise RuntimeError(
        f"{RUNTIME_ENDPOINT_ENV} is required for durable execution in Databricks Apps"
    )
