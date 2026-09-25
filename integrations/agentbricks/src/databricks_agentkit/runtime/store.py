"""Runtime Store contracts and process-local implementation."""

from __future__ import annotations

import asyncio
import copy
import os
from typing import Protocol

from databricks_agentkit.runtime.types import (
    Invocation,
    InvocationConflictError,
    InvocationEvent,
    InvocationStatus,
    JsonObject,
    JsonValue,
)

DEFAULT_RUNTIME_STORE_SCHEMA = "databricks_agentkit_runtime"
RUNTIME_STORE_LAKEBASE_BRANCH_ENV = "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LAKEBASE_BRANCH"
RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV = "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LAKEBASE_ENDPOINT"
RUNTIME_STORE_DATABASE_ENV = "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_DATABASE"
RUNTIME_STORE_USERNAME_ENV = "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_USERNAME"
RUNTIME_STORE_SCHEMA_ENV = "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_SCHEMA"
RUNTIME_STORE_LOCAL_ENV = "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL"


def _validate_invocation_id(invocation_id: str) -> None:
    if not invocation_id:
        raise ValueError("invocation_id must not be empty")


class RuntimeStore(Protocol):
    """Invocation state operations shared by local and durable Runtime Stores."""

    async def initialize(self) -> None:
        """Open the store and create any resources it owns."""
        ...

    async def close(self) -> None:
        """Release the store's connections and other resources."""
        ...

    async def accept(self, invocation_id: str, request: JsonValue) -> Invocation:
        """Create an invocation or return the identical request accepted for this ID.

        Reusing an invocation ID with a different request raises :class:`InvocationConflictError`.
        """
        ...

    async def get(self, invocation_id: str) -> Invocation | None:
        """Return the latest invocation state, or ``None`` for an unknown ID."""
        ...

    async def claim(self, invocation_id: str) -> Invocation | None:
        """Claim a newly queued invocation for its first attempt."""
        ...

    async def complete(
        self,
        invocation_id: str,
        attempt: int,
        response: JsonValue,
    ) -> bool:
        """Persist a result when this attempt still owns the active invocation."""
        ...

    async def fail(self, invocation_id: str, attempt: int) -> bool:
        """Mark an invocation failed when this attempt still owns it."""
        ...

    async def append_event(
        self,
        invocation_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None:
        """Append an event for an owned attempt and return its replay cursor."""
        ...

    async def events(
        self,
        invocation_id: str,
        after_sequence: int | None = None,
    ) -> list[InvocationEvent]:
        """Return invocation events after the optional exclusive replay cursor."""
        ...


class InMemoryRuntimeStore(RuntimeStore):
    """Process-local Runtime Store for development and tests."""

    def __init__(self) -> None:
        self.states: dict[str, Invocation] = {}
        self.persisted_events: list[InvocationEvent] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def accept(self, invocation_id: str, request: JsonValue) -> Invocation:
        _validate_invocation_id(invocation_id)
        async with self._lock:
            existing = self.states.get(invocation_id)
            if existing is not None:
                if existing.request != request:
                    raise InvocationConflictError(invocation_id)
                return copy.deepcopy(existing)
            state = Invocation(
                invocation_id=invocation_id,
                status=InvocationStatus.QUEUED,
                attempt=0,
                request=copy.deepcopy(request),
                response=None,
            )
            self.states[invocation_id] = state
            return copy.deepcopy(state)

    async def get(self, invocation_id: str) -> Invocation | None:
        _validate_invocation_id(invocation_id)
        async with self._lock:
            state = self.states.get(invocation_id)
            return copy.deepcopy(state) if state is not None else None

    async def claim(self, invocation_id: str) -> Invocation | None:
        _validate_invocation_id(invocation_id)
        async with self._lock:
            state = self.states.get(invocation_id)
            if state is None or state.status != InvocationStatus.QUEUED:
                return None
            claimed = Invocation(
                invocation_id=invocation_id,
                status=InvocationStatus.ACTIVE,
                attempt=state.attempt + 1,
                request=copy.deepcopy(state.request),
                response=None,
            )
            self.states[invocation_id] = claimed
            self._append_event(invocation_id, claimed.attempt, {"type": "run.started"})
            return copy.deepcopy(claimed)

    async def complete(self, invocation_id: str, attempt: int, response: JsonValue) -> bool:
        async with self._lock:
            state = self.states.get(invocation_id)
            if state is None or not self._owns_attempt(state, attempt):
                return False
            self.states[invocation_id] = Invocation(
                invocation_id=state.invocation_id,
                status=InvocationStatus.COMPLETED,
                attempt=state.attempt,
                request=state.request,
                response=copy.deepcopy(response),
            )
            self._append_event(invocation_id, attempt, {"type": "run.completed"})
            return True

    async def fail(self, invocation_id: str, attempt: int) -> bool:
        async with self._lock:
            state = self.states.get(invocation_id)
            if state is None or not self._owns_attempt(state, attempt):
                return False
            self.states[invocation_id] = Invocation(
                invocation_id=state.invocation_id,
                status=InvocationStatus.FAILED,
                attempt=state.attempt,
                request=state.request,
                response=None,
            )
            self._append_event(invocation_id, attempt, {"type": "run.failed"})
            return True

    async def append_event(
        self,
        invocation_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None:
        async with self._lock:
            state = self.states.get(invocation_id)
            if not self._owns_attempt(state, attempt):
                return None
            persisted = self._append_event(invocation_id, attempt, event)
            return persisted.sequence_number

    async def events(
        self,
        invocation_id: str,
        after_sequence: int | None = None,
    ) -> list[InvocationEvent]:
        _validate_invocation_id(invocation_id)
        async with self._lock:
            return [
                copy.deepcopy(event)
                for event in self.persisted_events
                if event.invocation_id == invocation_id
                and (after_sequence is None or event.sequence_number > after_sequence)
            ]

    @staticmethod
    def _owns_attempt(state: Invocation | None, attempt: int) -> bool:
        return bool(state and state.status == InvocationStatus.ACTIVE and state.attempt == attempt)

    def _append_event(self, invocation_id: str, attempt: int, event: JsonObject) -> InvocationEvent:
        persisted = InvocationEvent(
            sequence_number=len(self.persisted_events) + 1,
            invocation_id=invocation_id,
            attempt=attempt,
            event=copy.deepcopy(event),
        )
        self.persisted_events.append(persisted)
        return persisted


def runtime_store_is_persistent_environment() -> bool:
    """Whether the process environment selects a complete durable Runtime Store."""
    if os.getenv(RUNTIME_STORE_LOCAL_ENV, "").lower() == "true":
        return False
    managed = (
        os.getenv(RUNTIME_STORE_LAKEBASE_BRANCH_ENV),
        os.getenv(RUNTIME_STORE_DATABASE_ENV),
        os.getenv(RUNTIME_STORE_USERNAME_ENV),
    )
    if any(managed):
        return all(managed)
    legacy = (
        os.getenv(RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV),
        os.getenv(RUNTIME_STORE_SCHEMA_ENV),
    )
    return all(legacy)


def runtime_store_from_environment() -> RuntimeStore:
    """Construct the Runtime Store selected by the process environment.

    The local marker overrides any inherited Lakebase variables. Managed Runtime Stores provide a
    branch, database, and username; legacy Apps resources provide an endpoint plus ``PG*`` values.
    An incomplete durable configuration is rejected rather than mixed with another resource.
    """
    if os.getenv(RUNTIME_STORE_LOCAL_ENV, "").lower() == "true":
        return InMemoryRuntimeStore()

    branch = os.getenv(RUNTIME_STORE_LAKEBASE_BRANCH_ENV)
    database = os.getenv(RUNTIME_STORE_DATABASE_ENV)
    username = os.getenv(RUNTIME_STORE_USERNAME_ENV)
    managed = {
        RUNTIME_STORE_LAKEBASE_BRANCH_ENV: branch,
        RUNTIME_STORE_DATABASE_ENV: database,
        RUNTIME_STORE_USERNAME_ENV: username,
    }
    if any(managed.values()):
        missing = [name for name, value in managed.items() if not value]
        if missing:
            raise RuntimeError(
                "Managed Runtime Store configuration is missing: " + ", ".join(missing)
            )
        from databricks_agentkit.runtime.durability.lakebase_runtime_store import (
            LakebaseDurableRuntimeStore,
        )

        assert branch is not None
        assert database is not None
        assert username is not None
        schema = os.getenv(RUNTIME_STORE_SCHEMA_ENV) or DEFAULT_RUNTIME_STORE_SCHEMA
        return LakebaseDurableRuntimeStore.from_managed_runtime_store(
            branch=branch,
            database=database,
            username=username,
            schema=schema,
        )

    endpoint = os.getenv(RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV)
    schema = os.getenv(RUNTIME_STORE_SCHEMA_ENV)
    if endpoint and schema:
        from databricks_agentkit.runtime.durability.lakebase_runtime_store import (
            LakebaseDurableRuntimeStore,
        )

        return LakebaseDurableRuntimeStore.from_app_resource(endpoint=endpoint, schema=schema)
    if endpoint or schema:
        raise RuntimeError(
            f"{RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV} and {RUNTIME_STORE_SCHEMA_ENV} must be set "
            "together"
        )
    return InMemoryRuntimeStore()
