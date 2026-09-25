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


def _validate_session_id(session_id: str) -> None:
    if not session_id:
        raise ValueError("session_id must not be empty")


def _validate_read_scope(invocation_id: str | None, session_id: str | None) -> None:
    if (invocation_id is None) == (session_id is None):
        raise ValueError("exactly one of invocation_id or session_id must be provided")
    if invocation_id is not None:
        _validate_invocation_id(invocation_id)
    if session_id is not None:
        _validate_session_id(session_id)


class RuntimeStore(Protocol):
    """Invocation state operations shared by local and durable Runtime Stores."""

    async def initialize(self) -> None:
        """Open the store and create any resources it owns."""
        ...

    async def close(self) -> None:
        """Release the store's connections and other resources."""
        ...

    async def accept(
        self,
        invocation_id: str,
        request: JsonValue,
        session_id: str | None = None,
    ) -> Invocation:
        """Create an invocation or return the identical request accepted for this ID.

        Reusing an invocation ID with a different request or session raises
        :class:`InvocationConflictError`.
        """
        ...

    async def get(
        self,
        invocation_id: str | None = None,
        session_id: str | None = None,
    ) -> Invocation | None:
        """Return state for one invocation or the active/next invocation in a session."""
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
        invocation_id: str | None = None,
        after_sequence: int | None = None,
        session_id: str | None = None,
    ) -> list[InvocationEvent]:
        """Return invocation or session events after the optional exclusive replay cursor."""
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

    async def accept(
        self,
        invocation_id: str,
        request: JsonValue,
        session_id: str | None = None,
    ) -> Invocation:
        _validate_invocation_id(invocation_id)
        if session_id is not None:
            _validate_session_id(session_id)
        async with self._lock:
            existing = self.states.get(invocation_id)
            if existing is not None:
                if existing.request != request or existing.session_id != session_id:
                    raise InvocationConflictError(invocation_id)
                return copy.deepcopy(existing)
            queue_order = None
            if session_id is not None:
                queue_order = (
                    max(
                        (
                            state.queue_order or 0
                            for state in self.states.values()
                            if state.session_id == session_id
                        ),
                        default=0,
                    )
                    + 1
                )
            state = Invocation(
                invocation_id=invocation_id,
                status=InvocationStatus.QUEUED,
                attempt=0,
                request=copy.deepcopy(request),
                response=None,
                session_id=session_id,
                queue_order=queue_order,
            )
            self.states[invocation_id] = state
            return copy.deepcopy(state)

    async def get(
        self,
        invocation_id: str | None = None,
        session_id: str | None = None,
    ) -> Invocation | None:
        _validate_read_scope(invocation_id, session_id)
        async with self._lock:
            if invocation_id is not None:
                state = self.states.get(invocation_id)
            else:
                candidates = [
                    state
                    for state in self.states.values()
                    if state.session_id == session_id
                    and state.status in {InvocationStatus.ACTIVE, InvocationStatus.QUEUED}
                ]
                state = min(
                    candidates,
                    key=lambda candidate: (
                        0 if candidate.status == InvocationStatus.ACTIVE else 1,
                        candidate.queue_order or 0,
                    ),
                    default=None,
                )
            return copy.deepcopy(state) if state is not None else None

    async def claim(self, invocation_id: str) -> Invocation | None:
        _validate_invocation_id(invocation_id)
        async with self._lock:
            state = self.states.get(invocation_id)
            if state is None or state.status != InvocationStatus.QUEUED:
                return None
            if state.session_id is not None:
                session_states = [
                    candidate
                    for candidate in self.states.values()
                    if candidate.session_id == state.session_id
                ]
                if any(candidate.status == InvocationStatus.ACTIVE for candidate in session_states):
                    return None
                queued = [
                    candidate
                    for candidate in session_states
                    if candidate.status == InvocationStatus.QUEUED
                ]
                next_queued = min(queued, key=lambda candidate: candidate.queue_order or 0)
                if next_queued.invocation_id != invocation_id:
                    return None
            claimed = Invocation(
                invocation_id=invocation_id,
                status=InvocationStatus.ACTIVE,
                attempt=state.attempt + 1,
                request=copy.deepcopy(state.request),
                response=None,
                session_id=state.session_id,
                queue_order=state.queue_order,
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
                session_id=state.session_id,
                queue_order=state.queue_order,
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
                session_id=state.session_id,
                queue_order=state.queue_order,
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
        invocation_id: str | None = None,
        after_sequence: int | None = None,
        session_id: str | None = None,
    ) -> list[InvocationEvent]:
        _validate_read_scope(invocation_id, session_id)
        async with self._lock:
            return [
                copy.deepcopy(event)
                for event in self.persisted_events
                if (
                    event.invocation_id == invocation_id
                    if invocation_id is not None
                    else self.states[event.invocation_id].session_id == session_id
                )
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
