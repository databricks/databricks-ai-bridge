"""Public types for durable request execution."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

JsonObject = dict[str, Any]
DurableEventEmitter = Callable[[JsonObject], Awaitable[int]]


class DurableExecutionStatus(str, Enum):
    """Lifecycle states stored by the durability layer."""

    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DurableExecution:
    """Persisted state for one idempotent request."""

    execution_id: str
    status: DurableExecutionStatus
    attempt: int
    heartbeat_at: datetime | None
    request: JsonObject
    response: JsonObject | None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            DurableExecutionStatus.COMPLETED,
            DurableExecutionStatus.FAILED,
        }


@dataclass(frozen=True)
class DurableEvent:
    """One persisted event emitted by a durable execution attempt."""

    sequence_number: int
    execution_id: str
    attempt: int
    event: JsonObject


@dataclass(frozen=True)
class DurableExecutionContext:
    """Attempt metadata passed to the caller-owned executor."""

    execution_id: str
    attempt: int
    _emit: DurableEventEmitter | None = field(default=None, repr=False, compare=False)

    @property
    def is_recovery(self) -> bool:
        return self.attempt > 1

    async def emit(self, event: JsonObject) -> int:
        """Persist an ordered event and return its replay cursor."""
        if self._emit is None:
            raise RuntimeError("event emission is not available for this execution context")
        return await self._emit(event)


DurableExecutor = Callable[[JsonObject, DurableExecutionContext], Awaitable[JsonObject]]


class DurabilityStore(Protocol):
    """Persistence contract used by :class:`DatabricksDurableRuntime`."""

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def accept(self, execution_id: str, request: JsonObject) -> DurableExecution: ...

    async def get(self, execution_id: str) -> DurableExecution | None: ...

    async def recoverable_execution_ids(self, stale_seconds: float) -> list[str]: ...

    async def claim(
        self,
        execution_id: str,
        stale_seconds: float,
    ) -> DurableExecution | None: ...

    async def heartbeat(self, execution_id: str, attempt: int) -> bool: ...

    async def complete(
        self,
        execution_id: str,
        attempt: int,
        response: JsonObject,
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


class DurableRequestConflictError(ValueError):
    """Raised when an execution ID is reused with a different request."""


class DurableExecutionNotFoundError(LookupError):
    """Raised when waiting for an unknown execution ID."""


class DurableExecutionFailedError(RuntimeError):
    """Raised when a durable execution reaches the failed state."""
