"""Public contracts shared by Mason Runtime, stores, and agent applications.

The runtime returns ``DurableExecution`` snapshots and ordered ``DurableEvent`` records. Each
executor call receives a ``DurableExecutionContext`` for attempt fencing and event emission.
``AgentApp`` adapts that lower-level context into
``DurableAgentContext`` for functions registered with ``@app.invoke`` and
``@app.on_recovery``.

All request, response, and event payloads use the recursive ``JsonValue`` / ``JsonObject`` aliases,
so Runtime Store values can be persisted identically by in-memory and Lakebase implementations.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
DurableEventEmitter = Callable[[JsonObject], Awaitable[int]]


class DurableExecutionStatus(str, Enum):
    """Lifecycle states stored by Mason Runtime."""

    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DurableExecution:
    """A store snapshot for one idempotent execution and its current owning attempt."""

    execution_id: str
    status: DurableExecutionStatus
    attempt: int
    heartbeat_at: datetime | None
    request: JsonValue
    response: JsonValue

    @property
    def is_terminal(self) -> bool:
        """Whether this execution can no longer transition or emit events."""
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
    """Attempt metadata and event emission passed to the runtime's executor function."""

    execution_id: str
    attempt: int
    _emit: DurableEventEmitter | None = field(default=None, repr=False, compare=False)

    @property
    def is_recovery(self) -> bool:
        """Whether this is a replacement attempt after an earlier worker stopped heartbeating."""
        return self.attempt > 1

    async def emit(self, event: JsonObject) -> int:
        """Persist an ordered event and return its replay cursor."""
        if self._emit is None:
            raise RuntimeError("event emission is not available for this execution context")
        return await self._emit(event)


DurableExecutorFn = Callable[[JsonValue, DurableExecutionContext], Awaitable[JsonValue]]


@dataclass(frozen=True)
class DurableAgentContext:
    """Invocation/session metadata and event emission for a decorated agent function."""

    invocation_id: str
    session_id: str
    attempt: int
    _execution_context: DurableExecutionContext = field(repr=False, compare=False)

    @property
    def is_recovery(self) -> bool:
        """Whether ``@app.on_recovery`` is handling a replacement attempt."""
        return self.attempt > 1

    async def emit(self, event: JsonObject) -> int:
        """Persist an ordered application event and return its replay cursor."""
        return await self._execution_context.emit(event)


DurableAgentHook = Callable[[JsonValue, DurableAgentContext], Awaitable[JsonValue]]


class DurableRequestConflictError(ValueError):
    """Raised when an execution ID is reused with a different request."""


class DurableExecutionNotFoundError(LookupError):
    """Raised when waiting for an unknown execution ID."""


class DurableExecutionFailedError(RuntimeError):
    """Raised when a durable execution reaches the failed state."""
