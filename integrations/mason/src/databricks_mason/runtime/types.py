"""Contracts shared by Mason Runtime, stores, and agent applications.

The runtime returns :class:`Invocation` snapshots and ordered :class:`InvocationEvent` records.
An executor receives an :class:`InvocationAttemptContext` for attempt fencing and event emission.
``AgentApp`` adapts that lower-level context into :class:`InvocationContext` for functions
registered with ``@app.invoke`` and ``@app.recover``.

All request, response, and event payloads use the recursive ``JsonValue`` / ``JsonObject`` aliases,
so Runtime Store values can be stored consistently by in-memory and Lakebase implementations.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from databricks_mason.runtime.auth import RequestAuthContext

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
InvocationEventEmitter = Callable[[JsonObject], Awaitable[int]]


class InvocationStatus(str, Enum):
    """Lifecycle states stored by Mason Runtime."""

    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Invocation:
    """A Runtime Store snapshot for one idempotent invocation and its owning attempt."""

    invocation_id: str
    status: InvocationStatus
    attempt: int
    request: JsonValue
    response: JsonValue

    @property
    def is_terminal(self) -> bool:
        """Whether this invocation can no longer transition or emit events."""
        return self.status in {
            InvocationStatus.COMPLETED,
            InvocationStatus.FAILED,
        }


@dataclass(frozen=True)
class InvocationEvent:
    """One persisted event emitted by an invocation attempt."""

    sequence_number: int
    invocation_id: str
    attempt: int
    event: JsonObject


@dataclass(frozen=True)
class InvocationAttemptContext:
    """Attempt metadata and event emission passed to the runtime's executor function."""

    invocation_id: str
    attempt: int
    _emit: InvocationEventEmitter | None = field(default=None, repr=False, compare=False)

    @property
    def is_recovery(self) -> bool:
        """Whether this is a replacement attempt after an earlier worker stopped heartbeating."""
        return self.attempt > 1

    async def emit(self, event: JsonObject) -> int:
        """Persist an ordered event and return its replay cursor."""
        if self._emit is None:
            raise RuntimeError("event emission is not available for this invocation context")
        return await self._emit(event)


InvocationExecutorFn = Callable[[JsonValue, InvocationAttemptContext], Awaitable[JsonValue]]


@dataclass(frozen=True)
class InvocationContext:
    """Invocation/session metadata and event emission for a decorated agent function."""

    invocation_id: str
    session_id: str
    attempt: int
    _attempt_context: InvocationAttemptContext = field(repr=False, compare=False)
    request_auth: "RequestAuthContext | None" = field(default=None, repr=False, compare=False)

    @property
    def is_recovery(self) -> bool:
        """Whether ``@app.recover`` is handling a replacement attempt."""
        return self.attempt > 1

    async def emit(self, event: JsonObject) -> int:
        """Persist an ordered application event and return its replay cursor."""
        return await self._attempt_context.emit(event)


InvocationHook = Callable[[JsonValue, InvocationContext], Awaitable[JsonValue]]


class InvocationConflictError(ValueError):
    """Raised when an invocation ID is reused with a different request."""


class InvocationNotFoundError(LookupError):
    """Raised when waiting for an unknown invocation ID."""


class InvocationFailedError(RuntimeError):
    """Raised when an invocation reaches the failed state."""
