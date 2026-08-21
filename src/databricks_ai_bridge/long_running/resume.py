"""Registration primitives for durable resume handlers."""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ParamSpec, TypeVar

from mlflow.types.responses import ResponsesAgentRequest

_P = ParamSpec("_P")
_R = TypeVar("_R")

_on_resume_function: Callable[..., Any] | None = None


class ResumeStrategy(str, Enum):
    """Source of agent context when a stale execution is restarted."""

    EVENT_LOG = "event_log"
    AGENT_SESSION = "agent_session"


@dataclass(frozen=True)
class ResumeContext:
    """Context available while translating a stale execution's stored request.

    ``previous_attempt_events`` contains only the durable stream events emitted
    by the immediately preceding attempt. It is not the agent SDK transcript.
    ``default_resume_request()`` applies the server's configured
    :class:`ResumeStrategy`; an ``@on_resume()`` handler may instead return its
    own request transformation.
    """

    response_id: str
    attempt_number: int
    resume_strategy: ResumeStrategy
    previous_attempt_events: tuple[dict[str, Any], ...]
    _build_default_resume_request: Callable[
        [ResponsesAgentRequest], Awaitable[ResponsesAgentRequest]
    ] = field(repr=False)

    @property
    def previous_attempt_number(self) -> int:
        return self.attempt_number - 1

    async def default_resume_request(self, request: ResponsesAgentRequest) -> ResponsesAgentRequest:
        """Apply the configured strategy's built-in request transformation."""
        return await self._build_default_resume_request(request)


def get_on_resume_function() -> Callable[..., Any] | None:
    return _on_resume_function


def on_resume() -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Register a request transformer called after a stale attempt is claimed.

    The returned request is dispatched through the same ``@invoke()`` or
    ``@stream()`` handler mode selected for the original attempt.
    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        global _on_resume_function
        if _on_resume_function is not None:
            raise ValueError("on_resume decorator can only be used once")
        _on_resume_function = func

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return func(*args, **kwargs)

        return wrapper

    return decorator
