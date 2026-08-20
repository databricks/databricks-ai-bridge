"""Registration primitives for durable resume handlers."""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

from mlflow.types.responses import ResponsesAgentRequest

_P = ParamSpec("_P")
_R = TypeVar("_R")

_on_resume_function: Callable[..., Any] | None = None


@dataclass(frozen=True)
class ResumeContext:
    """Metadata and default behavior available to an ``@on_resume`` handler."""

    response_id: str
    attempt_number: int
    previous_events: tuple[dict[str, Any], ...]
    _default_request: Callable[[ResponsesAgentRequest], Awaitable[ResponsesAgentRequest]] = field(
        repr=False
    )

    @property
    def previous_attempt_number(self) -> int:
        return self.attempt_number - 1

    async def default_request(self, request: ResponsesAgentRequest) -> ResponsesAgentRequest:
        """Apply the server's configured default recovery policy."""
        return await self._default_request(request)


def get_on_resume_function() -> Callable[..., Any] | None:
    return _on_resume_function


def on_resume() -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Register the request transformer used when a stale attempt is resumed."""

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
