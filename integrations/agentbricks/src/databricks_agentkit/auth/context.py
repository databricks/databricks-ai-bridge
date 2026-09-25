"""Task-local authentication context for governed connection requests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from databricks_agentkit.auth.connections import ConnectionRegistry

if TYPE_CHECKING:
    from databricks_agentkit.runtime.auth import RequestAuthContext


_REQUEST_AUTH: ContextVar[RequestAuthContext | None] = ContextVar(
    "agentbricks_connection_request_auth", default=None
)


@contextmanager
def _bind_request_auth(request_auth: RequestAuthContext) -> Iterator[None]:
    """Bind request-user authentication for one active invocation attempt."""
    token = _REQUEST_AUTH.set(request_auth)
    try:
        yield
    finally:
        _REQUEST_AUTH.reset(token)


def _current_request_auth() -> RequestAuthContext | None:
    return _REQUEST_AUTH.get()


connections = ConnectionRegistry()

__all__ = ["connections"]
