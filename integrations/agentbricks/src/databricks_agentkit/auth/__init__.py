"""Governed, credential-free access to external UC Connections."""

from databricks_agentkit.auth.connections import (
    ConnectionClient,
    ConnectionError,
    ConnectionHTTPError,
    ConnectionResponse,
    ConnectionTimeoutError,
)

from . import context

__all__ = [
    "ConnectionClient",
    "ConnectionError",
    "ConnectionHTTPError",
    "ConnectionResponse",
    "ConnectionTimeoutError",
    "context",
]
