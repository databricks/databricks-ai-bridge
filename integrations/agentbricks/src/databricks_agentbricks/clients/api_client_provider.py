"""A per-command, lazily initialized Agent Bricks API client.

The client can be slow to construct because it resolves workspace authentication.  Deploy commands
share one provider across their collaborators so construction happens at the first API operation,
after local pre-flight checks have passed, and every later operation reuses the same client.

This is scoped dependency ownership, not a process-global singleton: each CLI invocation owns one
provider and therefore one client.
"""

from __future__ import annotations

from threading import Lock
from typing import Callable, Optional

from databricks_agentkit._api_client import _AgentBricksApiClient


class ApiClientProvider:
    """Own exactly one lazily created API client for a CLI invocation."""

    def __init__(
        self, profile: Optional[str], on_first_use: Optional[Callable[[], None]] = None
    ) -> None:
        self._profile = profile
        self._on_first_use = on_first_use
        self._client: Optional[_AgentBricksApiClient] = None
        self._lock = Lock()

    def use(self, client: _AgentBricksApiClient) -> None:
        """Adopt a client the command already validated, so first use doesn't build another."""
        with self._lock:
            self._client = client

    def get(self) -> _AgentBricksApiClient:
        """Return the shared client, constructing it on first use."""
        client = self._client
        if client is not None:
            return client

        with self._lock:
            if self._client is None:
                if self._on_first_use is not None:
                    self._on_first_use()
                self._client = _AgentBricksApiClient(self._profile)
            return self._client
