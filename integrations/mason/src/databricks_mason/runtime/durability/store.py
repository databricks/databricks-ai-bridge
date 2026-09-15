"""Runtime Store capabilities required for heartbeat-based recovery."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from databricks_mason.runtime.store import RuntimeStore
from databricks_mason.runtime.types import Invocation


@runtime_checkable
class DurableRuntimeStore(RuntimeStore, Protocol):
    """Extend Runtime Store operations with durable recovery coordination."""

    async def recoverable_invocation_ids(self, stale_seconds: float) -> list[str]:
        """List queued invocations and active invocations whose heartbeat is stale."""
        ...

    async def claim_recoverable(
        self,
        invocation_id: str,
        stale_seconds: float,
    ) -> Invocation | None:
        """Claim queued or stale work and return its new active attempt."""
        ...

    async def heartbeat(self, invocation_id: str, attempt: int) -> bool:
        """Refresh an active attempt's lease while the caller owns it."""
        ...
