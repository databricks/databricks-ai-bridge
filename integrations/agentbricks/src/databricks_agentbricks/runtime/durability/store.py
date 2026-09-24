"""Runtime Store capabilities required for heartbeat-based recovery."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from databricks_agentbricks.runtime.store import RuntimeStore
from databricks_agentbricks.runtime.types import Invocation


@runtime_checkable
class DurableRuntimeStore(RuntimeStore, Protocol):
    """Extend Runtime Store operations with durable recovery coordination."""

    async def queued_invocation_ids(self) -> list[str]:
        """List invocations waiting to begin their first attempt."""
        ...

    async def stale_invocation_ids(self, stale_seconds: float) -> list[str]:
        """List active invocations whose heartbeat is stale."""
        ...

    async def claim_recoverable(
        self,
        invocation_id: str,
        stale_seconds: float,
    ) -> Invocation | None:
        """Replace a stale active attempt and return the new active attempt."""
        ...

    async def heartbeat(self, invocation_id: str, attempt: int) -> bool:
        """Refresh an active attempt's lease while the caller owns it."""
        ...
