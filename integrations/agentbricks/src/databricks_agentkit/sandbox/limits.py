"""Deployment limits; advisory locks coordinate App replicas without catalog tables."""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields

from databricks_agentkit.sandbox.store import SessionBusy


class CapacityExceeded(RuntimeError):
    """Admission was refused before starting additional work."""


@dataclass(frozen=True)
class RuntimeLimits:
    max_sandboxes: int = 64
    max_concurrent_invocations: int = 16
    max_concurrent_preparations: int = 2
    max_pending_invocations: int = 128

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10000:
                raise ValueError(f"{field.name} must be between 1 and 10000")

    def env(self):
        return {
            f"SANDBOX_{field.name.upper()}": str(getattr(self, field.name))
            for field in fields(self)
        }

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(
            **{
                field.name: int(
                    os.getenv(f"SANDBOX_{field.name.upper()}", getattr(defaults, field.name))
                )
                for field in fields(cls)
            }
        )


class Capacity:
    def __init__(self, registry, limits):
        self.registry, self.limits = registry, limits

    def check_sandboxes(self, lakebox, *, existing_id=None):
        """Called under the allocation-capacity lock before reserving compute.

        Creating/resuming reservations count even if the platform has not yet
        reported them. Unknown states fail closed. Stopped mappings cost no slot.
        Pool and session aliases for the same sandbox count only once.
        """
        rows, after = [], ""
        while page := self.registry.mapping_page(after=after):
            rows.extend(page)
            after = page[-1]["session_key"]
        ids = {row["sandbox_id"] for row in rows if row["sandbox_id"]}
        statuses = lakebox.statuses(ids) if ids else {}
        reservations = {
            row["sandbox_id"] or row["session_key"]
            for row in rows
            if row["status"] in {"creating", "resuming"}
            or row["status"] == "bootstrapping"
            and statuses.get(row["sandbox_id"]) == "missing"
            or row["sandbox_id"]
            and statuses.get(row["sandbox_id"], "unknown")
            not in {"stopped", "terminated", "missing"}
        }
        if existing_id in reservations:
            return
        if len(reservations) >= self.limits.max_sandboxes:
            raise CapacityExceeded(
                "Sandbox capacity reached; stop a session or wait for idle shutdown"
            )

    @contextmanager
    def slot(self, kind, maximum, *, wait=True):
        # One connection per slot holder. Lock names include the deployment
        # namespace through Registry.session; no worker-local counters are authoritative.
        while True:
            for number in range(maximum):
                lock = self.registry.session(f"capacity:{kind}:{number}")
                try:
                    lock.__enter__()
                except SessionBusy:
                    continue
                try:
                    yield
                finally:
                    lock.__exit__(None, None, None)
                return
            if not wait:
                raise SessionBusy(f"{kind} capacity is occupied")
            time.sleep(0.1)
