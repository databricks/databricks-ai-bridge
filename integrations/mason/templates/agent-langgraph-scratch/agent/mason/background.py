"""Background-run store — plumbing, slated to move into a Databricks SDK / durable backend.

``BackgroundRuns`` tracks in-flight ``background: true`` requests by invocation id so ``GET
/invocations/{id}`` can poll them. This default is **in-memory and single-process**: runs live in a
dict in this process, so they do NOT survive a restart and are NOT shared across replicas.

For production durability (crash recovery, cross-pod resume, surviving the ~120s Apps proxy
timeout), swap this for a shared durable store (e.g. Lakebase) with the same interface — that's the
one edit needed; ``runtime/runtime.py`` only depends on ``start``/``complete``/``fail``/``get``.
"""

import uuid
from typing import Any


class BackgroundRuns:
    """In-memory store of background runs, keyed by invocation id. Single-process, non-durable."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    def start(self) -> str:
        invocation_id = f"inv_{uuid.uuid4().hex[:24]}"
        self._runs[invocation_id] = {"status": "in_progress", "output": None, "error": None}
        return invocation_id

    def complete(self, invocation_id: str, output: dict) -> None:
        self._runs[invocation_id] = {"status": "completed", "output": output, "error": None}

    def fail(self, invocation_id: str, error: str) -> None:
        self._runs[invocation_id] = {"status": "failed", "output": None, "error": error}

    def get(self, invocation_id: str) -> dict | None:
        return self._runs.get(invocation_id)
