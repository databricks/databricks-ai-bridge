"""Background-run store — plumbing, slated to move into a Databricks SDK / durable backend.

``BackgroundRuns`` tracks in-flight ``background: true`` requests by invocation id so ``GET
/invocations/{id}`` can poll them. This default is **in-memory and single-process**: runs live in a
dict in this process, so they do NOT survive a restart and are NOT shared across replicas.

TODO: the SDK contract for managing background-run lifecycle isn't finalized. When it lands, replace
this with a shared durable store (crash recovery, cross-pod resume, surviving the ~120s Apps proxy
timeout) with the same interface — the SDK runtime only depends on
``start``/``complete``/``fail``/``get``.
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
        """The run's record — ``status`` plus ``output`` (when completed) or ``error`` (when failed);
        ``None`` if the id is unknown. Not just the final result: it's the whole tracked state."""
        return self._runs.get(invocation_id)
