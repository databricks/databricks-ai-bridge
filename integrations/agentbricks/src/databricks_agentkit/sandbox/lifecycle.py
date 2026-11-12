"""Current-attempt timings, published only at lifecycle transitions."""

import copy
import logging
import time
from datetime import datetime, timezone


class RunLifecycle:
    def __init__(self, publish=None):
        self.publish = publish or (lambda snapshot: None)
        self.snapshot = {
            "steps": {
                name: {"status": "pending"}
                for name in ("sandbox_start", "source_copy", "install_build", "agent_run")
            }
        }
        self.current = None
        self.started = 0.0

    def update(self):
        # Callers may retain the snapshot (and tests do); never mutate an old event.
        try:
            self.publish(copy.deepcopy(self.snapshot))
        except Exception:
            # Losing diagnostics must not change an invocation's outcome after
            # the agent has already performed side effects. The result retains
            # the in-memory snapshot even if live timing updates are unavailable.
            logging.getLogger(__name__).warning("Live timing update unavailable", exc_info=True)
            self.publish = lambda snapshot: None

    def start(self, name):
        self.complete()
        self.current, self.started = name, time.perf_counter()
        self.snapshot["steps"][name] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.update()

    def complete(self, *, status="completed", details=None):
        if self.current is None:
            return
        step = self.snapshot["steps"][self.current]
        step.update(
            status=status,
            duration_ms=round((time.perf_counter() - self.started) * 1000, 2),
        )
        if details is not None:
            step["details_ms"] = details
        self.current, self.started = None, 0.0
        self.update()

    def reuse_installation(self):
        self.complete()
        for name in ("source_copy", "install_build"):
            self.snapshot["steps"][name] = {"status": "reused"}
        self.update()

    def ready(self, duration_ms, *, prepare_only=False):
        self.complete()
        self.snapshot["preparation_ms"] = duration_ms
        if prepare_only:
            self.snapshot["steps"]["agent_run"] = {"status": "skipped"}
        self.update()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind:
            self.complete(status="failed")
