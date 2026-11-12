"""Session-scoped sandbox allocation and lifecycle, independent of agent execution."""

import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import quote

from databricks.sdk.errors import Cancelled

from databricks_agentkit.sandbox.store import ExecutionIdentityConflict, SessionBusy


class RevisionConflict(ValueError):
    pass


@contextmanager
def timed(metrics, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[name] = round((time.perf_counter() - start) * 1000, 2)


@dataclass(frozen=True)
class SessionSandbox:
    sandbox: dict
    created: bool
    replaced: bool
    installation_status: str

    @property
    def id(self):
        return self.sandbox["sandboxId"]


class SessionSandboxClient:
    def __init__(self, registry, lakebox, namespace):
        self.registry, self.lakebox, self.namespace = registry, lakebox, namespace
        self._principal = None

    def execution_principal(self):
        if self._principal is None:
            self._principal = self.lakebox.workspace.current_user.me().user_name
        return self._principal

    @contextmanager
    def invocation(self, session_id):
        """Explicit sequential-turn policy, separate from allocation synchronization."""
        with self.registry.session("invocation:" + session_id):
            yield

    @contextmanager
    def _allocation(self, session_id):
        # A second get_or_create waits for the first allocation and reads its result.
        # This lock is not held while installing source or running the agent.
        deadline = time.monotonic() + 240
        while True:
            lock = self.registry.session(session_id)
            try:
                session = lock.__enter__()
            except SessionBusy:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
            else:
                break
        try:
            yield session
        finally:
            lock.__exit__(None, None, None)

    def get_or_create(
        self,
        session_id,
        *,
        source_hash,
        source_revision=None,
        allow_create=True,
        progress=None,
        metrics=None,
    ):
        metrics = metrics if metrics is not None else {}
        report = progress or (lambda stage: None)
        started = time.perf_counter()
        with self._allocation(session_id) as session:
            metrics["registry_connect_and_lock"] = round((time.perf_counter() - started) * 1000, 2)
            row = session.get()
            if row and row["source_hash"] and row["source_hash"] != source_hash:
                raise RevisionConflict(
                    "Agent source changed; use a new session ID for this revision"
                )
            if not row or row.get("execution_principal") != self.execution_principal():
                session.bind_owner(self.execution_principal())
            sandbox_id = row["sandbox_id"] if row else None
            with timed(metrics, "sandbox_lookup"):
                sandbox = self.lakebox.get(sandbox_id) if sandbox_id else None
            replaced, created = bool(sandbox_id and sandbox is None), False
            if sandbox is None:
                if not allow_create:
                    raise RuntimeError(
                        "Prepared sandbox is missing; prepare a new environment before invoking"
                    )
                report("Creating sandbox")
                label = f"ab-proto-{hashlib.sha256(self.namespace.encode()).hexdigest()[:8]}-{session.key[:16]}"
                with timed(metrics, "sandbox_create_or_reconcile"):
                    sandbox = self.lakebox.find_named(label)
                    if sandbox is None:
                        if row and row["status"] == "creating" and not sandbox_id:
                            raise RevisionConflict(
                                "An earlier create had an ambiguous outcome; inspect the named sandbox before retrying"
                            )
                        self._save(session, source_hash, None, "creating", source_revision)
                        sandbox = self.lakebox.create(label)
                        created = True
                    sandbox_id = sandbox["sandboxId"]
                    self._save(session, source_hash, sandbox_id, "bootstrapping", source_revision)
            self.lakebox.set_idle_timeout(sandbox_id)
            with timed(metrics, "sandbox_resume"):
                if sandbox.get("status", "").lower() != "running":
                    report("Resuming sandbox")
                    sandbox = self.lakebox.start(sandbox_id)
                    if sandbox is None:
                        raise RuntimeError("Sandbox disappeared while starting")
                deadline = time.monotonic() + 240
                while sandbox.get("status", "").lower() != "running":
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Sandbox did not reach running state")
                    time.sleep(1)
                    sandbox = self.lakebox.get(sandbox_id)
                    if sandbox is None:
                        raise RuntimeError("Sandbox disappeared while starting")
            return SessionSandbox(sandbox, created, replaced, row["status"] if row else "creating")

    @staticmethod
    def _save(session, digest, sandbox_id, status, revision=None, metrics=None):
        kwargs = {"source_revision": revision} if revision is not None else {}
        session.save(digest, sandbox_id, status, metrics, **kwargs)

    def record(self, session_id, source_hash, sandbox_id, status, metrics=None):
        with self._allocation(session_id) as session:
            session.bind_owner(self.execution_principal())
            self._save(session, source_hash, sandbox_id, status, metrics=metrics)

    def runtime(self, session_id):
        runtime = self.registry.runtime(session_id)
        owner = runtime["execution_principal"]
        if owner and owner != self.execution_principal():
            raise ExecutionIdentityConflict(
                "This session's sandbox belongs to another execution identity"
            )
        return runtime

    def status(self, session_id):
        runtime = self.runtime(session_id)
        row = runtime["sandbox"]
        sandbox_id = row["sandbox_id"] if row else None
        sandbox = self.lakebox.get(sandbox_id) if sandbox_id else None
        installation = row.get("status", "pending") if row else "not_created"
        status = (
            installation
            if installation in {"pending", "not_created", "creating", "bootstrapping", "failed"}
            else (sandbox.get("status", "unknown").lower() if sandbox else "missing")
        )
        return {**runtime, "sandbox_id": sandbox_id, "status": status}

    def stop(self, session_id):
        started, metrics = time.perf_counter(), {}
        with self.invocation(session_id), self._allocation(session_id) as session:
            row = session.get()
            if not row or not row["sandbox_id"]:
                raise KeyError("Session has no sandbox")
            session.bind_owner(self.execution_principal())
            metrics["registry_connect_and_lock"] = round((time.perf_counter() - started) * 1000, 2)
            sandbox_id = row["sandbox_id"]
            with timed(metrics, "sandbox_lookup"):
                result = self.lakebox.get(sandbox_id)
            if result is None:
                raise KeyError("Session sandbox no longer exists")
            with timed(metrics, "stop_request"):
                if result.get("status", "").lower() not in {"stopped", "terminated"}:
                    try:
                        result = self.lakebox.api(
                            "POST",
                            f"/sandboxes/{quote(sandbox_id, safe='')}/stop",
                            {"sandbox_id": sandbox_id},
                        )
                    except Cancelled:
                        result = self.lakebox.get(sandbox_id)
                        if not result or result.get("status", "").lower() not in {
                            "stopping",
                            "stopped",
                            "terminated",
                        }:
                            raise
            with timed(metrics, "wait_until_stopped"):
                deadline = time.monotonic() + 180
                while not result or result.get("status", "").lower() not in {
                    "stopped",
                    "terminated",
                }:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Sandbox is still stopping")
                    time.sleep(1)
                    result = self.lakebox.get(sandbox_id)
                    if result is None:
                        raise KeyError("Sandbox disappeared before shutdown was confirmed")
            self._save(session, row["source_hash"], sandbox_id, "stopped")
        metrics["total"] = round((time.perf_counter() - started) * 1000, 2)
        return {"sandbox_id": sandbox_id, "status": "stopped", "timings_ms": metrics}
