"""Session-scoped sandbox allocation and lifecycle, independent of agent execution."""

import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass

from databricks_agentkit.sandbox._apps_lifecycle import (
    LakeboxProvider,
    RegistryBindings,
    allocation,
    run_sync,
)
from databricks_agentkit.sandbox.session_lifecycle import (
    AgentRevision,
    ExecutionIdentityConflict,
    SessionSandboxManager,
    SessionScope,
)
from databricks_agentkit.sandbox.session_lifecycle import RevisionConflict as RevisionConflict


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

    def _allocation(self, session_id):
        return allocation(self.registry, session_id)

    def _manager(self, provider):
        return SessionSandboxManager(
            RegistryBindings(self.registry, self.namespace),
            provider,
            sandbox_name=lambda scope, key: (
                f"ab-proto-{hashlib.sha256(scope.namespace.encode()).hexdigest()[:8]}-{key[:16]}"
            ),
        )

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
        provider = LakeboxProvider(self.lakebox)
        result = run_sync(
            self._manager(provider).get_or_create(
                SessionScope(self.namespace, session_id, self.execution_principal()),
                AgentRevision(source_hash, source_revision),
                allow_create=allow_create,
                progress=progress,
                metrics=metrics,
            )
        )
        return SessionSandbox(
            provider.snapshots[result.sandbox.id],
            result.created,
            result.replaced,
            result.installation_status,
        )

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
        started = time.perf_counter()
        metrics = {}
        provider = LakeboxProvider(self.lakebox)
        with self.invocation(session_id):
            result = run_sync(
                self._manager(provider).stop(
                    SessionScope(self.namespace, session_id, self.execution_principal()),
                    metrics=metrics,
                )
            )
        metrics["total"] = round((time.perf_counter() - started) * 1000, 2)
        return {
            "sandbox_id": result.id,
            "status": "stopped",
            "timings_ms": metrics,
        }
