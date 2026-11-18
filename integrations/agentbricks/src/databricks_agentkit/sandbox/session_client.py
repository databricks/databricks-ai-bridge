"""Session-scoped sandbox allocation and lifecycle, independent of agent execution."""

import hashlib
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

from databricks_agentkit.sandbox._apps_lifecycle import (
    LakeboxProvider,
    RegistryBindings,
    allocation,
    blocking,
    run_sync,
)
from databricks_agentkit.sandbox.session_lifecycle import (
    AgentRevision,
    ExecutionIdentityConflict,
    SandboxAllocation,
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
class PreparedSandbox:
    """A spare validated during this claim, never a cross-request health cache."""

    sandbox: dict
    home: str
    bootstrap: dict
    state_dir: str
    entrypoint: str
    worker: dict
    result: dict


@dataclass(frozen=True)
class SessionSandbox:
    sandbox: dict
    created: bool
    replaced: bool
    installation_status: str
    state_key: str | None = None
    pool_hit: bool = False
    prepared: PreparedSandbox | None = None

    @property
    def id(self):
        return self.sandbox["sandboxId"]


class SessionSandboxClient:
    def __init__(self, registry, lakebox, namespace):
        self.registry, self.lakebox, self.namespace = registry, lakebox, namespace
        self._principal = None
        self.warm_pool = None
        self.capacity = None

    async def _reserve_compute(self, reservation, binding):
        def reserve():
            guard = self.capacity.slot("allocation", 1) if self.capacity else nullcontext()
            with guard:
                if self.capacity:
                    self.capacity.check_sandboxes(self.lakebox, existing_id=binding.sandbox_id)
                self._save(
                    reservation.record,
                    binding.source_hash,
                    binding.sandbox_id,
                    binding.status,
                    binding.revision,
                )

        await blocking(reserve)

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
            reserve_compute=self._reserve_compute,
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
        use_pool=True,
    ):
        metrics = metrics if metrics is not None else {}
        provider = LakeboxProvider(self.lakebox)
        prepared = None

        async def claim_prepared(reservation, row):
            nonlocal prepared
            if not use_pool or not self.warm_pool or (row and row.status != "pending"):
                return None
            with timed(metrics, "pool_claim"):
                prepared = await blocking(
                    self.warm_pool.claim, reservation.record, source_hash, source_revision
                )
            if prepared:
                binding = await reservation.read()
                if progress:
                    progress("Using prepared sandbox from warm pool")
                return SandboxAllocation(
                    provider.capture(prepared.sandbox),
                    False,
                    False,
                    "ready",
                    binding.worker_state_key,
                    prepared=True,
                )
            return None

        result = run_sync(
            self._manager(provider).get_or_create(
                SessionScope(self.namespace, session_id, self.execution_principal()),
                AgentRevision(source_hash, source_revision),
                allow_create=allow_create,
                progress=progress,
                metrics=metrics,
                claim_prepared=claim_prepared,
            )
        )
        return SessionSandbox(
            provider.snapshots[result.sandbox.id],
            result.created,
            result.replaced,
            result.installation_status,
            result.worker_state_key,
            pool_hit=result.prepared,
            prepared=prepared,
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
