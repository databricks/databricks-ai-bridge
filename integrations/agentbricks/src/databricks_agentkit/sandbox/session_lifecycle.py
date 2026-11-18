"""Backend-neutral, asynchronous session compute lifecycle; never claims invocations."""

import asyncio
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol


class RevisionConflict(ValueError):
    pass


class ExecutionIdentityConflict(ValueError):
    pass


@dataclass(frozen=True)
class SessionScope:
    """Host-authorized identity; namespace must include the host's tenant partition."""

    namespace: str
    session_id: str
    execution_principal: str

    def __post_init__(self):
        if not all((self.namespace, self.session_id, self.execution_principal)):
            raise ValueError("Session scope requires namespace, session and execution principal")


@dataclass(frozen=True)
class AgentRevision:
    source_hash: str
    revision: str | None = None


@dataclass(frozen=True)
class Sandbox:
    id: str
    status: str


@dataclass(frozen=True)
class SessionBinding:
    sandbox_id: str | None
    source_hash: str
    status: str
    revision: str | None = None
    execution_principal: str | None = None
    worker_state_key: str | None = None


@dataclass(frozen=True)
class SandboxAllocation:
    """Compute acceptance/running state is separate from agent preparation readiness."""

    sandbox: Sandbox
    created: bool
    replaced: bool
    installation_status: str
    worker_state_key: str | None = None
    prepared: bool = False


class BindingReservation(Protocol):
    key: str

    async def read(self) -> SessionBinding | None: ...

    async def bind_owner(self, principal: str) -> None: ...

    async def save(self, binding: SessionBinding) -> None: ...


class SessionBindingStore(Protocol):
    def reserve(self, scope: SessionScope) -> AbstractAsyncContextManager[BindingReservation]:
        """Serialize allocation for this scope; retain ambiguous reservations durably."""
        ...


class SandboxProvider(Protocol):
    async def get(self, sandbox_id: str) -> Sandbox | None: ...

    async def find(self, name: str) -> Sandbox | None: ...

    async def create(self, name: str) -> Sandbox: ...

    async def configure(self, sandbox_id: str) -> None: ...

    async def start(self, sandbox_id: str) -> Sandbox | None: ...

    async def stop(self, sandbox_id: str) -> Sandbox | None: ...


async def _reserve_compute(reservation, binding):
    await reservation.save(binding)


class SessionSandboxManager:
    """Shared compute policy with host-owned storage, authorization and scheduling.

    The host admits invocations independently. Neither allocating compute nor observing
    its status transfers execution ownership or permits replay of arbitrary agent code.
    """

    def __init__(
        self,
        bindings: SessionBindingStore,
        provider: SandboxProvider,
        *,
        sandbox_name: Callable[[SessionScope, str], str],
        reserve_compute=_reserve_compute,
        timeout: float = 240,
        poll_seconds: float = 1,
    ):
        self.bindings, self.provider = bindings, provider
        self.sandbox_name, self.reserve_compute = sandbox_name, reserve_compute
        self.timeout, self.poll_seconds = timeout, poll_seconds

    async def get_or_create(
        self,
        scope: SessionScope,
        revision: AgentRevision,
        *,
        allow_create: bool = True,
        wait_for_running: bool = True,
        progress=None,
        metrics=None,
        claim_prepared=None,
    ) -> SandboxAllocation:
        metrics = metrics if metrics is not None else {}
        report = progress or (lambda stage: None)
        started = time.perf_counter()
        async with self.bindings.reserve(scope) as reservation:
            metrics["registry_connect_and_lock"] = (time.perf_counter() - started) * 1000
            row = await reservation.read()
            if row and row.source_hash and row.source_hash != revision.source_hash:
                raise RevisionConflict(
                    "Agent source changed; use a new session ID for this revision"
                )
            if row and row.revision and revision.revision and row.revision != revision.revision:
                raise RevisionConflict("Agent revision changed; use a new session ID")
            if row and row.execution_principal not in {None, scope.execution_principal}:
                raise ExecutionIdentityConflict(
                    "Session sandbox belongs to another execution identity"
                )
            if not row or row.execution_principal != scope.execution_principal:
                await reservation.bind_owner(scope.execution_principal)
            if claim_prepared and allow_create and not (row and row.sandbox_id):
                if prepared := await claim_prepared(reservation, row):
                    return prepared
            sandbox_id = row.sandbox_id if row else None
            started = time.perf_counter()
            sandbox = await self.provider.get(sandbox_id) if sandbox_id else None
            metrics["sandbox_lookup"] = (time.perf_counter() - started) * 1000
            replaced, created = bool(sandbox_id and sandbox is None), False

            async def save(sandbox_id, status, *, reserve=False):
                write = self.reserve_compute if reserve else _reserve_compute
                await write(
                    reservation,
                    SessionBinding(
                        sandbox_id,
                        revision.source_hash,
                        status,
                        revision.revision,
                        scope.execution_principal,
                    ),
                )

            if sandbox is None:
                if not allow_create:
                    raise RuntimeError(
                        "Prepared sandbox is missing; prepare a new environment before invoking"
                    )
                report("Creating sandbox")
                started = time.perf_counter()
                name = self.sandbox_name(scope, reservation.key)
                sandbox = await self.provider.find(name)
                if sandbox is None:
                    if row and row.status == "creating" and not sandbox_id:
                        raise RevisionConflict(
                            "An earlier create had an ambiguous outcome; inspect the named sandbox before retrying"
                        )
                    await save(None, "creating", reserve=True)
                    sandbox = await self.provider.create(name)
                    created = True
                sandbox_id = sandbox.id
                await save(sandbox_id, "bootstrapping")
                metrics["sandbox_create_or_reconcile"] = (time.perf_counter() - started) * 1000
            await self.provider.configure(sandbox.id)
            started = time.perf_counter()
            if wait_for_running:
                if sandbox.status != "running":
                    report("Resuming sandbox")
                    await save(sandbox.id, "resuming", reserve=True)
                    sandbox = await self.provider.start(sandbox.id)
                    if sandbox is None:
                        raise RuntimeError("Sandbox disappeared while starting")
                    await save(sandbox.id, row.status if row else "bootstrapping")
                deadline = time.monotonic() + self.timeout
                while sandbox.status != "running":
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Sandbox did not reach running state")
                    await asyncio.sleep(self.poll_seconds)
                    sandbox = await self.provider.get(sandbox.id)
                    if sandbox is None:
                        raise RuntimeError("Sandbox disappeared while starting")
            metrics["sandbox_resume"] = (time.perf_counter() - started) * 1000
            return SandboxAllocation(
                sandbox,
                created,
                replaced,
                row.status if row else "creating",
                row.worker_state_key if row else None,
            )

    async def stop(
        self, scope: SessionScope, *, wait: bool = True, timeout: float = 180, metrics=None
    ):
        metrics = metrics if metrics is not None else {}
        started = time.perf_counter()
        async with self.bindings.reserve(scope) as reservation:
            metrics["registry_connect_and_lock"] = (time.perf_counter() - started) * 1000
            row = await reservation.read()
            if not row or not row.sandbox_id:
                raise KeyError("Session has no sandbox")
            if row.execution_principal not in {None, scope.execution_principal}:
                raise ExecutionIdentityConflict(
                    "Session sandbox belongs to another execution identity"
                )
            await reservation.bind_owner(scope.execution_principal)
            started = time.perf_counter()
            sandbox = await self.provider.get(row.sandbox_id)
            metrics["sandbox_lookup"] = (time.perf_counter() - started) * 1000
            if sandbox is None:
                raise KeyError("Session sandbox no longer exists")
            started = time.perf_counter()
            if sandbox.status not in {"stopped", "terminated"}:
                sandbox = await self.provider.stop(row.sandbox_id)
            metrics["stop_request"] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            deadline = time.monotonic() + timeout
            while wait and (sandbox is None or sandbox.status not in {"stopped", "terminated"}):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Sandbox is still stopping")
                await asyncio.sleep(self.poll_seconds)
                sandbox = await self.provider.get(row.sandbox_id)
                if sandbox is None:
                    raise KeyError("Sandbox disappeared before shutdown was confirmed")
            metrics["wait_until_stopped"] = (time.perf_counter() - started) * 1000
            confirmed = bool(sandbox and sandbox.status in {"stopped", "terminated"})
            await reservation.save(
                SessionBinding(
                    row.sandbox_id,
                    row.source_hash,
                    "stopped" if confirmed else "stopping",
                    row.revision,
                    scope.execution_principal,
                )
            )
            return sandbox
