"""The same compute policy works with native async storage and transport adapters."""

import asyncio
import hashlib
import subprocess
import sys
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from databricks_agentkit.sandbox.session_lifecycle import (
    AgentRevision,
    ExecutionIdentityConflict,
    RevisionConflict,
    Sandbox,
    SessionBinding,
    SessionSandboxManager,
    SessionScope,
)


class Bindings:
    def __init__(self):
        self.rows = {}
        self.locks = defaultdict(asyncio.Lock)

    @asynccontextmanager
    async def reserve(self, scope):
        key = (scope.namespace, scope.session_id)
        async with self.locks[key]:
            rows = self.rows

            class Reservation:
                async def read(self):
                    return rows.get(key)

                async def bind_owner(self, principal):
                    row = await self.read()
                    if row and row.execution_principal not in {None, principal}:
                        raise ExecutionIdentityConflict("Different owner")
                    rows[key] = replace(
                        row or SessionBinding(None, "", "pending"), execution_principal=principal
                    )

                async def save(self, binding):
                    rows[key] = binding

            reservation = Reservation()
            reservation.key = hashlib.sha256(repr(key).encode()).hexdigest()
            yield reservation


class Provider:
    def __init__(self):
        self.sandboxes = {}
        self.names = {}
        self.creates = self.starts = self.stops = 0
        self.lose_create_reply = False
        self.created_status = "running"

    async def get(self, sandbox_id):
        return self.sandboxes.get(sandbox_id)

    async def find(self, name):
        return self.sandboxes.get(self.names.get(name))

    async def create(self, name):
        self.creates += 1
        sandbox = Sandbox(str(self.creates), self.created_status)
        self.sandboxes[sandbox.id] = sandbox
        self.names[name] = sandbox.id
        if self.lose_create_reply:
            raise TimeoutError("Create reply lost")
        return sandbox

    async def configure(self, sandbox_id):
        pass

    async def start(self, sandbox_id):
        self.starts += 1
        self.sandboxes[sandbox_id] = Sandbox(sandbox_id, "running")
        return self.sandboxes[sandbox_id]

    async def stop(self, sandbox_id):
        self.stops += 1
        self.sandboxes[sandbox_id] = Sandbox(sandbox_id, "stopping")
        return self.sandboxes[sandbox_id]


@pytest.fixture
def backend():
    bindings, provider = Bindings(), Provider()
    manager = SessionSandboxManager(
        bindings, provider, sandbox_name=lambda scope, key: key, poll_seconds=0.001
    )
    return manager, bindings, provider


SCOPE = SessionScope("workspace:1/agent:one/store:one", "session", "principal")
REVISION = AgentRevision("source-hash", "revision-one")


@pytest.mark.asyncio
async def test_concurrent_allocation_has_one_binding_and_compute(backend):
    manager, bindings, provider = backend
    results = await asyncio.gather(*(manager.get_or_create(SCOPE, REVISION) for _ in range(12)))
    assert {result.sandbox.id for result in results} == {"1"}
    assert sum(result.created for result in results) == provider.creates == len(bindings.rows) == 1
    assert all(not result.prepared for result in results)


@pytest.mark.asyncio
async def test_same_session_id_in_another_tenant_uses_distinct_compute(backend):
    manager, _, provider = backend
    first = await manager.get_or_create(SCOPE, REVISION)
    second = await manager.get_or_create(replace(SCOPE, namespace="workspace:2"), REVISION)
    assert first.sandbox.id != second.sandbox.id and provider.creates == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["owner", "source", "revision"])
async def test_binding_identity_is_pinned_before_compute_operations(backend, change):
    manager, _, provider = backend
    await manager.get_or_create(SCOPE, REVISION)
    scope = replace(SCOPE, execution_principal="other") if change == "owner" else SCOPE
    revision = {
        "owner": REVISION,
        "source": AgentRevision("other-source", "revision-one"),
        "revision": AgentRevision("source-hash", "other-revision"),
    }[change]
    with pytest.raises((ExecutionIdentityConflict, RevisionConflict)):
        await manager.get_or_create(scope, revision)
    assert provider.creates == 1


@pytest.mark.asyncio
async def test_lost_create_reply_is_reconciled_without_creating_again(backend):
    manager, bindings, provider = backend
    provider.lose_create_reply = True
    with pytest.raises(TimeoutError):
        await manager.get_or_create(SCOPE, REVISION)
    assert next(iter(bindings.rows.values())).status == "creating"
    recovered = await manager.get_or_create(SCOPE, REVISION)
    assert recovered.sandbox.id == "1" and provider.creates == 1


@pytest.mark.asyncio
async def test_unknown_create_outcome_is_not_replayed(backend):
    manager, _, provider = backend
    provider.lose_create_reply = True
    with pytest.raises(TimeoutError):
        await manager.get_or_create(SCOPE, REVISION)
    provider.sandboxes.clear()
    with pytest.raises(RevisionConflict, match="ambiguous"):
        await manager.get_or_create(SCOPE, REVISION)
    assert provider.creates == 1


@pytest.mark.asyncio
async def test_create_acceptance_does_not_wait_for_worker_or_resume_compute(backend):
    manager, _, provider = backend
    provider.created_status = "pending"
    accepted = await manager.get_or_create(SCOPE, REVISION, wait_for_running=False)
    assert accepted.sandbox.status == "pending" and not accepted.prepared
    assert provider.starts == 0
    running = await manager.get_or_create(SCOPE, REVISION)
    assert running.sandbox.status == "running" and not running.prepared
    assert provider.creates == provider.starts == 1


@pytest.mark.asyncio
async def test_stop_acceptance_is_not_reported_as_confirmed_shutdown(backend):
    manager, bindings, provider = backend
    sandbox = (await manager.get_or_create(SCOPE, REVISION)).sandbox
    accepted = await manager.stop(SCOPE, wait=False)
    assert accepted.status == next(iter(bindings.rows.values())).status == "stopping"
    provider.sandboxes[sandbox.id] = Sandbox(sandbox.id, "stopped")
    confirmed = await manager.stop(SCOPE)
    assert confirmed.status == next(iter(bindings.rows.values())).status == "stopped"
    assert provider.stops == 1


@pytest.mark.asyncio
async def test_blocking_adapter_drains_cancelled_operation_before_releasing_resource():
    from databricks_agentkit.sandbox._apps_lifecycle import blocking

    entered, release = threading.Event(), threading.Event()
    cleaned = []

    def acquire():
        entered.set()
        assert release.wait(5)
        return "resource"

    task = asyncio.create_task(blocking(acquire, on_cancel=cleaned.append))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not cleaned
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == ["resource"]


def test_lifecycle_import_has_no_apps_storage_transport_or_framework_dependency():
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'databricks_agentbricks', 'psycopg', 'fastapi', 'paramiko', 'grpc', 'agents', 'langgraph'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from databricks_agentkit.sandbox.session_lifecycle import SessionSandboxManager
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
