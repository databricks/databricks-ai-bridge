"""Blocking Workspace/Lakebase adapters for the synchronous Apps orchestrator."""

import asyncio
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from urllib.parse import quote

from databricks.sdk.errors import Cancelled

from databricks_agentkit.sandbox.session_lifecycle import Sandbox, SessionBinding
from databricks_agentkit.sandbox.store import SessionBusy


async def blocking(function, *args, on_cancel=None, **kwargs):
    """Drain a blocking operation before cancellation releases its allocation lock."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        if on_cancel:
            await blocking(on_cancel, result)
        raise asyncio.CancelledError
    return result


@contextmanager
def allocation(registry, session_id):
    deadline = time.monotonic() + 240
    while True:
        lock = registry.session(session_id)
        try:
            record = lock.__enter__()
        except SessionBusy:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
        else:
            break
    try:
        yield record
    finally:
        lock.__exit__(None, None, None)


class RegistryReservation:
    def __init__(self, record):
        self.record = record

    @property
    def key(self):
        return self.record.key

    async def read(self):
        row = await blocking(self.record.get)
        if row is None:
            return None
        return SessionBinding(
            row["sandbox_id"],
            row.get("source_hash", ""),
            row.get("status", "pending"),
            row.get("source_revision"),
            row.get("execution_principal"),
            (row.get("bootstrap_metrics") or {}).get("worker_state_key"),
        )

    async def bind_owner(self, principal):
        await blocking(self.record.bind_owner, principal)

    async def save(self, binding):
        kwargs = {"source_revision": binding.revision} if binding.revision is not None else {}
        await blocking(
            self.record.save,
            binding.source_hash,
            binding.sandbox_id,
            binding.status,
            None,
            **kwargs,
        )


class RegistryBindings:
    def __init__(self, registry, namespace):
        self.registry, self.namespace = registry, namespace

    @asynccontextmanager
    async def reserve(self, scope):
        if scope.namespace != self.namespace:
            raise ValueError("Session scope does not belong to this registry")
        lock = allocation(self.registry, scope.session_id)
        record = await blocking(
            lock.__enter__, on_cancel=lambda record: lock.__exit__(None, None, None)
        )
        try:
            yield RegistryReservation(record)
        except BaseException:
            await blocking(lock.__exit__, *sys.exc_info())
            raise
        else:
            await blocking(lock.__exit__, None, None, None)


class LakeboxProvider:
    def __init__(self, lakebox):
        self.lakebox = lakebox
        self.snapshots = {}

    def capture(self, value, sandbox_id=None):
        if value is None:
            return None
        sandbox_id = value.get("sandboxId") or sandbox_id
        if not sandbox_id:
            raise ValueError("Sandbox response is missing its identity")
        self.snapshots[sandbox_id] = {**value, "sandboxId": sandbox_id}
        return Sandbox(sandbox_id, value.get("status", "unknown").lower())

    async def get(self, sandbox_id):
        return self.capture(await blocking(self.lakebox.get, sandbox_id), sandbox_id)

    async def find(self, name):
        return self.capture(await blocking(self.lakebox.find_named, name))

    async def create(self, name):
        return self.capture(await blocking(self.lakebox.create, name))

    async def configure(self, sandbox_id):
        await blocking(self.lakebox.set_idle_timeout, sandbox_id)

    async def start(self, sandbox_id):
        return self.capture(await blocking(self.lakebox.start, sandbox_id), sandbox_id)

    async def stop(self, sandbox_id):
        try:
            result = await blocking(
                self.lakebox.api,
                "POST",
                f"/sandboxes/{quote(sandbox_id, safe='')}/stop",
                {"sandbox_id": sandbox_id},
            )
        except Cancelled:
            result = await blocking(self.lakebox.get, sandbox_id)
            if not result or result.get("status", "").lower() not in {
                "stopping",
                "stopped",
                "terminated",
            }:
                raise
        return self.capture(result, sandbox_id)


def run_sync(coroutine):
    """The existing blocking orchestrator calls this outside the service event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise RuntimeError("Use SessionSandboxManager directly from async code")
