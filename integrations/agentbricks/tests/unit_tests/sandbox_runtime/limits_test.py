"""Capacity admission across controllers sharing one deployment registry."""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from databricks_agentkit.runtime.types import InvocationConflictError
from databricks_agentkit.sandbox.control import ControlStore
from databricks_agentkit.sandbox.limits import Capacity, CapacityExceeded, RuntimeLimits
from databricks_agentkit.sandbox.store import SessionBusy


class Locks:
    namespace = "deployment"

    def __init__(self):
        self.locks = {}
        self.lock = threading.Lock()

    @contextmanager
    def session(self, key):
        with self.lock:
            lock = self.locks.setdefault(key, threading.Lock())
        if not lock.acquire(blocking=False):
            raise SessionBusy(key)
        try:
            yield
        finally:
            lock.release()


def test_atomic_queue_admission_allows_duplicate_reconnect_at_limit():
    registry = Locks()
    control = ControlStore(registry, MagicMock(), runtime_store=InMemoryRuntimeStore())
    control.initialize()
    control.capacity = Capacity(registry, RuntimeLimits(max_pending_invocations=1))
    barrier = threading.Barrier(2)

    def submit(number):
        barrier.wait()
        try:
            return control.create_run("agent", f"session-{number}", str(number), {}, public_id=True)
        except CapacityExceeded:
            return None

    try:
        with ThreadPoolExecutor(2) as pool:
            rows = list(pool.map(submit, [1, 2]))
        accepted = [row for row in rows if row]
        assert len(accepted) == 1
        row = accepted[0]
        assert (
            control.create_run("agent", row["session_id"], row["id"], {}, public_id=True)["id"]
            == row["id"]
        )
        with pytest.raises(InvocationConflictError):
            control.create_run(
                "agent", row["session_id"], row["id"], {"different": True}, public_id=True
            )
    finally:
        control.close()


def test_execution_slots_are_shared_and_released_after_error():
    registry = Locks()
    a, b = Capacity(registry, RuntimeLimits()), Capacity(registry, RuntimeLimits())
    with pytest.raises(ValueError):
        with a.slot("invocation", 1):
            with pytest.raises(SessionBusy):
                with b.slot("invocation", 1, wait=False):
                    pytest.fail("Acquired occupied capacity")
            raise ValueError("failed execution")
    with b.slot("invocation", 1, wait=False):
        pass


def test_compute_count_includes_spares_and_ambiguous_creates_but_deduplicates_aliases():
    registry, lakebox = MagicMock(), MagicMock()
    registry.mapping_page.side_effect = [
        [
            {"session_key": "s", "sandbox_id": "box", "status": "ready"},
            {"session_key": "pool-alias", "sandbox_id": "box", "status": "ready"},
            {"session_key": "old", "sandbox_id": "stopped", "status": "ready"},
            {"session_key": "reservation", "sandbox_id": None, "status": "creating"},
        ],
        [],
    ]
    lakebox.statuses.return_value = {"box": "running", "stopped": "stopped"}
    capacity = Capacity(registry, RuntimeLimits(max_sandboxes=2))
    with pytest.raises(CapacityExceeded):
        capacity.check_sandboxes(lakebox)


@pytest.mark.parametrize("value", [0, -1, True, 10001])
def test_invalid_capacity_is_rejected(value):
    with pytest.raises(ValueError):
        RuntimeLimits(max_sandboxes=value)
