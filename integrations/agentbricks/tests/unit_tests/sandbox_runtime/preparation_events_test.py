"""Readiness notifications wake observers without owning preparation work."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from databricks_agentkit.sandbox.preparation import SessionPreparation


def test_readiness_written_between_snapshots_is_not_lost():
    service = SessionPreparation(MagicMock(), MagicMock())
    row = {"status": "queued"}
    service.get = lambda *args: row.copy()
    watch = service.watch("agent", "session")
    assert next(watch)["status"] == "queued"
    row["status"] = "ready"
    service.notify()
    with ThreadPoolExecutor() as worker:
        assert worker.submit(next, watch).result(timeout=0.5)["status"] == "ready"
    watch.close()
    service.orchestrator.prepare_session.assert_not_called()


def test_scheduler_wakes_for_new_request_without_waiting_for_rescan():
    service = SessionPreparation(MagicMock(), MagicMock())
    waiting, queued, executed = threading.Event(), threading.Event(), threading.Event()

    class ObservedCondition(threading.Condition):
        def wait_for(self, *args, **kwargs):
            waiting.set()
            return super().wait_for(*args, **kwargs)

    service.changed = ObservedCondition()
    service.rows = lambda: (
        [{"preparation": {"agent_id": "a", "session_id": "s"}}] if queued.is_set() else []
    )
    service.execute = lambda *args: (queued.clear(), executed.set())
    thread = threading.Thread(target=service._worker)
    thread.start()
    try:
        assert waiting.wait(2)
        queued.set()
        service.notify()
        assert executed.wait(0.5)
    finally:
        service.stop()
        thread.join(2)
    assert not thread.is_alive()
