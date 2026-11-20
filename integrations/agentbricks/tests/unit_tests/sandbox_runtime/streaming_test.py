"""Direct subscribers reconstruct previews without persisting token events."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from databricks_agentkit.sandbox.control import ControlStore
from databricks_agentkit.sandbox.streaming import LivePreview


@pytest.fixture
def live(monkeypatch):
    readers, content = [], ["hello"]

    class Reader:
        def __init__(self, remote, path, emit):
            self.emit, self.closed, self.error = emit, False, None
            self.decoder = SimpleNamespace(first_delta_at=12, ended=threading.Event())
            readers.append(self)

        def __enter__(self):
            for text in content:
                self.emit(text)
            return self

        def __exit__(self, *exc):
            self.closed = True

    monkeypatch.setattr("databricks_agentkit.sandbox.streaming.FileFollower", Reader)
    registry = SimpleNamespace(namespace="test")
    factory = MagicMock(side_effect=lambda source, attempt: LivePreview(None, source, attempt))
    control = ControlStore(
        registry, MagicMock(), runtime_store=InMemoryRuntimeStore(), preview_factory=factory
    )
    control.initialize()
    row = control.create_run("agent", "session", "call", {})
    state = control.call(control.runtime_store.claim(row["id"]))
    control.call(
        control.runtime_store.append_event(
            row["id"],
            state.attempt,
            {
                "type": "sandbox.stream.ready",
                "source": {"sandbox_id": "sandbox", "path": "/file", "started_at": 10},
            },
        )
    )
    yield control, row["id"], state.attempt, readers, content, factory
    control.close()


def test_reconnect_replays_file_and_disconnect_only_closes_reader(live):
    control, run_id, attempt, readers, content, _ = live
    watch = control.watch(run_id)
    first = next(watch)
    assert first["stream"]["text"] == "hello"
    assert first["stream"]["first_token_ms"] == 2000
    watch.close()
    assert readers[0].closed
    assert control.run(run_id)["status"] == "running"
    assert control.run(run_id)["stream"] is None
    content.append(" world")
    watch = control.watch(run_id)
    try:
        assert next(watch)["stream"]["text"] == "hello world"
        assert control.call(
            control.runtime_store.complete(run_id, attempt, {"result": {"reply": "hello world"}})
        )
        readers[-1].decoder.ended.set()
        assert next(watch)["status"] == "succeeded"
    finally:
        watch.close()
    events = control.call(control.runtime_store.events(run_id))
    assert not any(item.event["type"] == "sandbox.text.delta" for item in events)
    count = len(readers)
    assert list(control.watch(run_id))[0]["result"]["result"]["reply"] == "hello world"
    assert len(readers) == count  # Completed runs need no sandbox access.


def test_live_deltas_do_not_poll_sql_or_block_on_slow_consumers(live, monkeypatch):
    control, run_id, _, readers, _, _ = live
    reads = MagicMock(wraps=control._state)
    monkeypatch.setattr(control, "_state", reads)
    watch = control.watch(run_id)
    try:
        next(watch)
        for _ in range(4):
            readers[-1].emit("a" * 300_000)
            row = next(watch)
        assert len(row["stream"]["text"]) == 256 * 1024
        assert reads.call_count == 1
    finally:
        watch.close()


def test_lost_sandbox_preview_does_not_change_execution_state(live):
    control, run_id, attempt, _, _, factory = live
    factory.side_effect = RuntimeError("sandbox disappeared")
    watch = control.watch(run_id)
    try:
        row = next(watch)
        assert row["status"] == "running"
        assert "unavailable" in row["stream_error"]
        assert control.call(control.runtime_store.fail(run_id, attempt, {"error": "worker lost"}))
        assert next(watch)["status"] == "failed"
    finally:
        watch.close()


def test_attempt_change_discards_previous_preview_and_closes_reader(live, monkeypatch):
    from dataclasses import replace

    control, run_id, attempt, readers, _, _ = live
    watch = control.watch(run_id)
    try:
        assert next(watch)["stream"]["text"] == "hello"
        state = control._state(run_id)
        monkeypatch.setattr(control, "_state", lambda _: replace(state, attempt=attempt + 1))
        readers[0].decoder.ended.set()  # Wake the state check, without sleeping a second.
        row = next(watch)
        assert row["attempt_id"] == attempt + 1
        assert row["stream"] is None
        assert row["stream_source"] is None
        assert readers[0].closed
    finally:
        watch.close()
