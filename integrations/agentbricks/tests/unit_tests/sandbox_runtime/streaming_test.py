"""The preview writer must flush even when no further token arrives."""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from databricks_agentkit.sandbox.control import PartialText


def test_quiet_last_fragment_is_published_before_invocation_exit():
    published = queue.Queue()
    db = MagicMock()
    db.execute.side_effect = lambda statement, params: published.put(params[0])
    with PartialText(db, "namespace", "run", "attempt") as stream:
        stream.emit("Hello")
        assert published.get(timeout=1) == "Hello"
        stream.emit(" world")
        # No further emit and no context exit. The independent timer must wake.
        assert published.get(timeout=1) == "Hello world"
        assert stream.thread.is_alive()
    assert not stream.thread.is_alive()


def test_close_flushes_pending_text_without_waiting_for_batch_deadline():
    published = queue.Queue()
    db = MagicMock()
    db.execute.side_effect = lambda statement, params: published.put(params[0])
    with PartialText(db, "namespace", "run", "attempt", interval=60) as stream:
        stream.emit("First")
        assert published.get(timeout=1) == "First"
        stream.emit(" last")
    assert published.get_nowait() == "First last"
    count = db.execute.call_count
    stream.emit(" after close")
    assert db.execute.call_count == count
    assert not stream.thread.is_alive()


def test_slow_database_does_not_block_reader_and_pending_text_is_bounded():
    started, release = threading.Event(), threading.Event()
    published = []
    db = MagicMock()

    def write(statement, params):
        if not published:
            started.set()
            assert release.wait(2)
        published.append(params[0])

    db.execute.side_effect = write
    with PartialText(db, "namespace", "run", "attempt") as stream:
        stream.emit("first")
        try:
            assert started.wait(1)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(stream.emit, "a" * 300_000).result(timeout=1)
            assert len(stream.text) == 256 * 1024
        finally:
            release.set()
    assert published == ["first", "first" + "a" * (256 * 1024 - 5)]
    assert not stream.thread.is_alive()


def test_preview_failure_does_not_override_agent_result_and_stops_writer(caplog):
    db = MagicMock()
    db.execute.side_effect = RuntimeError("database unavailable")
    with PartialText(db, "namespace", "run", "attempt") as stream:
        stream.emit("answer")
        stream.thread.join(timeout=1)
        assert isinstance(stream.error, RuntimeError)
        stream.emit("ignored after preview failure")
    assert db.execute.call_count == 1
    assert not stream.thread.is_alive()
    assert "final result will carry the answer" in caplog.text
