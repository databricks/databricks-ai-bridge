import threading
import time
from unittest.mock import MagicMock

import pytest

from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from databricks_agentkit.runtime.types import InvocationAttemptContext
from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.control import ControlStore, scoped_session
from databricks_agentkit.sandbox.jobs import RunManager


def setup_job():
    registry, catalog, orchestrator = MagicMock(), MagicMock(), MagicMock()
    registry.namespace = "test"
    control = ControlStore(registry, catalog, runtime_store=InMemoryRuntimeStore())
    control.initialize()
    bundle = bundle_files({"pyproject.toml": b"test", "uv.lock": b"lock"})
    catalog.agent.return_value = {
        "id": "agent-a",
        "bundle_data": bundle.data,
        "source_hash": bundle.digest,
        "file_count": bundle.file_count,
        "entrypoint": "custom.agent:invoke",
    }
    manager = RunManager(control, orchestrator)
    manager.start()
    row = {
        "agent_id": "agent-a",
        "session_id": "session-a",
        "invocation_id": "call-a",
        "input": {"message": "hello"},
    }
    return manager, control, orchestrator, row, bundle


@pytest.fixture
def job():
    values = setup_job()
    yield values
    values[0].stop()
    values[1].close()


def submit(job, invocation_id="call-a", session="session-a", value=None):
    manager, control, *_ = job
    row = control.create_run("agent-a", session, invocation_id, value or {"message": "hello"})
    manager.submit(row["id"])
    return row


def terminal(control, run_id):
    deadline = time.monotonic() + 3
    while True:
        row = control.run(run_id)
        if row["status"] not in {"queued", "running"}:
            return row
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_job_uses_shared_runtime_and_saved_revision(job):
    manager, control, orchestrator, _, bundle = job

    def invoke(*args, **kwargs):
        kwargs["progress"]("Installing")
        kwargs["lifecycle"]({"steps": []})
        kwargs["stream_ready"]({"sandbox_id": "sandbox-a", "path": "/attempt.jsonl"})
        return {"result": {"reply": "hello"}}

    orchestrator.invoke.side_effect = invoke
    row = submit(job)
    result = terminal(control, row["id"])
    args, kwargs = orchestrator.invoke.call_args
    assert args == ("session-a", "call-a", {"message": "hello"})
    assert kwargs["registry_session_id"] == scoped_session("agent-a", "session-a")
    assert kwargs["bundle"] == bundle
    assert kwargs["entrypoint"] == "custom.agent:invoke"
    assert result["status"] == "succeeded"
    assert result["stream"] is None
    assert result["stream_source"]["sandbox_id"] == "sandbox-a"
    events = control.call(control.runtime_store.events(row["id"]))
    assert not any(event.event["type"] == "sandbox.text.delta" for event in events)
    assert events[0].event == {"type": "run.started"}
    assert events[-1].event == {"type": "run.completed"}
    assert any(event.event.get("stage") == "Installing" for event in events)
    assert control.history("agent-a", "session-a") == [result]
    assert control.sessions("agent-a")[0]["run_id"] == row["id"]


def test_failed_invocations_stay_terminal_and_errors_are_redacted(job):
    _, control, orchestrator, *_ = job
    orchestrator.invoke.side_effect = RuntimeError("bad dependency token=dapi123456789abc")
    row = submit(job)
    result = terminal(control, row["id"])
    assert result["status"] == "failed"
    assert "dapi123456789abc" not in result["error"]
    assert submit(job)["id"] == row["id"]
    assert orchestrator.invoke.call_count == 1
    with pytest.raises(ValueError, match="new invocation ID"):
        control.create_run("agent-a", "session-a", "call-a", {"message": "hello"}, retry=True)
    orchestrator.invoke.side_effect = None
    orchestrator.invoke.return_value = {"result": {"reply": "retry"}}
    retry = submit(job, "call-b")
    assert terminal(control, retry["id"])["status"] == "succeeded"
    assert len(control.history("agent-a", "session-a")) == 2


def test_duplicate_submission_and_session_fifo(job):
    _, control, orchestrator, *_ = job
    entered, release = threading.Event(), threading.Event()
    calls = []

    def invoke(session, invocation, value, **kwargs):
        calls.append(invocation)
        if invocation == "call-a":
            entered.set()
            assert release.wait(3)
        return {"result": {"reply": invocation}}

    orchestrator.invoke.side_effect = invoke
    first = submit(job)
    try:
        assert entered.wait(1)
        assert submit(job)["id"] == first["id"]
        with pytest.raises(ValueError):
            submit(job, value={"message": "different"})
        second = submit(job, "call-b")
        assert control.run(second["id"])["status"] == "queued"
        assert calls == ["call-a"]
    finally:
        release.set()
    assert terminal(control, second["id"])["status"] == "succeeded"
    assert calls == ["call-a", "call-b"]


def test_recovery_does_not_replay_arbitrary_agent_code(job):
    manager, control, orchestrator, request, _ = job
    with pytest.raises(RuntimeError, match="Automatic replay is disabled"):
        control.call(manager.execute(request, InvocationAttemptContext("run", 2)))
    orchestrator.invoke.assert_not_called()


def test_watch_does_not_skip_events_when_claim_happens_between_reads(job, monkeypatch):
    _, control, *_ = job
    row = control.create_run("agent-a", "session-a", "race", {})
    original = control._state
    first = True

    def read(run_id):
        nonlocal first
        state = original(run_id)
        if first:
            first = False
            attempt = control.call(control.runtime_store.claim(run_id))
            control.call(
                control.runtime_store.append_event(
                    run_id, attempt.attempt, {"type": "sandbox.text.delta", "delta": "racing text"}
                )
            )
        return state

    monkeypatch.setattr(control, "_state", read)
    watch = control.watch(row["id"])
    try:
        assert next(watch)["status"] == "queued"
        assert next(watch)["stream"]["text"] == "racing text"
    finally:
        watch.close()
