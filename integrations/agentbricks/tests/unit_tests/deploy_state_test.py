"""Unit tests for the local deploy record and the server-state classifier behind `deployments wait`."""

from __future__ import annotations

import json
import os
import pathlib
from contextlib import nullcontext
from unittest import mock

import pytest

from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deployment import deploy_state
from databricks_agentbricks.services.deployment.deploy_state import (
    DeployStateReporter,
    DeployStateTracker,
    classify_app,
)

_T0 = 1_800_000_000.0


def _tracker(*, background: bool = False, pid: int = 4242, now: list[float] | None = None):
    clock = now if now is not None else [_T0]
    return DeployStateTracker(
        result_payload=lambda result: {"deployment": result, "url": "https://agent.example"},
        background=background,
        pid=pid,
        clock=lambda: clock[0],
    )


def test_state_and_log_live_under_the_project_state_dir(tmp_path: pathlib.Path):
    assert deploy_state.state_path(tmp_path, "agent-bricks-x") == (
        tmp_path / ".agentbricks" / "deploy-agent-bricks-x.json"
    )
    assert deploy_state.log_path(tmp_path, "agent-bricks-x") == (
        tmp_path / ".agentbricks" / "deploy-agent-bricks-x.log"
    )


def test_tracker_records_each_phase_and_the_successful_result(tmp_path: pathlib.Path):
    clock = [_T0]
    tracker = _tracker(now=clock)
    path = deploy_state.state_path(tmp_path, "agent-bricks-x")

    tracker.start(tmp_path, "agent-bricks-x")
    started = deploy_state.read_state(path)
    assert started is not None
    assert started["status"] == "running"
    assert started["pid"] == 4242
    assert started["phase"] == deploy_state.STARTING_PHASE
    assert started["error"] is None
    assert started["log"] is None  # a foreground deploy writes no log

    clock[0] = _T0 + 30
    tracker.phase("Waiting for agent compute to start…")
    waiting = deploy_state.read_state(path)
    assert waiting is not None
    assert waiting["phase"] == "Waiting for agent compute to start"
    assert waiting["phase_started_at"] == deploy_state.timestamp(_T0 + 30)
    assert waiting["started_at"] == deploy_state.timestamp(_T0)

    tracker.succeeded("agent-bricks-x")
    done = deploy_state.read_state(path)
    assert done is not None
    assert done["status"] == "succeeded"
    assert done["result"] == {"deployment": "agent-bricks-x", "url": "https://agent.example"}
    # No temp file is left beside the record.
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_tracker_records_failure_and_ignores_phases_once_finished(tmp_path: pathlib.Path):
    tracker = _tracker()
    tracker.start(tmp_path, "agent-bricks-x")
    tracker.failed("Could not deploy 'agent-bricks-x'.")
    tracker.phase("Deleting Runtime Store…")  # a later, unrelated step

    state = deploy_state.read_state(deploy_state.state_path(tmp_path, "agent-bricks-x"))
    assert state is not None
    assert state["status"] == "failed"
    assert state["error"] == "Could not deploy 'agent-bricks-x'."
    assert state["phase"] == deploy_state.STARTING_PHASE


def test_unstarted_tracker_records_nothing(tmp_path: pathlib.Path):
    tracker = _tracker()
    tracker.phase("Reconciling Runtime Store…")
    tracker.succeeded("x")
    assert not (tmp_path / ".agentbricks").exists()


def test_background_child_inherits_the_parents_record(tmp_path: pathlib.Path):
    deploy_state.record_background_start(tmp_path, "agent-bricks-x", pid=4242, now=_T0)
    tracker = _tracker(background=True, pid=4242, now=[_T0 + 5])

    tracker.start(tmp_path, "agent-bricks-x")

    state = deploy_state.read_state(deploy_state.state_path(tmp_path, "agent-bricks-x"))
    assert state is not None
    assert state["started_at"] == deploy_state.timestamp(_T0)
    assert state["log"] == str(deploy_state.log_path(tmp_path, "agent-bricks-x"))


def test_record_background_start_records_an_absolute_log_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "proj").mkdir()

    deploy_state.record_background_start(pathlib.Path("proj"), "agent-bricks-x", pid=4242, now=_T0)

    state = deploy_state.read_state(
        tmp_path / "proj" / ".agentbricks" / "deploy-agent-bricks-x.json"
    )
    assert state is not None
    assert state["log"] == str(tmp_path / "proj" / ".agentbricks" / "deploy-agent-bricks-x.log")


def test_record_background_start_does_not_overwrite_a_child_that_already_started(
    tmp_path: pathlib.Path,
):
    tracker = _tracker(background=True, pid=4242)
    tracker.start(tmp_path, "agent-bricks-x")
    tracker.phase("Reconciling Runtime Store…")

    deploy_state.record_background_start(tmp_path, "agent-bricks-x", pid=4242, now=_T0 + 1)

    state = deploy_state.read_state(deploy_state.state_path(tmp_path, "agent-bricks-x"))
    assert state is not None
    assert state["phase"] == "Reconciling Runtime Store"


def test_atomic_write_cleans_up_when_serialization_fails(tmp_path: pathlib.Path):
    path = tmp_path / ".agentbricks" / "deploy-x.json"
    deploy_state.write_state(path, {"status": "running"})

    with mock.patch.object(deploy_state.json, "dump", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            deploy_state.write_state(path, {"status": "succeeded"})

    assert json.loads(path.read_text()) == {"status": "running"}
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_state_reporter_records_phases_and_delegates(tmp_path: pathlib.Path):
    inner = mock.Mock(spec=Reporter)
    inner.progress.return_value = nullcontext()
    inner.status.return_value = nullcontext()
    tracker = _tracker()
    tracker.start(tmp_path, "agent-bricks-x")
    reporter = DeployStateReporter(inner, tracker)

    with reporter.progress("Deploying the agent source…"):
        pass
    reporter.note("App is busy (another operation in progress); retrying in 5s…")
    reporter.echo("out", add_newline=False)

    state = deploy_state.read_state(deploy_state.state_path(tmp_path, "agent-bricks-x"))
    assert state is not None
    assert state["phase"] == "Deploying the agent source"
    inner.progress.assert_called_once_with("Deploying the agent source…")
    inner.note.assert_called_once()
    inner.echo.assert_called_once_with("out", add_newline=False)


def test_pid_alive():
    assert deploy_state.pid_alive(os.getpid()) is True
    assert deploy_state.pid_alive(None) is False
    assert deploy_state.pid_alive(0) is False
    with mock.patch.object(deploy_state.os, "kill", side_effect=ProcessLookupError):
        assert deploy_state.pid_alive(12345) is False
    with mock.patch.object(deploy_state.os, "kill", side_effect=PermissionError):
        assert deploy_state.pid_alive(12345) is True


def test_pid_alive_on_windows_queries_the_process_instead_of_signaling_it():
    # On Windows, signal 0 is CTRL_C_EVENT, so os.kill must never be used as the probe.
    with (
        mock.patch.object(deploy_state.os, "name", "nt"),
        mock.patch.object(deploy_state.os, "kill") as kill,
        mock.patch.object(deploy_state, "_windows_pid_alive", return_value=True) as query,
    ):
        assert deploy_state.pid_alive(12345) is True
    kill.assert_not_called()
    query.assert_called_once_with(12345)


def _running(**overrides):
    state = {
        "pid": 99,
        "status": "running",
        "phase": "Deploying the agent source",
        "started_at": deploy_state.timestamp(_T0),
        "phase_started_at": deploy_state.timestamp(_T0),
        "error": None,
    }
    state.update(overrides)
    return state


def test_running_record_with_dead_pid_counts_as_failed():
    resolved = deploy_state.resolve_state(_running(), now=_T0 + 10, is_alive=lambda pid: False)
    assert resolved["status"] == "failed"
    assert resolved["error"] == "deploy process exited unexpectedly"


def test_running_record_that_has_not_moved_for_too_long_is_stale():
    now = _T0 + deploy_state.STALE_AFTER_S + 1
    resolved = deploy_state.resolve_state(_running(), now=now, is_alive=lambda pid: True)
    assert resolved["status"] == "failed"


def test_live_running_record_is_left_alone():
    state = _running()
    assert deploy_state.resolve_state(state, now=_T0 + 60, is_alive=lambda pid: True) == state


def test_running_deploy_excludes_the_caller_and_dead_processes(tmp_path: pathlib.Path):
    path = tmp_path / "deploy.json"
    deploy_state.write_state(path, _running(pid=99))

    live = deploy_state.running_deploy(path, now=_T0 + 1, is_alive=lambda pid: True, own_pid=1)
    assert live is not None and live["pid"] == 99
    assert (
        deploy_state.running_deploy(path, now=_T0 + 1, is_alive=lambda pid: True, own_pid=99)
        is None
    )
    assert (
        deploy_state.running_deploy(path, now=_T0 + 1, is_alive=lambda pid: False, own_pid=1)
        is None
    )
    assert deploy_state.running_deploy(tmp_path / "missing.json", now=_T0) is None


def test_log_tail_returns_the_last_lines(tmp_path: pathlib.Path):
    log = tmp_path / "deploy.log"
    log.write_text("".join(f"line {i}\n" for i in range(30)))

    assert deploy_state.log_tail(log) == [f"line {i}" for i in range(10, 30)]
    assert deploy_state.log_tail(tmp_path / "missing.log") == []
    assert deploy_state.log_tail(None) == []


# --- server-side classification -----------------------------------------------------------


def _app(compute="ACTIVE", *, pending=None, active=None, url="https://agent.example"):
    payload: dict = {"name": "agent-bricks-x", "url": url, "compute_status": {"state": compute}}
    if pending is not None:
        payload["pending_deployment"] = {"status": {"state": pending, "message": f"{pending} msg"}}
    if active is not None:
        payload["active_deployment"] = {"status": {"state": active, "message": f"{active} msg"}}
    return payload


@pytest.mark.parametrize(
    ("payload", "status", "phase"),
    [
        (_app("STARTING"), "running", "Waiting for agent compute to start"),
        (_app("ACTIVE", pending="IN_PROGRESS"), "running", "Deploying the agent source"),
        (_app("STARTING", pending="IN_PROGRESS"), "running", "Deploying the agent source"),
        (_app("ACTIVE", active="IN_PROGRESS"), "running", "Deploying the agent source"),
        (_app("ACTIVE", active="SUCCEEDED"), "succeeded", "Deployed"),
        (
            _app("ACTIVE", pending="FAILED", active="SUCCEEDED"),
            "failed",
            "Deploying the agent source",
        ),
        (_app("ACTIVE", active="FAILED"), "failed", "Deploying the agent source"),
        (_app("STOPPED", active="SUCCEEDED"), "failed", "Waiting for agent compute to start"),
        (_app("ERROR"), "failed", "Waiting for agent compute to start"),
        (_app("ACTIVE"), "failed", "Deploying the agent source"),
    ],
)
def test_classify_app(payload, status, phase):
    outcome = classify_app("agent-bricks-x", payload)
    assert (outcome.status, outcome.phase) == (status, phase)
    if status == "succeeded":
        assert outcome.url == "https://agent.example"
    if status == "failed":
        assert outcome.error


def test_classify_missing_app_as_failed():
    outcome = classify_app("agent-bricks-x", None)
    assert outcome.status == "failed"
    assert outcome.error == "Deployment 'agent-bricks-x' does not exist."


def test_classify_reads_camel_case_payloads():
    payload = {
        "computeStatus": {"state": "ACTIVE"},
        "activeDeployment": {"status": {"state": "SUCCEEDED"}},
        "url": "https://agent.example",
    }
    assert classify_app("agent-bricks-x", payload).status == "succeeded"


def test_classify_reports_the_failed_deployment_message():
    outcome = classify_app("agent-bricks-x", _app("ACTIVE", pending="FAILED"))
    assert outcome.message == "FAILED msg"
