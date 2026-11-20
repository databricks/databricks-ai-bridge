import json
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from databricks_agentkit.sandbox.bundle import build_bundle
from databricks_agentkit.sandbox.orchestrator import Orchestrator, RevisionConflict


class MemorySession:
    key = "a" * 64

    def __init__(self):
        self.row = None
        self.owner = None

    def bind_owner(self, principal):
        from databricks_agentkit.sandbox.store import ExecutionIdentityConflict

        if self.owner and self.owner != principal:
            raise ExecutionIdentityConflict("Environment runtime belongs to another identity")
        self.owner = principal

    def get(self):
        return self.row

    def save(self, digest, sandbox_id, status, metrics=None):
        self.row = {"source_hash": digest, "sandbox_id": sandbox_id, "status": status}


class MemoryRegistry:
    def __init__(self):
        self.record = MemorySession()

    @contextmanager
    def session(self, session_id):
        yield self.record


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"\n')
    (tmp_path / "uv.lock").write_text("version=1\n")
    registry = MemoryRegistry()
    lakebox = MagicMock()
    lakebox.transport = "ssh"
    lakebox.find_named.return_value = None
    lakebox.create.return_value = {"sandboxId": "sandbox-one", "status": "Running"}
    lakebox.get.return_value = lakebox.create.return_value
    remote = lakebox.connect.return_value.__enter__.return_value

    def run(command, *args, **kwargs):
        if command.startswith('printf "%s"'):
            return "/home/sandbox"
        if command.startswith("if test"):
            return ""
        if command.startswith("python3 -c"):
            return json.dumps(
                {"python": "/venv/bin/python", "timings_ms": {"dependency_install": 42}}
            )
        if "sandbox_agent.worker:run" in command:
            return json.dumps({"reply": "hello"})
        return ""

    remote.run.side_effect = run
    return (
        Orchestrator(registry, lakebox, tmp_path, "test", worker_mode="oneshot"),
        registry,
        lakebox,
        remote,
    )


def test_reuses_persisted_mapping_from_another_orchestrator(setup):
    orchestrator, registry, lakebox, _ = setup
    first = orchestrator.invoke("session", "call-one", {})
    replacement = Orchestrator(
        registry, lakebox, orchestrator.source, "test", worker_mode="oneshot"
    )
    second = replacement.invoke("session", "call-two", {})
    assert first["sandbox_created"]
    assert not second["sandbox_created"]
    assert first["sandbox_id"] == second["sandbox_id"]
    assert lakebox.create.call_count == 1


def test_skips_transfer_and_install_when_environment_is_ready(setup):
    orchestrator, registry, lakebox, remote = setup
    registry.record.save(build_bundle(orchestrator.source).digest, "sandbox-one", "ready")
    remote.run.side_effect = [
        "/home/sandbox",
        json.dumps({"python": "/venv/bin/python", "timings_ms": {"dependency_install": 42}}),
        json.dumps({"reply": "hello"}),
    ]
    result = orchestrator.invoke("session", "call-two", {})
    assert result["environment_reused"]
    assert "bootstrap" not in result["timings_ms"]
    assert "source_transfer" not in result["timings_ms"]
    lakebox.create.assert_not_called()


def test_does_not_create_again_after_an_ambiguous_create(setup):
    orchestrator, registry, lakebox, _ = setup
    lakebox.create.side_effect = TimeoutError("reply lost")
    with pytest.raises(TimeoutError):
        orchestrator.invoke("session", "first", {})
    assert registry.record.get()["status"] == "creating"
    with pytest.raises(RevisionConflict, match="ambiguous"):
        orchestrator.invoke("session", "retry", {})
    assert lakebox.create.call_count == 1


def test_source_change_requires_an_explicit_new_session(setup):
    orchestrator, registry, lakebox, _ = setup
    registry.record.save("old-revision", "sandbox-one", "ready")
    with pytest.raises(RevisionConflict, match="source changed"):
        orchestrator.invoke("session", "call", {})
    lakebox.get.assert_not_called()


def test_registered_revision_is_used_instead_of_mutable_local_source(setup):
    orchestrator, _, _, remote = setup
    bundle = build_bundle(orchestrator.source)
    (orchestrator.source / "uv.lock").unlink()
    stages = []
    result = orchestrator.invoke(
        "registered-session",
        "call",
        {},
        bundle=bundle,
        progress=stages.append,
        registry_session_id="agent:revision\0registered-session",
    )
    assert result["source_hash"] == bundle.digest
    assert "Creating sandbox" in stages
    assert "Installing dependencies and building agent" in stages
    assert stages[-1] == "Running agent"
    assert "importlib" in remote.run.call_args.args[0]
    assert json.loads(remote.run.call_args.args[1])["session_id"] == "registered-session"


def test_startup_duration_excludes_agent_execution(setup, monkeypatch):
    orchestrator, _, lakebox, remote = setup
    clock = [10.0]
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.orchestrator.time.perf_counter", lambda: clock[0]
    )
    run = remote.run.side_effect

    def create(name):
        clock[0] += 2
        return {"sandboxId": "sandbox-one", "status": "Running"}

    def execute(command, *args, **kwargs):
        if command.startswith("cd ") and "sandbox_agent.worker:run" in command:
            clock[0] += 12
        return run(command, *args, **kwargs)

    lakebox.create.side_effect = create
    remote.run.side_effect = execute
    result = orchestrator.invoke("session", "call", {})
    assert result["timings_ms"]["environment_startup"] == 2000
    assert result["timings_ms"]["worker_process_and_execution"] == 12000
    assert result["timings_ms"]["total"] == 14000


def test_live_lifecycle_reports_each_phase_before_completion(setup, monkeypatch):
    orchestrator, _, lakebox, remote = setup
    clock, events = [10.0], []
    monkeypatch.setattr("databricks_agentkit.sandbox.lifecycle.time.perf_counter", lambda: clock[0])
    run = remote.run.side_effect

    def create(name):
        assert events[-1]["steps"]["sandbox_start"]["status"] == "running"
        clock[0] += 2
        return {"sandboxId": "sandbox-one", "status": "Running"}

    def execute(command, *args, **kwargs):
        if "cat > " in command:
            assert events[-1]["steps"]["source_copy"]["status"] == "running"
            clock[0] += 1
        if command.startswith("python3 -c"):
            assert events[-1]["steps"]["install_build"]["status"] == "running"
            clock[0] += 5
        if command.startswith("cd "):
            assert events[-1]["steps"]["agent_run"]["status"] == "running"
            assert events[-1]["preparation_ms"] == 8000
            clock[0] += 3
            return json.dumps(
                {"reply": "hello", "worker_timings_ms": {"prepare": 2000, "agent_execution": 500}}
            )
        return run(command, *args, **kwargs)

    lakebox.create.side_effect = create
    remote.run.side_effect = execute
    result = orchestrator.invoke("session", "call", {}, lifecycle=events.append)
    steps = result["lifecycle"]["steps"]
    assert {name: step["duration_ms"] for name, step in steps.items()} == {
        "sandbox_start": 2000,
        "source_copy": 1000,
        "install_build": 5000,
        "agent_run": 3000,
    }
    assert all(step["status"] == "completed" for step in steps.values())
    assert steps["install_build"]["details_ms"] == {"dependency_install": 42}
    assert steps["agent_run"]["details_ms"] == {"prepare": 2000, "agent_execution": 500}
    # Prior live snapshots must not be mutated by later transitions.
    assert events[0]["steps"]["sandbox_start"]["status"] == "running"
    assert events[0]["steps"]["agent_run"]["status"] == "pending"


def test_reused_installation_does_not_report_original_build_as_current_work(setup):
    orchestrator, _, _, remote = setup
    remote.run.side_effect = [
        "/home/sandbox",
        json.dumps({"python": "/venv/bin/python", "timings_ms": {"dependency_install": 42000}}),
        json.dumps({"reply": "hello"}),
    ]
    result = orchestrator.invoke("session", "call", {})
    assert result["lifecycle"]["steps"]["source_copy"] == {"status": "reused"}
    assert result["lifecycle"]["steps"]["install_build"] == {"status": "reused"}


def test_install_failure_retains_completed_steps_and_identifies_failed_phase(setup):
    orchestrator, _, _, remote = setup
    events = []
    run = remote.run.side_effect

    def execute(command, *args, **kwargs):
        if command.startswith("python3 -c"):
            raise RuntimeError("resolution failed")
        return run(command, *args, **kwargs)

    remote.run.side_effect = execute
    with pytest.raises(RuntimeError, match="resolution failed"):
        orchestrator.invoke("session", "call", {}, lifecycle=events.append)
    steps = events[-1]["steps"]
    assert steps["sandbox_start"]["status"] == "completed"
    assert steps["source_copy"]["status"] == "completed"
    assert steps["install_build"]["status"] == "failed"
    assert steps["install_build"]["duration_ms"] >= 0
    assert steps["agent_run"] == {"status": "pending"}


@pytest.mark.parametrize("fail_at", ["start", "completion"])
def test_timing_persistence_failure_does_not_discard_successful_agent_result(setup, fail_at):
    orchestrator, registry, _, _ = setup

    def publish(snapshot):
        if fail_at == "start" or snapshot["steps"]["agent_run"]["status"] == "completed":
            raise ConnectionError("Timing store disconnected")

    result = orchestrator.invoke("session", "call", {}, lifecycle=publish)
    assert result["result"] == {"reply": "hello"}
    assert result["lifecycle"]["steps"]["agent_run"]["status"] == "completed"
    assert registry.record.get()["status"] == "ready"


def test_persistent_worker_envelope_preserves_agent_result_and_reports_reuse(setup):
    orchestrator, _, _, remote = setup
    orchestrator.worker_mode = "persistent"
    run = remote.run.side_effect
    calls = []

    def execute(command, *args, **kwargs):
        if command.startswith("cd "):
            calls.append(command)
            return json.dumps(
                {
                    "ok": True,
                    "result": {"reply": "hello", "worker": "agent-owned-field"},
                    "worker": {
                        "mode": "persistent",
                        "id": "worker-a",
                        "pid": 123,
                        "reused": len(calls) > 1,
                        "invocation_count": len(calls),
                    },
                }
            )
        return run(command, *args, **kwargs)

    remote.run.side_effect = execute
    first = orchestrator.invoke("session", "first", {})
    second = orchestrator.invoke("session", "second", {})
    assert first["result"] == second["result"] == {"reply": "hello", "worker": "agent-owned-field"}
    assert not first["worker"]["reused"] and second["worker"]["reused"]
    assert all("DAEMON_SCRIPT" in command and "python3 -c" in command for command in calls)


def test_default_worker_mode_and_explicit_oneshot_override(setup, monkeypatch):
    previous, registry, lakebox, _ = setup
    monkeypatch.delenv("SANDBOX_WORKER_MODE", raising=False)
    assert Orchestrator(registry, lakebox, previous.source, "test").worker_mode == "persistent"
    monkeypatch.setenv("SANDBOX_WORKER_MODE", "oneshot")
    assert Orchestrator(registry, lakebox, previous.source, "test").worker_mode == "oneshot"
    monkeypatch.setenv("SANDBOX_WORKER_MODE", "invalid")
    with pytest.raises(ValueError, match="persistent or oneshot"):
        Orchestrator(registry, lakebox, previous.source, "test")


def test_streamed_text_uses_an_attempt_file_and_follows_before_execution(setup, monkeypatch):
    orchestrator, _, _, remote = setup
    run = remote.run.side_effect
    chunks, paths = [], []

    class Follower:
        def __init__(self, remote, path, emit):
            self.emit = emit
            paths.append(path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def finish(self):
            self.emit("Hello")

    monkeypatch.setattr("databricks_agentkit.sandbox.orchestrator.FileFollower", Follower)
    remote.run.side_effect = run
    result = orchestrator.invoke("session", "call", {}, emit=chunks.append)
    first_path = paths[0]
    assert chunks == ["Hello"]
    assert result["result"] == {"reply": "hello"}
    assert result["stream_transport"] == "ExecuteCommand(tail -F)"
    command = next(
        call.args[0] for call in remote.run.call_args_list if " --stream-file " in call.args[0]
    )
    assert command.endswith(first_path)
    assert "/streams/" in first_path and first_path.endswith(".jsonl")
    orchestrator.invoke("session", "call", {}, emit=chunks.append)
    assert paths[1] != first_path


def test_app_execution_publishes_descriptor_without_owning_a_tail_reader(setup, monkeypatch):
    orchestrator, _, _, remote = setup
    remote.sandbox_id = "sandbox-one"
    follower = MagicMock(side_effect=AssertionError("execution must not own a subscriber"))
    monkeypatch.setattr("databricks_agentkit.sandbox.orchestrator.FileFollower", follower)
    descriptors = []
    result = orchestrator.invoke("session", "call", {}, stream_ready=descriptors.append)
    assert result["result"] == {"reply": "hello"}
    assert descriptors[0]["sandbox_id"] == "sandbox-one"
    command = next(c.args[0] for c in remote.run.call_args_list if " --stream-file " in c.args[0])
    assert command.endswith(descriptors[0]["path"])
    follower.assert_not_called()
    orchestrator.invoke("session", "call", {}, stream_ready=descriptors.append)
    assert descriptors[0]["path"] != descriptors[1]["path"]
