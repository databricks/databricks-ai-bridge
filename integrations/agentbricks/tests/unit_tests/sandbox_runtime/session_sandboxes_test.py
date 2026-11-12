"""Session compute isolation through real orchestration with in-memory external boundaries."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from databricks_agentkit.sandbox.app import app
from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.control import scoped_session
from databricks_agentkit.sandbox.environment import PUBLISHED_FORMAT, bind_environment
from databricks_agentkit.sandbox.orchestrator import Orchestrator
from databricks_agentkit.sandbox.store import ExecutionIdentityConflict, SessionBusy


class MemoryRegistry:
    def __init__(self):
        self.records = {}
        self.owners = {}
        self.locks = {}

    @contextmanager
    def session(self, identity):
        lock = self.locks.setdefault(identity, threading.Lock())
        if not lock.acquire(blocking=False):
            raise SessionBusy("Session already running")
        registry = self

        class Record:
            key = hashlib.sha256(identity.encode()).hexdigest()

            def get(self):
                return registry.records.get(identity)

            def bind_owner(self, principal):
                owner = registry.owners.setdefault(identity, principal)
                if owner != principal:
                    raise ExecutionIdentityConflict("Sandbox belongs to another principal")

            def save(self, digest, sandbox_id, status, metrics=None):
                previous = self.get() or {}
                registry.records[identity] = {
                    "sandbox_id": sandbox_id,
                    "source_hash": digest,
                    "status": status,
                    "bootstrap_metrics": metrics or previous.get("bootstrap_metrics"),
                }

        try:
            yield Record()
        finally:
            lock.release()

    def runtime(self, identity):
        return {
            "sandbox": self.records.get(identity),
            "execution_principal": self.owners.get(identity),
        }


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(app.state, "preparations", None, raising=False)
    registry = MemoryRegistry()
    lakebox = MagicMock()
    lakebox.transport = "https"
    lakebox.workspace.current_user.me.return_value.user_name = "app-principal"
    lakebox.find_named.return_value = None
    sandboxes, installs, workers = {}, [], []
    block = {}

    def create(label):
        sandbox = {"sandboxId": f"sandbox-{len(sandboxes) + 1}", "status": "Running"}
        sandboxes[sandbox["sandboxId"]] = sandbox
        return sandbox

    def start(sandbox_id):
        sandboxes[sandbox_id]["status"] = "Running"
        return sandboxes[sandbox_id]

    def stop(method, path, body):
        assert method == "POST" and path.endswith("/stop")
        sandboxes[body["sandbox_id"]]["status"] = "Stopped"
        return sandboxes[body["sandbox_id"]]

    @contextmanager
    def connect(sandbox):
        sandbox_id = sandbox["sandboxId"]
        ready = {"python": "/venv/bin/python", "timings_ms": {"dependency_install": 42}}

        def run(command, data=None, **kwargs):
            if command.startswith("printf"):
                return "/home/sandbox"
            if command.startswith("if test"):
                return json.dumps(ready) if sandbox_id in installs else ""
            if command.startswith("python3 -c"):
                installs.append(sandbox_id)
                return json.dumps(ready)
            if command.startswith("cd "):
                request = json.loads(data)
                workers.append((sandbox_id, request["session_id"], command))
                if gate := block.get(request["session_id"]):
                    gate[0].set()
                    assert gate[1].wait(5), "Worker gate timed out"
                return json.dumps({"reply": request["input"].get("message", "hello")})
            return ""

        yield SimpleNamespace(run=run)

    lakebox.create.side_effect = create
    lakebox.get.side_effect = sandboxes.get
    lakebox.connect.side_effect = connect
    lakebox.start.side_effect = start
    lakebox.api.side_effect = stop
    orchestrator = Orchestrator(registry, lakebox, tmp_path, "test", worker_mode="oneshot")
    manifest = {
        "format": PUBLISHED_FORMAT,
        "status": "published",
        "namespace": "test",
        "runtime_key": "environment:test",
        "agent_id": "agent",
        "name": "test",
    }
    bind_environment(orchestrator, manifest)
    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"lock"})

    def invoke(session, invocation="call"):
        return orchestrator.invoke(
            session,
            invocation,
            {"message": "hello"},
            bundle=bundle,
            registry_session_id=scoped_session("agent", session),
        )

    control = MagicMock()
    control.agent.return_value = {"source_hash": bundle.digest}
    monkeypatch.setattr(app.state, "orchestrator", orchestrator, raising=False)
    monkeypatch.setattr(app.state, "control", control, raising=False)
    return SimpleNamespace(
        registry=registry,
        lakebox=lakebox,
        orchestrator=orchestrator,
        manifest=manifest,
        bundle=bundle,
        invoke=invoke,
        installs=installs,
        workers=workers,
        block=block,
        client=TestClient(app),
    )


def test_new_sessions_create_distinct_sandboxes_and_later_turns_reuse_installation(runtime):
    first = runtime.invoke("one")
    again = runtime.invoke("one", "second")
    other = runtime.invoke("two")
    assert first["sandbox_id"] == again["sandbox_id"] != other["sandbox_id"]
    assert first["sandbox_created"] and other["sandbox_created"]
    assert not again["sandbox_created"] and again["environment_reused"]
    assert runtime.installs == [first["sandbox_id"], other["sandbox_id"]]
    assert runtime.lakebox.create.call_count == 2
    assert len({call.args[0] for call in runtime.lakebox.create.call_args_list}) == 2
    assert set(runtime.registry.records) == {
        scoped_session("agent", "one"),
        scoped_session("agent", "two"),
    }
    assert set(runtime.registry.owners.values()) == {"app-principal"}
    assert all(result["sandbox_scope"] == "session" for result in [first, again, other])


def test_mapping_survives_orchestrator_restart(runtime):
    first = runtime.invoke("one")
    previous = runtime.orchestrator
    replacement = Orchestrator(
        previous.registry, previous.lakebox, previous.source, "test", worker_mode="oneshot"
    )
    bind_environment(replacement, runtime.manifest)
    again = replacement.invoke(
        "one",
        "second",
        {},
        bundle=runtime.bundle,
        registry_session_id=scoped_session("agent", "one"),
    )
    assert again["sandbox_id"] == first["sandbox_id"] and again["environment_reused"]
    assert runtime.lakebox.create.call_count == 1


def test_different_sessions_can_run_while_same_session_is_locked(runtime):
    entered, release = threading.Event(), threading.Event()
    runtime.block["one"] = (entered, release)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(runtime.invoke, "one")
        try:
            assert entered.wait(5)
            with pytest.raises(SessionBusy):
                runtime.invoke("one", "concurrent")
            other = executor.submit(runtime.invoke, "two").result(timeout=5)
            assert other["sandbox_id"] == "sandbox-2"
        finally:
            release.set()
        assert first.result(timeout=5)["sandbox_id"] == "sandbox-1"


def test_status_before_first_message_is_read_only(runtime):
    response = runtime.client.get("/api/agents/agent/sandbox", params={"session_id": "one"})
    assert response.status_code == 200
    assert response.json()["status"] == "not_created"
    assert response.json()["sandbox_id"] is None
    assert not runtime.registry.records and not runtime.registry.owners
    runtime.lakebox.create.assert_not_called()
    runtime.lakebox.get.assert_not_called()


def test_stop_only_affects_selected_session_and_next_message_resumes_it(runtime):
    first, other = runtime.invoke("one"), runtime.invoke("two")
    response = runtime.client.post(
        "/sessions/stop", json={"agent_id": "agent", "session_id": "one"}
    )
    assert response.status_code == 200
    assert response.json()["sandbox_id"] == first["sandbox_id"]
    one = runtime.client.get("/api/agents/agent/sandbox", params={"session_id": "one"}).json()
    two = runtime.client.get("/api/agents/agent/sandbox", params={"session_id": "two"}).json()
    assert one["status"] == "stopped" and two["status"] == "running"
    assert two["sandbox_id"] == other["sandbox_id"]
    resumed = runtime.invoke("one", "after-stop")
    assert resumed["sandbox_id"] == first["sandbox_id"] and resumed["environment_reused"]
    assert runtime.lakebox.create.call_count == 2 and len(runtime.installs) == 2


def test_status_rejects_a_different_owner_without_calling_compute(runtime):
    identity = scoped_session("agent", "one")
    runtime.registry.owners[identity] = "other-app"
    response = runtime.client.get("/api/agents/agent/sandbox", params={"session_id": "one"})
    assert response.status_code == 409
    runtime.lakebox.get.assert_not_called()
    runtime.lakebox.create.assert_not_called()
