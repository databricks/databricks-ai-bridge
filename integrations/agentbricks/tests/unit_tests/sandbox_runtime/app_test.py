import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound, PermissionDenied
from fastapi.testclient import TestClient

from databricks_agentkit.sandbox.app import app
from databricks_agentkit.sandbox.bundle import bundle_files, files_digest
from databricks_agentkit.sandbox.control import scoped_session
from databricks_agentkit.sandbox.orchestrator import Orchestrator
from databricks_agentkit.sandbox.session_client import SessionSandboxClient


@pytest.fixture
def client(monkeypatch):
    # App routing runs normally. External source, database, and execution services
    # are replaced so API contract tests do not allocate live resources.
    for name in ("sources", "control", "runs", "orchestrator", "environments"):
        monkeypatch.setattr(app.state, name, MagicMock(), raising=False)
    from databricks_agentkit.sandbox.preparation import SessionPreparation

    monkeypatch.setattr(app.state, "preparations", MagicMock(spec=SessionPreparation), raising=False)
    app.state.orchestrator.prepared_session_id = None
    app.state.orchestrator.environment_session_id = None
    app.state.orchestrator.sandbox_scope = "session"
    app.state.orchestrator.sandboxes = SessionSandboxClient(
        app.state.orchestrator.registry, app.state.orchestrator.lakebox, "test"
    )
    app.state.orchestrator.execution_session_id.side_effect = lambda key: (
        Orchestrator.execution_session_id(app.state.orchestrator, key)
    )
    monkeypatch.setattr(app.state, "prepared_environment", None, raising=False)
    return TestClient(app)


def test_ui_and_static_assets_are_served_by_the_orchestrator(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Workspace directory" in page.text
    assert "Unity Catalog Volume" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_chat_model_discovery_uses_app_identity_without_starting_a_sandbox(client, monkeypatch):
    discover = MagicMock(return_value=["system.ai.gpt-5-mini", "system.ai.claude-sonnet-5-5"])
    monkeypatch.setattr("databricks_agentkit.sandbox.app.list_ai_gateway_model_services", discover)
    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json() == {"available": discover.return_value}
    discover.assert_called_once_with(app.state.orchestrator.lakebox.workspace)
    app.state.orchestrator.invoke.assert_not_called()
    app.state.orchestrator.lakebox.create.assert_not_called()


def test_chat_model_discovery_failure_does_not_break_static_ui(client, monkeypatch):
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.app.list_ai_gateway_model_services",
        MagicMock(side_effect=PermissionDenied("No model-list access")),
    )
    assert client.get("/api/models").status_code == 403
    assert client.get("/").status_code == 200
    app.state.orchestrator.invoke.assert_not_called()


def test_registration_snapshots_source_without_executing_it(client):
    bundle = bundle_files({"pyproject.toml": b"test", "uv.lock": b"lock"})
    app.state.sources.load.return_value = bundle
    app.state.control.create_agent.return_value = {"id": "revision"}
    response = client.post(
        "/api/agents",
        json={
            "name": "Research",
            "source_kind": "workspace",
            "source_path": "/Workspace/Users/a/source",
            "entrypoint": "research.agent:run",
        },
    )
    assert response.status_code == 201
    app.state.sources.load.assert_called_once_with("workspace", "/Users/a/source")
    app.state.control.create_agent.assert_called_once_with(
        "Research", "workspace", "/Users/a/source", "research.agent:run", bundle
    )
    app.state.orchestrator.invoke.assert_not_called()


@pytest.mark.parametrize(
    "error,status",
    [
        (PermissionDenied("Grant the App principal read access"), 403),
        (NotFound("Source path was not found; check the path and read access"), 404),
    ],
)
def test_source_access_failure_is_actionable_and_does_not_register_revision(client, error, status):
    app.state.sources.load.side_effect = error
    response = client.post(
        "/api/agents",
        json={
            "name": "Research",
            "source_kind": "volume",
            "source_path": "/Volumes/c/s/v/agent",
        },
    )
    assert response.status_code == status
    assert "read access" in response.json()["detail"]
    app.state.control.create_agent.assert_not_called()


def test_invalid_entrypoint_is_rejected_before_source_download(client):
    response = client.post(
        "/api/agents",
        json={
            "name": "Research",
            "source_kind": "bundled",
            "entrypoint": "python agent.py",
        },
    )
    assert response.status_code == 400
    app.state.sources.load.assert_not_called()


def test_invocation_is_enqueued_and_returns_status_without_waiting_for_sandbox(client):
    app.state.control.create_run.return_value = {"id": "run-one", "status": "queued"}
    response = client.post(
        "/api/runs",
        json={
            "agent_id": "agent-one",
            "session_id": "session-one",
            "invocation_id": "call-one",
            "input": {"message": "hello"},
        },
    )
    assert response.status_code == 202
    assert response.json() == {"id": "run-one", "status": "queued"}
    app.state.runs.submit.assert_called_once_with("run-one")
    app.state.orchestrator.invoke.assert_not_called()


def test_completed_invocation_is_not_enqueued_again(client):
    app.state.control.create_run.return_value = {"id": "run-one", "status": "succeeded"}
    response = client.post(
        "/api/runs", json={"agent_id": "a", "session_id": "s", "invocation_id": "i"}
    )
    assert response.status_code == 202
    app.state.runs.submit.assert_not_called()


def test_missing_run_returns_404(client):
    app.state.control.run.side_effect = KeyError("Invocation not found")
    assert client.get("/api/runs/missing").status_code == 404
    assert client.get("/api/runs/missing/events").status_code == 404


def test_sse_replays_snapshots_and_final_result_without_launching_work(client):
    import json

    partial = {
        "id": "run",
        "status": "running",
        "stream": {"attempt": "one", "sequence": 1, "text": "Hello\n🌍"},
    }
    final = {**partial, "status": "succeeded", "result": {"result": {"reply": "Hello\n🌍"}}}
    app.state.control.run.return_value = partial
    app.state.control.watch.side_effect = lambda run_id: iter([partial, partial, final])
    for _ in range(2):
        response = client.get("/api/runs/run/events")
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        rows = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert rows == [partial, final]
    app.state.runs.submit.assert_not_called()
    app.state.orchestrator.invoke.assert_not_called()


def test_cancelled_sse_subscriber_closes_its_database_iterator(client):
    import asyncio

    from anyio import CancelScope

    from databricks_agentkit.sandbox.app import run_events

    closed = []

    def watch(run_id):
        try:
            yield {"id": run_id, "status": "running"}
            yield {"id": run_id, "status": "succeeded"}
        finally:
            closed.append(True)

    app.state.control.watch.side_effect = watch

    async def consume():
        response = run_events("run")
        await anext(response.body_iterator)
        with CancelScope() as scope:
            scope.cancel()
            await response.body_iterator.aclose()
        assert closed == [True]

    asyncio.run(consume())


def test_legacy_reasoning_display_is_consistent_across_history_poll_sse_and_replay(client):
    encoded = json.dumps(
        [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "", "signature": "test-signature"}],
            }
        ]
    )
    partial = {"id": "run", "status": "running", "stream": {"text": encoded + "Hello"}}
    final = {**partial, "status": "succeeded", "result": {"result": {"reply": encoded + "Hello!"}}}
    before = copy.deepcopy(final)
    app.state.control.run.return_value = final
    app.state.control.history.return_value = [final]
    app.state.control.create_run.return_value = final
    app.state.environments.invoke.return_value = final
    app.state.control.watch.side_effect = lambda _: iter([partial, final])
    assert client.get("/api/runs/run").json()["display_text"] == "Hello!"
    assert client.get("/api/runs?agent_id=a&session_id=s").json()[0]["display_text"] == "Hello!"
    payload = {"agent_id": "a", "session_id": "s", "invocation_id": "i"}
    assert client.post("/api/runs", json=payload).json()["display_text"] == "Hello!"
    assert (
        client.post("/api/environments/demo/runs", json=payload).json()["display_text"] == "Hello!"
    )
    response = client.get("/api/runs/run/events")
    rows = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert [row["display_text"] for row in rows] == ["Hello", "Hello!"]
    assert rows[-1]["result"] == final["result"]
    assert final == before
    app.state.runs.submit.assert_not_called()


def test_stop_uses_agent_and_session_scope(client):
    selected = []

    @contextmanager
    def session(key):
        selected.append(key)
        yield SimpleNamespace(
            get=lambda: {"sandbox_id": "ours", "source_hash": "digest"},
            save=MagicMock(),
            bind_owner=MagicMock(),
        )

    app.state.orchestrator.registry.session.side_effect = session
    app.state.orchestrator.lakebox.get.return_value = {"status": "Running"}
    app.state.orchestrator.lakebox.api.return_value = {"status": "stopped"}
    response = client.post(
        "/sessions/stop", json={"agent_id": "agent-a", "session_id": "session-a"}
    )
    assert response.status_code == 200
    assert selected == [
        "invocation:" + scoped_session("agent-a", "session-a"),
        scoped_session("agent-a", "session-a"),
    ]
    assert response.json()["sandbox_id"] == "ours"
    assert response.json()["timings_ms"]["total"] >= 0


def test_stopping_a_published_environment_before_first_use_does_not_create_compute(client):
    selected = []

    @contextmanager
    def session(key):
        selected.append(key)
        yield SimpleNamespace(get=lambda: None)

    app.state.orchestrator.environment_session_id = "environment:published"
    app.state.orchestrator.registry.session.side_effect = session
    response = client.post("/sessions/stop", json={"agent_id": "a", "session_id": "s"})
    assert response.status_code == 404
    assert selected == ["invocation:" + scoped_session("a", "s"), scoped_session("a", "s")]
    app.state.orchestrator.lakebox.api.assert_not_called()
    app.state.orchestrator.lakebox.create.assert_not_called()


def test_stop_duration_includes_api_request_and_confirmed_shutdown_wait(client, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.session_client.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.session_client.time.sleep", lambda seconds: None
    )

    @contextmanager
    def session(key):
        yield SimpleNamespace(
            get=lambda: {"sandbox_id": "ours", "source_hash": "digest"},
            save=MagicMock(),
            bind_owner=MagicMock(),
        )

    def stop(*args):
        clock[0] += 2
        return {"status": "Stopping"}

    def stopped(sandbox_id):
        clock[0] += 15
        return {"status": "Stopped"}

    app.state.orchestrator.registry.session.side_effect = session
    app.state.orchestrator.lakebox.api.side_effect = stop
    lookups = iter([{"status": "Running"}])
    app.state.orchestrator.lakebox.get.side_effect = lambda sandbox_id: (
        next(lookups, None) or stopped(sandbox_id)
    )
    response = client.post("/sessions/stop", json={"agent_id": "a", "session_id": "s"})
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"
    assert response.json()["timings_ms"] == {
        "registry_connect_and_lock": 0,
        "sandbox_lookup": 0,
        "stop_request": 2000,
        "wait_until_stopped": 15000,
        "total": 17000,
    }
    assert app.state.orchestrator.lakebox.get.call_count == 2


@pytest.mark.parametrize("status", ["Stopped", "Terminated"])
def test_stop_is_idempotent_after_platform_auto_stop(client, status):
    saved = MagicMock()
    app.state.orchestrator.registry.session.return_value.__enter__.return_value = SimpleNamespace(
        get=lambda: {"sandbox_id": "ours", "source_hash": "digest"},
        save=saved,
        bind_owner=MagicMock(),
    )
    app.state.orchestrator.lakebox.get.return_value = {"status": status}
    response = client.post("/sessions/stop", json={"agent_id": "a", "session_id": "s"})
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"
    app.state.orchestrator.lakebox.api.assert_not_called()
    saved.assert_called_once_with("digest", "ours", "stopped", None)


@pytest.mark.parametrize("reply", [None, {}])
def test_empty_stop_reply_requires_observed_shutdown(client, monkeypatch, reply):
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.session_client.time.sleep", lambda seconds: None
    )
    saved = MagicMock()
    app.state.orchestrator.registry.session.return_value.__enter__.return_value = SimpleNamespace(
        get=lambda: {"sandbox_id": "ours", "source_hash": "digest"},
        save=saved,
        bind_owner=MagicMock(),
    )
    app.state.orchestrator.lakebox.api.return_value = reply
    app.state.orchestrator.lakebox.get.side_effect = [
        {"status": "Running"},
        {"status": "Stopping"},
        {"status": "Stopped"},
    ]
    response = client.post("/sessions/stop", json={"agent_id": "a", "session_id": "s"})
    assert response.status_code == 200
    assert app.state.orchestrator.lakebox.get.call_count == 3
    saved.assert_called_once_with("digest", "ours", "stopped", None)


def test_missing_sandbox_does_not_record_confirmed_shutdown(client, monkeypatch):
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.session_client.time.sleep", lambda seconds: None
    )
    saved = MagicMock()
    app.state.orchestrator.registry.session.return_value.__enter__.return_value = SimpleNamespace(
        get=lambda: {"sandbox_id": "ours", "source_hash": "digest"},
        save=saved,
        bind_owner=MagicMock(),
    )
    app.state.orchestrator.lakebox.api.return_value = {}
    app.state.orchestrator.lakebox.get.side_effect = [{"status": "Running"}, None]
    response = client.post("/sessions/stop", json={"agent_id": "a", "session_id": "s"})
    assert response.status_code == 404
    assert "confirmed" in response.json()["detail"]
    saved.assert_not_called()


def test_shutdown_deadline_does_not_record_success(client, monkeypatch):
    clock = iter([10, 10, 191])
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.session_client.time",
        SimpleNamespace(perf_counter=lambda: 0.0, monotonic=lambda: next(clock)),
    )
    saved = MagicMock()
    app.state.orchestrator.registry.session.return_value.__enter__.return_value = SimpleNamespace(
        get=lambda: {"sandbox_id": "ours", "source_hash": "digest"},
        save=saved,
        bind_owner=MagicMock(),
    )
    app.state.orchestrator.lakebox.get.return_value = {"status": "Running"}
    app.state.orchestrator.lakebox.api.return_value = {"status": "Stopping"}
    response = client.post("/sessions/stop", json={"agent_id": "a", "session_id": "s"})
    assert response.status_code == 504
    assert response.json()["detail"] == "Sandbox is still stopping"
    saved.assert_not_called()


def test_environment_creation_checks_source_access_before_saving_location(client):
    app.state.environments.create.return_value = {"name": "demo"}
    response = client.post(
        "/api/environments",
        json={
            "name": "demo",
            "source_path": "/Workspace/Users/me/agent",
            "entrypoint": "agent:invoke",
        },
    )
    assert response.status_code == 201
    app.state.sources.load.assert_called_once_with("workspace", "/Users/me/agent")
    app.state.environments.create.assert_called_once_with(
        "demo",
        "workspace",
        "/Users/me/agent",
        "agent:invoke",
        bundle=app.state.sources.load.return_value,
        git_source=None,
    )


def test_environment_run_uses_stable_environment_identity_and_dispatcher(client):
    app.state.environments.invoke.return_value = {"id": "run", "status": "queued"}
    response = client.post(
        "/api/environments/demo/runs",
        json={
            "session_id": "s",
            "invocation_id": "i",
            "input": {"message": "hi"},
            "retry": True,
        },
    )
    assert response.status_code == 202
    app.state.environments.invoke.assert_called_once_with(
        "demo", "s", "i", {"message": "hi"}, retry=True
    )
    app.state.runs.submit.assert_called_once_with("run")


def test_duplicate_environment_returns_conflict(client):
    from databricks_agentkit.sandbox.environments import EnvironmentExists

    app.state.environments.create.side_effect = EnvironmentExists("already exists")
    response = client.post(
        "/api/environments", json={"name": "demo", "source_path": "/Users/me/agent"}
    )
    assert response.status_code == 409


def test_git_creation_verifies_uploaded_bytes_before_registering_a_commit(client):
    files = {"pyproject.toml": b"project", "uv.lock": b"lock"}
    bundle = bundle_files(files)
    app.state.sources.load.return_value = bundle
    app.state.environments.create.return_value = {"name": "committed"}
    git = {
        "commit": "a" * 40,
        "requested_revision": "main",
        "repository_url": None,
        "subdirectory": "",
        "files_digest": files_digest(files),
    }
    request = {"name": "committed", "source_path": "/Users/me/git", "git_source": git}
    assert client.post("/api/environments", json=request).status_code == 201
    assert app.state.environments.create.call_args.kwargs["git_source"] == git
    app.state.environments.create.reset_mock()
    app.state.sources.load.return_value = bundle_files({**files, "changed.py": b"other"})
    response = client.post("/api/environments", json=request)
    assert response.status_code == 400
    assert "differs from the selected Git snapshot" in response.json()["detail"]
    app.state.environments.create.assert_not_called()


def test_source_view_reads_saved_bundle_and_escapes_source_content(client):
    bundle = bundle_files(
        {
            "pyproject.toml": b"project",
            "uv.lock": b"lock",
            "agent.py": b'<script>alert("example")</script>',
        }
    )
    app.state.control.agent.return_value = {
        "name": "Saved agent",
        "source_hash": bundle.digest,
        "bundle_data": bundle.data,
        "git_source": {"commit": "a" * 40, "subdirectory": "agent"},
    }
    response = client.get("/api/agents/pinned/source", params={"path": "agent.py"})
    assert response.status_code == 200
    assert "&lt;script&gt;alert(" in response.text
    assert "<script>alert(" not in response.text
    assert "a" * 40 in response.text
    app.state.sources.load.assert_not_called()
    assert client.get("/api/agents/pinned/source", params={"path": "../other"}).status_code == 404


@pytest.mark.parametrize("observed", ["Stopped", "Stopping"])
def test_cancelled_stop_response_is_reconciled_without_replaying_command(
    client, monkeypatch, observed
):
    from databricks.sdk.errors import Cancelled

    monkeypatch.setattr(
        "databricks_agentkit.sandbox.session_client.time.sleep", lambda seconds: None
    )
    saved = MagicMock()
    app.state.orchestrator.registry.session.return_value.__enter__.return_value = SimpleNamespace(
        get=lambda: {"sandbox_id": "ours", "source_hash": "digest"},
        save=saved,
        bind_owner=MagicMock(),
    )
    app.state.orchestrator.lakebox.get.side_effect = [
        {"status": "Running"},
        {"status": observed},
        {"status": "Stopped"},
    ]
    app.state.orchestrator.lakebox.api.side_effect = Cancelled("response cancelled")
    response = client.post("/sessions/stop", json={"agent_id": "a", "session_id": "s"})
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"
    app.state.orchestrator.lakebox.api.assert_called_once_with(
        "POST", "/sandboxes/ours/stop", {"sandbox_id": "ours"}
    )
    saved.assert_called_once_with("digest", "ours", "stopped", None)


@pytest.mark.parametrize("observed", [None, {"status": "Running"}])
def test_cancelled_stop_never_confirms_shutdown_without_observed_state(client, observed):
    from databricks.sdk.errors import Cancelled

    saved = MagicMock()
    app.state.orchestrator.registry.session.return_value.__enter__.return_value = SimpleNamespace(
        get=lambda: {"sandbox_id": "ours", "source_hash": "digest"},
        save=saved,
        bind_owner=MagicMock(),
    )
    app.state.orchestrator.lakebox.get.side_effect = [{"status": "Running"}, observed]
    app.state.orchestrator.lakebox.api.side_effect = Cancelled("response cancelled")
    response = client.post("/sessions/stop", json={"agent_id": "a", "session_id": "s"})
    assert response.status_code == 502
    app.state.orchestrator.lakebox.api.assert_called_once()
    saved.assert_not_called()


def test_create_session_is_async_and_does_not_create_a_run(client):
    app.state.preparations.create.return_value = {"session_id": "new", "status": "queued"}
    response = client.post("/api/agents/revision/sessions", json={"session_id": "new"})
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    app.state.preparations.create.assert_called_once_with("revision", "new", retry=False)
    app.state.control.create_run.assert_not_called()
    app.state.orchestrator.invoke.assert_not_called()


def test_session_status_and_list_do_not_prepare_compute(client):
    app.state.preparations.get.return_value = {"session_id": "new", "status": "ready"}
    app.state.preparations.rows.return_value = [
        {"preparation": {"session_id": "new", "status": "ready"}}
    ]
    app.state.control.sessions.return_value = []
    assert client.get("/api/agents/revision/sessions/new").json()["status"] == "ready"
    assert client.get("/api/agents/revision/sessions").json()[0]["session_id"] == "new"
    app.state.preparations.create.assert_not_called()


def test_invoke_while_preparing_returns_conflict(client):
    from databricks_agentkit.sandbox.store import SessionBusy

    app.state.preparations.assert_invocable.side_effect = SessionBusy("Wait for preparation")
    response = client.post("/api/runs", json={"agent_id": "revision", "session_id": "new"})
    assert response.status_code == 409
    app.state.control.create_run.assert_not_called()
