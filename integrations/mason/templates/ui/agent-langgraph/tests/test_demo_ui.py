import pytest
from fastapi.testclient import TestClient
from runtime import ui
from runtime.runtime import build_app


class _FakeStateClient:
    def create_memory_entry(self, request, session_id):
        return {
            "name": "memory-stores/store/entries/entry",
            "session_id": session_id,
            **request.model_dump(),
        }

    def list_memory_entries(self, path_prefix=None):
        return {"managed_memory_entries": [{"path": f"{path_prefix or ''}/profile.md"}]}

    def search_memory_entries(self, request):
        return {
            "managed_memory_entries": [
                {"path": "/profile.md", "content": request.query}
            ]
        }

    def ensure_session(self, session_id):
        return {"session_id": session_id, "actor_id": "alice"}

    def get_session(self, session_id):
        return {"session_id": session_id, "actor_id": "alice"}

    def append_session_items(self, session_id, items):
        return {"session_items": [{"item_id": "1", "data": item} for item in items]}

    def list_session_items(self, session_id):
        return {
            "session_items": [
                {"item_id": "1", "data": {"role": "user", "content": session_id}}
            ]
        }


class _FakeInterrupt:
    def __init__(self, value, id):
        self.value = value
        self.id = id


async def _session_history(session_id):
    return {
        "session_id": session_id,
        "session_items": [
            {"item_id": "1", "data": {"type": "human", "content": session_id}},
            {"item_id": "2", "data": {"type": "ai", "content": "checkpoint reply"}},
        ],
        "interrupts": [],
    }


async def _recovery_status(session_id):
    return {
        "session_id": session_id,
        "status": "stopped",
        "steps": ["tool_step_1", "tool_step_2"],
        "outputs": [{"tool": "tool_step_1", "output": "done"}],
        "current_step": "tool_step_2",
        "worker_active": False,
        "owner_active": False,
        "needs_resume": True,
        "error": None,
        "instance_id": "instance-1",
        "step_seconds": 1,
        "execution_id": "execution-1",
        "attempt": 1,
        "owner_id": "instance-0",
        "heartbeat_at": "2026-08-28T00:00:00+00:00",
        "heartbeat_age_seconds": 11,
        "heartbeat_fresh": False,
        "heartbeat_interval_seconds": 3,
        "stale_after_seconds": 10,
        "durability_event_count": 4,
        "recent_durability_events": [],
        "claim_mode": "session_store_last_writer_wins",
        "atomic_claim": False,
    }


async def _recovery_start(session_id):
    result = await _recovery_status(session_id)
    return {
        **result,
        "status": "running",
        "worker_active": True,
        "owner_active": True,
        "heartbeat_fresh": True,
        "needs_resume": False,
    }


async def _recovery_resume(session_id):
    return await _recovery_start(session_id)


def _client(
    monkeypatch, *, configured=False, history=False, session_id="routing-session"
):
    monkeypatch.delenv("MASON_DEMO_TOOL_STEP_SECONDS", raising=False)
    if configured:
        monkeypatch.setenv("AGENT_MEMORY_STORE", "store")
        monkeypatch.setenv("AGENT_MEMORY_ACTOR_ID", "alice")
        monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
        monkeypatch.setenv("AGENT_SESSION_ACTOR_ID", "alice")
        monkeypatch.setattr(ui, "_state_client", lambda: _FakeStateClient())
    else:
        monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
        monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    monkeypatch.setattr(ui.recovery, "status", _recovery_status)
    monkeypatch.setattr(ui.recovery, "start", _recovery_start)
    monkeypatch.setattr(ui.recovery, "resume", _recovery_resume)
    if history:
        monkeypatch.setattr(ui, "_checkpoint_history", _session_history)

    async def invoke_handler(request):
        return {"output": [], "session_id": request["session_id"]}

    async def stream_handler(request):
        if False:
            yield request

    app = build_app(invoke_handler, stream_handler)
    ui.install_ui(app)
    client = TestClient(app)
    client.cookies.set("__Host-databricks-app-router", session_id)
    return client


def test_demo_ui_routes(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/").status_code == 200
    app_script = client.get("/ui-assets/app.js")
    assert app_script.status_code == 200
    assert "refreshSession({ hydrateChat: true })" in app_script.text
    assert "session_id: ensureSessionId()" not in app_script.text

    config = client.get("/api/demo/config").json()
    assert config["session_id"] == "routing-session"
    assert config["streaming"]["enabled"] is True
    assert config["background"]["enabled"] is True
    assert config["memory"]["enabled"] is False
    assert config["session"]["managed"] is False
    assert config["session"]["history"] is True
    assert config["app_control"]["stop_enabled"] is True
    assert config["durability"]["enabled"] is False
    assert config["heartbeat"]["enabled"] is False
    assert config["recovery"]["enabled"] is False

    assert (
        client.post("/api/demo/memory/search", json={"query": "profile"}).status_code
        == 503
    )
    assert (
        client.post("/api/demo/sessions", json={"session_id": "ignored"}).status_code
        == 503
    )


def test_unmanaged_checkpoint_history_route(monkeypatch):
    client = _client(monkeypatch, history=True, session_id="local-session")

    config = client.get("/api/demo/config").json()
    assert config["session"]["managed"] is False
    assert config["session"]["history"] is True

    result = client.get("/api/demo/session/items")
    assert result.status_code == 200
    assert [item["data"]["content"] for item in result.json()["session_items"]] == [
        "local-session",
        "checkpoint reply",
    ]


@pytest.mark.asyncio
async def test_checkpoint_history_reads_messages_and_interrupts(monkeypatch):
    import agent.agent as agent_module

    monkeypatch.delenv("AGENT_SESSION_ACTOR_ID", raising=False)

    class Message:
        id = "message-1"

        def model_dump(self):
            return {"type": "human", "content": "saved message"}

    class Snapshot:
        values = {"messages": [Message()]}
        tasks = [
            type(
                "Task",
                (),
                {"interrupts": [_FakeInterrupt({"approval": True}, "int-1")]},
            )()
        ]

    class FakeAgent:
        async def aget_state(self, config):
            assert config == {
                "configurable": {
                    "thread_id": "saved-session",
                    "actor_id": "saved-session",
                }
            }
            return Snapshot()

    async def fake_create_agent_graph():
        return FakeAgent()

    monkeypatch.setattr(agent_module, "create_agent_graph", fake_create_agent_graph)
    result = await ui._checkpoint_history("saved-session")

    assert result == {
        "session_id": "saved-session",
        "session_items": [
            {
                "item_id": "message-1",
                "data": {"type": "human", "content": "saved message"},
            }
        ],
        "interrupts": [{"id": "int-1", "value": {"approval": True}}],
    }


def test_managed_memory_and_session_routes(monkeypatch):
    client = _client(monkeypatch, configured=True, session_id="s1")

    config = client.get("/api/demo/config").json()
    assert config["memory"] == {
        "enabled": True,
        "store": "memory-stores/store",
        "actor": "alice",
    }
    assert config["session"]["store"] == "sessions"
    assert config["session"]["actor"] == "alice"
    assert config["session"]["history"] is True
    assert config["durability"] == {
        "enabled": True,
        "mode": "Session Store checkpoint + event log",
        "claim_mode": "Last-writer-wins demo lease",
        "atomic_claim": False,
    }
    assert config["heartbeat"] == {
        "enabled": True,
        "interval_seconds": 3,
        "stale_after_seconds": 10,
    }
    assert config["recovery"] == {
        "enabled": True,
        "automatic_resume": True,
        "steps": ["tool_step_1", "tool_step_2", "tool_step_3", "tool_step_4"],
        "step_seconds": 6,
    }

    created = client.post(
        "/api/demo/memory/entries",
        json={"path": "/profile.md", "content": "I work at Databricks"},
    )
    assert created.status_code == 200
    assert created.json()["path"] == "/profile.md"
    assert created.json()["session_id"] == "s1"
    assert (
        client.get("/api/demo/memory/entries", params={"path_prefix": "/"}).status_code
        == 200
    )
    search = client.post("/api/demo/memory/search", json={"query": "Databricks"})
    assert search.json()["managed_memory_entries"][0]["content"] == "Databricks"

    assert (
        client.post("/api/demo/sessions", json={"session_id": "ignored"}).json()[
            "session_id"
        ]
        == "s1"
    )
    assert client.get("/api/demo/session").json()["session_id"] == "s1"
    appended = client.post(
        "/api/demo/session/items",
        json={"items": [{"role": "user", "content": "hello"}]},
    )
    assert appended.json()["session_items"][0]["data"]["content"] == "hello"
    assert (
        client.get("/api/demo/session/items").json()["session_items"][0]["data"][
            "content"
        ]
        == "s1"
    )

    assert client.get("/api/demo/recovery").json()["needs_resume"] is True
    assert client.post("/api/demo/recovery/start").json()["worker_active"] is True
    assert client.post("/api/demo/app/start").json()["worker_active"] is True
    assert client.post("/api/demo/recovery/resume").json()["worker_active"] is True
