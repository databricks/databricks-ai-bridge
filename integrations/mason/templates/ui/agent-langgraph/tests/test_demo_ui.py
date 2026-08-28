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
        return {"managed_memory_entries": [{"path": "/profile.md", "content": request.query}]}

    def ensure_session(self, session_id):
        return {"session_id": session_id, "actor_id": "alice"}

    def get_session(self, session_id):
        return {"session_id": session_id, "actor_id": "alice"}

    def list_sessions(self):
        return {
            "sessions": [
                {
                    "session_id": "s1",
                    "actor_id": "alice",
                    "last_activity_time": "2026-08-28T12:00:00Z",
                },
                {
                    "session_id": "s2",
                    "actor_id": "alice",
                    "last_activity_time": "2026-08-27T12:00:00Z",
                },
                {
                    "session_id": "durability-s1",
                    "actor_id": "alice",
                    "metadata": {
                        "client": "mason-demo-durability",
                        "public_session_id": "s1",
                    },
                    "last_activity_time": "2026-08-28T12:01:00Z",
                },
            ]
        }

    def append_session_items(self, session_id, items):
        return {"session_items": [{"item_id": "1", "data": item} for item in items]}

    def list_session_items(self, session_id):
        return {
            "session_items": [
                {"item_id": "1", "data": {"role": "user", "content": session_id}},
                {
                    "item_id": "2",
                    "data": {"type": "assistant", "content": "saved reply"},
                },
                {
                    "item_id": "3",
                    "data": {"event_type": "checkpoint", "checkpoint_id": "checkpoint-1"},
                },
                {
                    "item_id": "4",
                    "data": {
                        "event_type": "mason_demo_durability",
                        "event": "heartbeat",
                    },
                },
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


def _client(monkeypatch, *, configured=False, history=False, session_id="routing-session"):
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
    client = TestClient(app, base_url="https://testserver")
    client.cookies.set("__Host-databricks-app-router", session_id)
    return client


def test_demo_ui_routes(monkeypatch):
    client = _client(monkeypatch)

    index = client.get("/")
    assert index.status_code == 200
    assert 'id="new-session"' in index.text
    assert 'id="session-list"' in index.text
    app_script = client.get("/ui-assets/app.js")
    assert app_script.status_code == 200
    assert "refreshSessionView({ hydrateChat: true })" in app_script.text
    assert 'fetch("/api/session/new"' in app_script.text
    assert "/api/demo/sessions/${encodeURIComponent(sessionId)}/open" in app_script.text
    assert "session_id: ensureSessionId()" not in app_script.text
    styles = client.get("/ui-assets/styles.css").text
    assert "@media (min-width: 1181px)" in styles
    assert "scrollbar-gutter: stable" in styles

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

    sessions = client.get("/api/demo/sessions").json()
    assert sessions == {
        "sessions": [
            {
                "session_id": "routing-session",
                "actor_id": "agent",
                "metadata": {"client": "mason-demo-ui-local"},
            }
        ],
        "current_session_id": "routing-session",
        "managed": False,
    }

    assert client.post("/api/demo/memory/search", json={"query": "profile"}).status_code == 503
    assert client.post("/api/demo/sessions", json={"session_id": "ignored"}).status_code == 503


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


def test_managed_session_list_is_actor_scoped(monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    monkeypatch.setenv("AGENT_SESSION_ACTOR_ID", 'alice "demo"')
    state_client = object.__new__(ui._ManagedStateClient)
    calls = []
    state_client._do = lambda method, path, **kwargs: calls.append((method, path, kwargs)) or {
        "sessions": []
    }

    assert state_client.list_sessions() == {"sessions": []}
    assert calls == [
        (
            "GET",
            "/api/agents/v1/session-stores/sessions/sessions",
            {
                "query": {
                    "filter": 'actor_id = "alice \\"demo\\""',
                    "order_by": "last_activity_time desc",
                    "page_size": 50,
                }
            },
        )
    ]


def test_chat_session_items_exclude_checkpoints_and_durability_events():
    result = ui._chat_session_items(
        {
            "session_items": [
                {"item_id": "1", "data": {"role": "user", "content": "hello"}},
                {"item_id": "2", "data": {"type": "ai", "content": "hi"}},
                {"item_id": "3", "data": {"event_type": "checkpoint"}},
                {
                    "item_id": "4",
                    "data": {"event_type": "mason_demo_durability", "event": "heartbeat"},
                },
                {"item_id": "5", "data": {"content": "missing role"}},
            ],
            "next_page_token": "next",
        }
    )

    assert result == {
        "session_items": [
            {"item_id": "1", "data": {"role": "user", "content": "hello"}},
            {"item_id": "2", "data": {"type": "ai", "content": "hi"}},
        ],
        "next_page_token": "next",
    }


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
    assert client.get("/api/demo/memory/entries", params={"path_prefix": "/"}).status_code == 200
    search = client.post("/api/demo/memory/search", json={"query": "Databricks"})
    assert search.json()["managed_memory_entries"][0]["content"] == "Databricks"

    assert (
        client.post("/api/demo/sessions", json={"session_id": "ignored"}).json()["session_id"]
        == "s1"
    )
    listed = client.get("/api/demo/sessions").json()
    assert [session["session_id"] for session in listed["sessions"]] == ["s1", "s2"]
    assert listed["current_session_id"] == "s1"
    assert listed["managed"] is True
    assert client.get("/api/demo/session").json()["session_id"] == "s1"
    appended = client.post(
        "/api/demo/session/items",
        json={"items": [{"role": "user", "content": "hello"}]},
    )
    assert appended.json()["session_items"][0]["data"]["content"] == "hello"
    assert (
        client.get("/api/demo/session/items").json()["session_items"][0]["data"]["content"] == "s1"
    )
    assert [
        item["data"]["content"]
        for item in client.get("/api/demo/session/items").json()["session_items"]
    ] == ["s1", "saved reply"]

    opened = client.post("/api/demo/sessions/s2/open")
    assert opened.json() == {
        "session_id": "s2",
        "previous_session_id": "s1",
        "managed": True,
    }
    assert client.get("/api/demo/config").json()["session_id"] == "s2"
    assert (
        client.get("/api/demo/session/items").json()["session_items"][0]["data"]["content"] == "s2"
    )

    assert client.get("/api/demo/recovery").json()["needs_resume"] is True
    assert client.post("/api/demo/recovery/start").json()["worker_active"] is True
    assert client.post("/api/demo/app/start").json()["worker_active"] is True
    assert client.post("/api/demo/recovery/resume").json()["worker_active"] is True


def test_open_session_rejects_another_actor(monkeypatch):
    client = _client(monkeypatch, configured=True, session_id="s1")

    class _ForeignActorClient(_FakeStateClient):
        def get_session(self, session_id):
            return {"session_id": session_id, "actor_id": "bob"}

    monkeypatch.setattr(ui, "_state_client", lambda: _ForeignActorClient())

    response = client.post("/api/demo/sessions/s2/open")
    assert response.status_code == 403
    assert response.json()["detail"] == "Session belongs to another actor."
