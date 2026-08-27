from fastapi import FastAPI
from fastapi.testclient import TestClient
from runtime import ui


class _FakeStateClient:
    def create_memory_entry(self, request):
        return {"name": "memory-stores/store/entries/entry", **request.model_dump()}

    def list_memory_entries(self, path_prefix=None):
        return {"managed_memory_entries": [{"path": f"{path_prefix or ''}/profile.md"}]}

    def search_memory_entries(self, request):
        return {"managed_memory_entries": [{"path": "/profile.md", "content": request.query}]}

    def ensure_session(self, session_id):
        return {"session_id": session_id, "actor_id": "alice"}

    def get_session(self, session_id):
        return {"session_id": session_id, "actor_id": "alice"}

    def append_session_items(self, session_id, items):
        return {"session_items": [{"item_id": "1", "data": item} for item in items]}

    def list_session_items(self, session_id):
        return {
            "session_items": [{"item_id": "1", "data": {"role": "user", "content": session_id}}]
        }


async def _session_history(session_id):
    return {
        "session_id": session_id,
        "session_items": [
            {"item_id": "1", "data": {"type": "human", "content": session_id}},
            {"item_id": "2", "data": {"type": "ai", "content": "checkpoint reply"}},
        ],
        "interrupts": [],
    }


def _client(monkeypatch, *, configured=False, history=False):
    monkeypatch.delenv("MASON_DEMO_CRASH_ENABLED", raising=False)
    if configured:
        monkeypatch.setenv("AGENT_MEMORY_STORE", "memory-stores/store")
        monkeypatch.setenv("AGENT_MEMORY_ACTOR_ID", "alice")
        monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
        monkeypatch.setenv("AGENT_SESSION_ACTOR_ID", "alice")
        monkeypatch.setattr(ui, "_state_client", lambda: _FakeStateClient())
    else:
        monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
        monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    app = FastAPI()
    ui.install_ui(app, session_history=_session_history if history else None)
    return TestClient(app)


def test_demo_ui_routes(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/").status_code == 200
    assert client.get("/ui-assets/app.js").status_code == 200

    config = client.get("/api/demo/config").json()
    assert config["streaming"]["enabled"] is True
    assert config["background"]["enabled"] is True
    assert config["memory"]["enabled"] is False
    assert config["session"]["managed"] is False
    assert config["session"]["history"] is False
    assert config["crash"]["enabled"] is False

    assert client.post("/api/demo/memory/search", json={"query": "profile"}).status_code == 503
    assert client.post("/api/demo/sessions", json={"session_id": "s1"}).status_code == 503
    assert client.post("/api/demo/crash").status_code == 403


def test_unmanaged_checkpoint_history_route(monkeypatch):
    client = _client(monkeypatch, history=True)

    config = client.get("/api/demo/config").json()
    assert config["session"]["managed"] is False
    assert config["session"]["history"] is True

    result = client.get("/api/demo/sessions/local-session/items")
    assert result.status_code == 200
    assert [item["data"]["content"] for item in result.json()["session_items"]] == [
        "local-session",
        "checkpoint reply",
    ]


def test_managed_memory_and_session_routes(monkeypatch):
    client = _client(monkeypatch, configured=True)

    config = client.get("/api/demo/config").json()
    assert config["memory"] == {
        "enabled": True,
        "store": "memory-stores/store",
        "actor": "alice",
    }
    assert config["session"]["store"] == "sessions"
    assert config["session"]["actor"] == "alice"
    assert config["session"]["history"] is True

    created = client.post(
        "/api/demo/memory/entries",
        json={"path": "/profile.md", "content": "I work at Databricks"},
    )
    assert created.status_code == 200
    assert created.json()["path"] == "/profile.md"
    assert client.get("/api/demo/memory/entries", params={"path_prefix": "/"}).status_code == 200
    search = client.post("/api/demo/memory/search", json={"query": "Databricks"})
    assert search.json()["managed_memory_entries"][0]["content"] == "Databricks"

    assert client.post("/api/demo/sessions", json={"session_id": "s1"}).status_code == 200
    assert client.get("/api/demo/sessions/s1").json()["session_id"] == "s1"
    appended = client.post(
        "/api/demo/sessions/s1/items",
        json={"items": [{"role": "user", "content": "hello"}]},
    )
    assert appended.json()["session_items"][0]["data"]["content"] == "hello"
    assert (
        client.get("/api/demo/sessions/s1/items").json()["session_items"][0]["data"]["content"]
        == "s1"
    )
