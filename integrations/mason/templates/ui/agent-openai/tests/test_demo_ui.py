import pytest
from fastapi.testclient import TestClient
from runtime import ui
from runtime.runtime import build_app


class _FakeStateClient:
    def create_memory_entry(self, actor, request, session_id):
        return {
            "name": "memory-stores/store/entries/entry",
            "session_id": session_id,
            "actor_id": actor,
            **request.model_dump(),
        }

    def list_memory_entries(self, actor, path_prefix=None):
        return {"managed_memory_entries": [{"path": f"{path_prefix or ''}/profile.md"}]}

    def search_memory_entries(self, actor, request):
        return {"managed_memory_entries": [{"path": "/profile.md", "content": request.query}]}

    def ensure_session(self, actor, session_id):
        return {"session_id": session_id, "actor_id": actor}

    def get_session(self, session_id):
        return {"session_id": session_id, "actor_id": "alice"}

    def list_sessions(self, actor):
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
                    "session_id": "public-s1",
                    "actor_id": "alice",
                    "metadata": {"public_session_id": "s1"},
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
            ]
        }


async def _session_history(session_id):
    return {
        "session_id": session_id,
        "session_items": [
            {"item_id": "1", "data": {"role": "user", "content": session_id}},
            {"item_id": "2", "data": {"role": "assistant", "content": "in-process reply"}},
        ],
        "interrupts": [],
    }


def _client(monkeypatch, *, configured=False, history=False, session_id="routing-session"):
    if configured:
        monkeypatch.setenv("AGENT_MEMORY_STORE", "store")
        monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
        monkeypatch.setattr(ui, "_state_client", lambda: _FakeStateClient())
    else:
        monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
        monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    if history:
        monkeypatch.setattr(ui, "_local_history", _session_history)

    async def invoke_handler(request):
        return {"output": [], "session_id": request["session_id"]}

    async def stream_handler(request):
        if False:
            yield request

    app = build_app(invoke_handler, stream_handler)
    ui.install_ui(app)
    client = TestClient(app, base_url="https://testserver")
    client.cookies.set("__Host-databricks-app-router", session_id)
    if configured:
        # The actor is the signed-in user from this forwarded-identity header (ui._request_actor);
        # unconfigured requests have no header and fall back to the "agent" actor.
        client.headers["X-Forwarded-Email"] = "alice"
    return client


def test_demo_ui_routes(monkeypatch):
    client = _client(monkeypatch)

    index = client.get("/")
    assert index.status_code == 200
    assert 'id="new-session"' in index.text
    assert 'id="session-list"' in index.text
    app_script = client.get("/ui-assets/app.js")
    assert app_script.status_code == 200
    assert "startDraft({ waiting: true })" in app_script.text
    assert "Waiting for agent response." in app_script.text
    assert "mason dev --memory <store-name>" in app_script.text
    assert "mason deploy <app-name> --source . --memory <store-name>" in app_script.text
    assert "refreshSessionView({ hydrateChat: true })" in app_script.text
    assert 'fetch("/api/session/new"' in app_script.text
    assert "/api/demo/sessions/${encodeURIComponent(sessionId)}/open" in app_script.text
    assert "session_id: ensureSessionId()" not in app_script.text
    styles = client.get("/ui-assets/styles.css").text
    assert "@media (min-width: 1181px)" in styles
    assert ".message.waiting .message-content::before" in styles
    assert "@keyframes response-spin" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert "scrollbar-gutter: stable" in styles

    config = client.get("/api/demo/config").json()
    assert config["session_id"] == "routing-session"
    assert config["deployed"] is False
    assert config["streaming"]["enabled"] is True
    assert config["background"]["enabled"] is True
    assert config["memory"]["enabled"] is False
    assert config["session"]["managed"] is False
    assert config["session"]["history"] is True
    assert config["session"]["mode"] == "In-process session"
    assert "durability" not in config
    assert "recovery" not in config

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


def test_demo_config_distinguishes_run_local_from_a_deployed_app(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "app")
    monkeypatch.setenv("DATABRICKS_APP_URL", "http://127.0.0.1:8000")
    assert _client(monkeypatch).get("/api/demo/config").json()["deployed"] is False

    monkeypatch.setenv("DATABRICKS_APP_URL", "https://agent.example.databricksapps.com")
    assert _client(monkeypatch).get("/api/demo/config").json()["deployed"] is True


def test_unmanaged_local_history_route(monkeypatch):
    client = _client(monkeypatch, history=True, session_id="local-session")

    config = client.get("/api/demo/config").json()
    assert config["session"]["managed"] is False
    assert config["session"]["history"] is True

    result = client.get("/api/demo/session/items")
    assert result.status_code == 200
    assert [item["data"]["content"] for item in result.json()["session_items"]] == [
        "local-session",
        "in-process reply",
    ]


def test_managed_session_list_is_actor_scoped(monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    state_client = object.__new__(ui._ManagedStateClient)
    calls = []
    state_client._do = lambda method, path, **kwargs: calls.append((method, path, kwargs)) or {
        "sessions": []
    }

    # The actor (a signed-in user) is escaped into the list filter.
    assert state_client.list_sessions('alice "demo"') == {"sessions": []}
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


def test_chat_session_items_exclude_non_message_items():
    result = ui._chat_session_items(
        {
            "session_items": [
                {"item_id": "1", "data": {"role": "user", "content": "hello"}},
                {"item_id": "2", "data": {"type": "ai", "content": "hi"}},
                {"item_id": "3", "data": {"event_type": "checkpoint"}},
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
async def test_local_history_reads_messages_from_in_process_session(monkeypatch):
    import databricks_mason.openai.sessions as ss

    class _FakeSession:
        async def get_items(self, limit=None):
            return [
                {"id": "m1", "role": "user", "content": "saved message"},
                {"role": "assistant", "content": "saved reply"},
            ]

    monkeypatch.setattr(ss, "session_store", lambda session_id: _FakeSession())
    result = await ui._local_history("saved-session")

    assert result == {
        "session_id": "saved-session",
        "session_items": [
            {"item_id": "m1", "data": {"id": "m1", "role": "user", "content": "saved message"}},
            {"item_id": "1", "data": {"role": "assistant", "content": "saved reply"}},
        ],
        "interrupts": [],
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


def test_open_session_rejects_another_actor(monkeypatch):
    client = _client(monkeypatch, configured=True, session_id="s1")

    class _ForeignActorClient(_FakeStateClient):
        def get_session(self, session_id):
            return {"session_id": session_id, "actor_id": "bob"}

    monkeypatch.setattr(ui, "_state_client", lambda: _ForeignActorClient())

    response = client.post("/api/demo/sessions/s2/open")
    assert response.status_code == 403
    assert response.json()["detail"] == "Session belongs to another actor."
