import pytest
from fastapi.testclient import TestClient
from runtime import ui

from databricks_mason import AgentApp
from databricks_mason.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
    RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
    RUNTIME_STORE_LOCAL_ENV,
    RUNTIME_STORE_SCHEMA_ENV,
    RUNTIME_STORE_USERNAME_ENV,
    InMemoryRuntimeStore,
)


@pytest.fixture(autouse=True)
def _clear_runtime_store_env(monkeypatch):
    for name in (
        RUNTIME_STORE_LOCAL_ENV,
        RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
        RUNTIME_STORE_DATABASE_ENV,
        RUNTIME_STORE_USERNAME_ENV,
        RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
        RUNTIME_STORE_SCHEMA_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


class _FakeStateClient:
    def create_memory_entry(self, actor, request, session_id):
        return {
            "name": "memory-stores/store/entries/entry",
            "session_id": session_id,
            "actor_id": actor,
            **request.model_dump(),
        }

    def list_memory_entries(self, actor, path_prefix=None):
        return {
            "managed_memory_entries": [
                {"path": f"{path_prefix or ''}/profile.md", "actor_id": actor}
            ]
        }

    def search_memory_entries(self, actor, request):
        return {
            "managed_memory_entries": [
                {"path": "/profile.md", "content": request.query, "actor_id": actor}
            ]
        }

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


class _FakeInterrupt:
    def __init__(self, value, id):
        self.value = value
        self.id = id


async def _session_history(session_id, actor):
    return {
        "session_id": session_id,
        "session_items": [
            {"item_id": "1", "data": {"type": "human", "content": session_id}},
            {"item_id": "2", "data": {"type": "ai", "content": "checkpoint reply"}},
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
        monkeypatch.setattr(ui, "_checkpoint_history", _session_history)
    # Keep model discovery deterministic and offline (no AI Gateway listing call).
    monkeypatch.setattr(ui, "_default_model", lambda: "system.ai.claude-sonnet-4-5")
    monkeypatch.setattr(ui, "_discover_chat_models", lambda: ["system.ai.claude-sonnet-4-5"])

    async def invoke_handler(request, context):
        return {"output": [], "session_id": context.session_id}

    app = AgentApp(runtime_store=InMemoryRuntimeStore())
    app.invoke(invoke_handler)
    app.recover(invoke_handler)
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
    assert index.headers["cache-control"] == "no-store"
    assert 'id="new-session"' in index.text
    assert 'id="session-list"' in index.text
    assert 'id="model-select"' in index.text
    assert 'id="app-settings-link"' in index.text
    assert 'id="app-logs-link"' in index.text
    app_script = client.get("/ui-assets/app.js")
    assert app_script.status_code == 200
    assert app_script.headers["cache-control"] == "no-store"
    assert "mason memory bind <store-name>" in app_script.text
    assert "refreshSessionView({ hydrateChat: true })" in app_script.text
    assert "function renderModels(" in app_script.text
    assert 'demoUrl("/api/ui/config")' in app_script.text
    assert 'demoUrl("/api/demo/models")' in app_script.text
    assert 'fetch("/api/session/new"' not in app_script.text
    assert "/api/demo/sessions/${encodeURIComponent(sessionId)}/open" in app_script.text
    assert "session_id: sessionId" in app_script.text
    assert 'fetch("/api/invocations"' in app_script.text
    styles = client.get("/ui-assets/styles.css").text
    assert "@media (min-width: 1181px)" in styles
    assert "scrollbar-gutter: stable" in styles

    config = client.get("/api/ui/config").json()
    assert config["session_id"] == "routing-session"
    assert config["deployed"] is False
    assert config["models"] == {
        "default": "system.ai.claude-sonnet-4-5",
        "available": ["system.ai.claude-sonnet-4-5"],
    }
    assert config["streaming"]["enabled"] is True
    assert config["background"]["enabled"] is True
    assert config["streaming"]["persistent"] is False
    assert config["streaming"]["mode"] == "In-process Runtime Store"
    assert config["background"]["persistent"] is False
    assert config["background"]["mode"] == "In-process Runtime Store"
    assert config["memory"]["enabled"] is False
    assert config["session"]["managed"] is False
    assert config["session"]["history"] is True
    assert "durability" not in config
    assert "recovery" not in config
    assert client.get("/api/demo/config").status_code == 404

    assert client.get("/api/demo/models").json() == {
        "default": "system.ai.claude-sonnet-4-5",
        "available": ["system.ai.claude-sonnet-4-5"],
    }

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


@pytest.mark.parametrize(
    ("local", "endpoint", "schema", "persistent"),
    [
        (None, None, None, False),
        ("true", None, None, False),
        ("true", "endpoint", "runtime_schema", False),
        ("TRUE", "endpoint", "runtime_schema", False),
        ("TrUe", "endpoint", "runtime_schema", False),
        (None, "endpoint", "runtime_schema", True),
        ("false", "endpoint", "runtime_schema", True),
        (None, "endpoint", None, False),
        (None, None, "runtime_schema", False),
        (None, "", "runtime_schema", False),
        (None, "endpoint", "", False),
    ],
)
def test_demo_config_reports_runtime_store_configuration(
    monkeypatch, local, endpoint, schema, persistent
):
    for name, value in (
        (RUNTIME_STORE_LOCAL_ENV, local),
        (RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV, endpoint),
        (RUNTIME_STORE_SCHEMA_ENV, schema),
    ):
        if value is not None:
            monkeypatch.setenv(name, value)
    client = _client(monkeypatch)

    response = client.get("/api/ui/config")
    assert response.status_code == 200
    config = response.json()
    for capability in ("streaming", "background"):
        assert config[capability]["enabled"] is True
        assert config[capability]["persistent"] is persistent
        assert config[capability]["mode"] == (
            "Runtime Store" if persistent else "In-process Runtime Store"
        )


def test_demo_config_reports_managed_runtime_store(monkeypatch):
    monkeypatch.setenv(RUNTIME_STORE_LAKEBASE_BRANCH_ENV, "projects/p/branches/b")
    monkeypatch.setenv(RUNTIME_STORE_DATABASE_ENV, "runtime-db")
    monkeypatch.setenv(RUNTIME_STORE_USERNAME_ENV, "app-sp")

    config = _client(monkeypatch).get("/api/ui/config").json()

    assert config["background"]["persistent"] is True


def test_demo_config_distinguishes_run_local_from_a_deployed_app(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "app")
    monkeypatch.setenv("DATABRICKS_APP_URL", "http://127.0.0.1:8000")
    assert _client(monkeypatch).get("/api/ui/config").json()["deployed"] is False

    monkeypatch.setenv("DATABRICKS_APP_URL", "https://agent.example.databricksapps.com")
    assert _client(monkeypatch).get("/api/ui/config").json()["deployed"] is True


def test_deployed_app_management_links_target_workspace_pages(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-mason-demo")
    monkeypatch.setenv("DATABRICKS_APP_URL", "https://demo.databricksapps.com")
    monkeypatch.setattr(ui, "_workspace_host", lambda: "https://example.databricks.com")

    assert ui._app_links() == {
        "name": "agent-mason-demo",
        "settings_url": "https://example.databricks.com/apps-v2/app/agent-mason-demo",
        "logs_url": "https://example.databricks.com/apps-v2/app/agent-mason-demo/logs",
    }


def test_demo_config_does_not_wait_for_model_discovery(monkeypatch):
    client = _client(monkeypatch)
    calls = []
    monkeypatch.setattr(
        ui,
        "_discover_chat_models",
        lambda: calls.append(True) or ["system.ai.claude-sonnet-4-5", "system.ai.llama-4-maverick"],
    )

    assert client.get("/api/ui/config").status_code == 200
    assert calls == []
    assert client.get("/api/demo/models").json()["available"] == [
        "system.ai.claude-sonnet-4-5",
        "system.ai.llama-4-maverick",
    ]
    assert calls == [True]


def test_unmanaged_checkpoint_history_route(monkeypatch):
    client = _client(monkeypatch, history=True, session_id="local-session")

    config = client.get("/api/ui/config").json()
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
            "/api/2.0/agents/session-stores/sessions/sessions",
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


def test_discover_chat_models_pins_default_and_dedups(monkeypatch):
    monkeypatch.setattr(ui, "_default_model", lambda: "system.ai.claude-sonnet-4-5")
    monkeypatch.setattr(
        ui,
        "list_ai_gateway_model_services",
        lambda _client: [
            "system.ai.llama-4-maverick",
            "system.ai.claude-sonnet-4-5",
            "system.ai.claude-opus-4-8",
        ],
    )
    monkeypatch.setattr(ui, "workspace_client", lambda: object())

    # Default pinned first, the rest alphabetical, and the default not repeated by discovery.
    assert ui._discover_chat_models() == [
        "system.ai.claude-sonnet-4-5",
        "system.ai.claude-opus-4-8",
        "system.ai.llama-4-maverick",
    ]


def test_discover_chat_models_falls_back_to_default_on_error(monkeypatch):
    monkeypatch.setattr(ui, "_default_model", lambda: "system.ai.claude-sonnet-4-5")

    def _boom(_client):
        raise PermissionError("cannot read system.ai")

    monkeypatch.setattr(ui, "list_ai_gateway_model_services", _boom)
    monkeypatch.setattr(ui, "workspace_client", lambda: object())

    # A workspace that can't list the gateway still gets a working picker.
    assert ui._discover_chat_models() == ["system.ai.claude-sonnet-4-5"]


def test_discover_chat_models_caps_the_picker(monkeypatch):
    monkeypatch.setattr(ui, "_default_model", lambda: "system.ai.claude-sonnet-4-5")
    monkeypatch.setattr(
        ui,
        "list_ai_gateway_model_services",
        # More services than the cap allows; synthetic because no real catalog is this long.
        lambda _client: [f"system.ai.test-model-{i:03d}" for i in range(30)],
    )
    monkeypatch.setattr(ui, "workspace_client", lambda: object())

    result = ui._discover_chat_models()
    assert len(result) == ui._MODEL_LIMIT
    assert result[0] == "system.ai.claude-sonnet-4-5"  # the default survives truncation


@pytest.mark.asyncio
async def test_checkpoint_history_reads_messages_and_interrupts(monkeypatch):
    import agent.agent as agent_module

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
                    "actor_id": "alice",
                }
            }
            return Snapshot()

    async def fake_create_agent_graph(actor):
        assert actor == "alice"
        return FakeAgent()

    monkeypatch.setattr(agent_module, "create_agent_graph", fake_create_agent_graph)
    result = await ui._checkpoint_history("saved-session", "alice")

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

    config = client.get("/api/ui/config").json()
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

    # The UI can browse another actor's memories via ?actor= (list) or payload.actor (search);
    # both default to the viewer when omitted.
    listed_default = client.get("/api/demo/memory/entries").json()
    assert listed_default["managed_memory_entries"][0]["actor_id"] == "alice"
    listed_bob = client.get("/api/demo/memory/entries", params={"actor": "bob"}).json()
    assert listed_bob["managed_memory_entries"][0]["actor_id"] == "bob"
    search_bob = client.post("/api/demo/memory/search", json={"query": "x", "actor": "bob"})
    assert search_bob.json()["managed_memory_entries"][0]["actor_id"] == "bob"

    # Tracing surfaces in config when a destination + experiment are set, with an experiment link.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "123456")
    monkeypatch.setattr(ui, "_workspace_host", lambda: "https://example.databricks.com")
    tracing = client.get("/api/ui/config").json()["tracing"]
    assert tracing["enabled"] is True
    assert tracing["url"] == "https://example.databricks.com/ml/experiments/123456"
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    assert client.get("/api/ui/config").json()["tracing"]["enabled"] is False

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
    assert client.get("/api/ui/config", params={"session_id": "s2"}).json()["session_id"] == "s2"
    assert (
        client.get("/api/demo/session/items", params={"session_id": "s2"}).json()["session_items"][
            0
        ]["data"]["content"]
        == "s2"
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
