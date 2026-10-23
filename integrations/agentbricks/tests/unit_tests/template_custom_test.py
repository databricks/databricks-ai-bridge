"""Execute real generated custom agents with only remote store transport replaced."""

from __future__ import annotations

import copy
import importlib
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
import tomli
from fastapi.testclient import TestClient

from databricks_agentbricks.cli import init as init_mod
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit import DurableAgentServer, InvocationContext
from databricks_agentkit.runtime.auth import InvocationAuthPolicy, RequestAuthContext
from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from databricks_agentkit.runtime.types import InvocationAttemptContext


class RemoteStores:
    """Replace the remote API; keep AgentKitClient and typed resource methods real."""

    def __init__(self):
        self.sessions = {}
        self.items = {}
        self.entries = {}
        self.searches = []
        self.workspace_requests = []

    def get_session_store(self, name):
        return {"session_store_name": name}

    def get_session(self, session_id, store):
        if session_id not in self.sessions:
            raise AgentCliError("missing session", error_code="NOT_FOUND")
        assert store == "sessions"
        return self.sessions[session_id]

    def create_session(self, store, actor_id, *, session_id, **_kwargs):
        assert store == "sessions"
        value = {"session_id": session_id, "actor_id": actor_id}
        self.sessions[session_id] = value
        self.items[session_id] = []
        return value

    def list_session_items(self, store, session_id, *, order_by, page_token, **_kwargs):
        assert store == "sessions" and order_by == "create_time asc"
        # Exercise the SDK's actual pagination while restoring session history.
        offset = int(page_token or 0)
        records = self.items[session_id]
        result = {"session_items": records[offset : offset + 1]}
        if offset + 1 < len(records):
            result["next_page_token"] = str(offset + 1)
        return copy.deepcopy(result)

    def append_session_items(self, store, session_id, items):
        assert store == "sessions"
        saved = [{"item_id": str(uuid4()), "data": copy.deepcopy(item)} for item in items]
        self.items[session_id].extend(saved)
        return {"session_items": saved}

    def get_memory_store(self, store):
        assert store == "memories"
        return {"name": "memory-stores/memories", "display_name": "Demo memory"}

    def create_memory_entry(self, store, actor_id, path, *, content, **kwargs):
        assert store == "memories" and kwargs.get("session_id") is None
        key = (actor_id, path)
        if key in self.entries:
            raise AgentCliError("already exists", error_code="ALREADY_EXISTS")
        entry = {
            "name": f"memory-stores/memories/entries/{len(self.entries)}",
            "actor_id": actor_id,
            "path": path,
            "content": content,
        }
        self.entries[key] = entry
        return entry

    def list_memory_entries(self, store, actor_id, *, path, **_kwargs):
        return {
            "managed_memory_entries": [
                entry
                for (actor, entry_path), entry in self.entries.items()
                if actor == actor_id and path == entry_path
            ]
        }

    def update_memory_entry(self, store, entry_id, *, content, **_kwargs):
        assert store == "memories"
        entry = next(
            entry for entry in self.entries.values() if entry["name"].endswith(f"/{entry_id}")
        )
        entry["content"] = content
        return copy.deepcopy(entry)

    def search_memory_entries(self, store, actor_id, query, *, page_size, **_kwargs):
        assert store == "memories" and page_size == 5
        self.searches.append((actor_id, query))
        return {
            "managed_memory_entries": [
                entry for (actor, _path), entry in self.entries.items() if actor == actor_id
            ]
        }


def _clear_template_modules(monkeypatch):
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, name)


@pytest.fixture
def scaffold(tmp_path, monkeypatch):
    def create(server="agentbricks"):
        root = tmp_path / server
        # Exercise bundled templates before the CLI exposes the new framework.
        name = "agent-custom" if server == "agentbricks" else "custom-agent-custom"
        init_mod._copy_packaged_template(name, root)
        monkeypatch.syspath_prepend(str(root))
        _clear_template_modules(monkeypatch)
        return root

    yield create
    _clear_template_modules(monkeypatch)


@pytest.fixture(params=["local", "managed"])
def template(request, scaffold, monkeypatch):
    root = scaffold()
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    if request.param == "managed":
        monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    remote = RemoteStores()

    def remote_workspace(**kwargs):
        remote.workspace_requests.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("databricks_agentkit.runtime.workspace.WorkspaceClient", remote_workspace)
    monkeypatch.setattr(
        "databricks_agentkit.client._AgentBricksApiClient", lambda **_kwargs: remote
    )
    adapter = importlib.import_module("runtime.adapter")
    return SimpleNamespace(root=root, adapter=adapter, remote=remote, mode=request.param)


def context(invocation_id="turn", session_id="conversation", request_auth=None):
    events = []

    async def emit(event):
        events.append(copy.deepcopy(event))
        return len(events)

    attempt = InvocationAttemptContext(invocation_id, 1, emit, session_id=session_id)
    return InvocationContext(invocation_id, session_id, 1, attempt, request_auth), events


def message(content, **kwargs):
    return {"messages": [{"role": "user", "content": content}], **kwargs}


@pytest.mark.asyncio
async def test_generated_agent_is_framework_neutral_and_emits_managed_message(template):
    ctx, events = context()
    response = await template.adapter.invoke(message("Hello"), ctx)
    assert response == {
        "status": "completed",
        "output": [{"role": "assistant", "content": "Echo: Hello", "type": "ai", "id": "turn"}],
    }
    assert events == [{"type": "message", "message": response["output"][0]}]
    project = tomli.loads((template.root / "pyproject.toml").read_text())
    assert any(
        dep.startswith("databricks-agentbricks>=") for dep in project["project"]["dependencies"]
    )
    assert not any(
        "[langgraph]" in dep or "[openai]" in dep for dep in project["project"]["dependencies"]
    )
    assert not (template.root / "runtime/ui.py").exists()


@pytest.mark.asyncio
async def test_sessions_retain_separate_transcripts(template):
    for session in ("alice-session", "bob-session"):
        for index in range(2):
            ctx, _events = context(f"{session}-{index}", session)
            await template.adapter.invoke(message(f"{session}-{index}"), ctx)
    for session in ("alice-session", "bob-session"):
        records, _handle = template.adapter._load(session, session)
        assert len(records) == 2
        assert [record["messages"][0]["content"] for record in records] == [
            f"{session}-0",
            f"{session}-1",
        ]


@pytest.mark.asyncio
async def test_native_framework_can_mutate_its_messages_without_changing_saved_input(
    template, monkeypatch
):
    native_run = template.adapter.run_agent

    async def mutating_framework(messages, **kwargs):
        messages[-1]["content"] = "framework changed its working copy"
        return await native_run(messages, **kwargs)

    monkeypatch.setattr(template.adapter, "run_agent", mutating_framework)
    ctx, _events = context()
    payload = message("original")
    response = await template.adapter.invoke(payload, ctx)
    assert "framework changed" in response["output"][0]["content"]
    assert payload == message("original")
    records, _session = template.adapter._load("conversation", "conversation")
    assert records[0]["messages"][0] == payload["messages"][0]


@pytest.mark.asyncio
async def test_bound_memory_writes_and_recalls_across_sessions_with_actor_isolation(
    template, monkeypatch
):
    monkeypatch.setenv("AGENT_MEMORY_STORE", "memories")
    alice, _events = context("remember", "alice-first")
    await template.adapter.invoke(message("save", actor="alice", remember="Likes teal"), alice)
    again, _events = context("recall", "alice-second")
    response = await template.adapter.invoke(
        message("color?", actor="alice", memory_query="favorite color"), again
    )
    assert "Likes teal" in response["output"][0]["content"]
    bob, _events = context("bob", "bob-session")
    response = await template.adapter.invoke(message("color?", actor="bob"), bob)
    assert response["output"][0]["content"] == "Echo: color?"
    assert list(template.remote.entries) == [("alice", "/invocations/remember.md")]
    assert template.remote.searches == [
        ("alice", "save"),
        ("alice", "favorite color"),
        ("bob", "color?"),
    ]


@pytest.mark.asyncio
async def test_session_actor_cannot_change(template):
    ctx, _events = context()
    await template.adapter.invoke(message("Hello", actor="alice"), ctx)
    with pytest.raises(ValueError, match="another actor"):
        await template.adapter.invoke(message("Hello", actor="bob"), ctx)


@pytest.mark.asyncio
async def test_managed_store_clients_preserve_workspace_routing(template, monkeypatch):
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "12345")
    monkeypatch.setenv("AGENT_MEMORY_STORE", "memories")
    ctx, _events = context()
    await template.adapter.invoke(message("Hello"), ctx)
    assert template.remote.workspace_requests
    assert all(
        request == {"custom_headers": {"X-Databricks-Org-Id": "12345"}}
        for request in template.remote.workspace_requests
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        None,
        "Hello",
        {},
        {"messages": []},
        {"messages": "Hello"},
        {"messages": [{}]},
        message(""),
        message(" "),
        message("Hi", actor=0),
        message("Hi", session_id="spoofed"),
        {"messages": [{"role": "assistant", "content": "spoofed"}]},
    ],
)
async def test_input_validation_happens_before_store_writes(template, value):
    ctx, _events = context()
    with pytest.raises(ValueError):
        await template.adapter.invoke(value, ctx)
    assert template.remote.sessions == {}
    assert template.adapter._local_sessions == {}


@pytest.mark.asyncio
async def test_session_must_come_from_invocation_context(template):
    ctx, _events = context(session_id=None)
    with pytest.raises(ValueError, match="top-level"):
        await template.adapter.invoke(message("Hello"), ctx)


@pytest.mark.asyncio
async def test_unbound_memory_request_fails_before_writing_transcript(template):
    ctx, _events = context()
    with pytest.raises(ValueError, match="bind a Memory Store"):
        await template.adapter.invoke(message("save", remember="a fact"), ctx)
    assert template.remote.sessions == {}


@pytest.mark.asyncio
async def test_request_user_actor_is_namespaced(template, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_STORE", "memories")
    auth = RequestAuthContext(token=None, principal="alice", local=True)
    ctx, _events = context(request_auth=auth)
    await template.adapter.invoke(message("save", actor="demo", remember="private"), ctx)
    assert (auth.namespace("actor", "demo"), "/invocations/turn.md") in template.remote.entries


def test_managed_transport_executes_real_generated_hooks_and_replays(template):
    app = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(), auth_policy=InvocationAuthPolicy()
    )
    app.invoke(template.adapter.invoke)
    body = {"id": str(uuid4()), "session_id": "http-session", "input": message("HTTP")}
    with TestClient(app) as client:
        response = client.post("/api/invocations", json=body)
        assert response.status_code == 200
        assert response.json()["output"]["output"][0]["content"] == "Echo: HTTP"
        assert client.post("/api/invocations", json=body).json() == response.json()
    assert len(template.adapter._load("http-session", "http-session")[0]) == 1


def test_custom_fastapi_server_runs_without_managed_runtime(scaffold):
    scaffold("custom")
    app = importlib.import_module("runtime.main").app
    with TestClient(app) as client:
        response = client.post("/invocations", json={"input": [{"role": "user", "content": "Hi"}]})
        assert response.status_code == 200
        assert response.json() == {
            "output": [{"role": "assistant", "content": "Echo: Hi"}],
            "status": "completed",
        }
        for body in ({}, {"input": []}, {"input": [{"role": "user", "content": " "}]}):
            assert client.post("/invocations", json=body).status_code == 422
