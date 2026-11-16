"""Per-user memory on Databricks managed memory: store ids, the worker's memory tools, launcher and App wiring."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import AlreadyExists, NotFound, PermissionDenied

from databricks_anthropic.sandbox import memory as mem
from databricks_anthropic.sandbox.launcher import DatabricksSandboxLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

KEY = "k" * 64
ALICE, BOB = "1001", "2002"


def load_worker():
    spec = importlib.util.find_spec("databricks_anthropic.sandbox.worker")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMemoryApi:
    """The memory REST API's semantics for one caller (``api_client.do``), kept in memory."""

    def __init__(self, caller: str, stores: dict[str, dict] | None = None) -> None:
        self.caller = caller
        self.stores: dict[str, dict] = stores if stores is not None else {}
        self.calls: list[tuple[str, str]] = []
        self.fail: Exception | None = None
        self._next = 0

    @property
    def api_client(self):
        return self

    def do(
        self, method: str, path: str, query: dict | None = None, body: dict | None = None
    ) -> Any:
        self.calls.append((method, path))
        if self.fail is not None:
            raise self.fail
        query, body = query or {}, body or {}
        parts = (
            path.removeprefix(mem.MEMORY_API).strip("/").split("/")
            if path != mem.MEMORY_API
            else []
        )
        if not parts:
            if method == "POST":
                sid = query["managed_memory_store_id"]
                if sid in self.stores:
                    raise AlreadyExists("exists")
                self.stores[sid] = {
                    "name": f"memory-stores/{sid}",
                    "owner_user_id": self.caller,
                    "entries": {},
                }
                return {k: v for k, v in self.stores[sid].items() if k != "entries"}
            return {"managed_memory_stores": [{"name": s["name"]} for s in self.stores.values()]}
        sid = parts[0].split(":")[0]
        store = self.stores.get(sid)
        if store is None:
            raise NotFound("no store")
        if store["owner_user_id"] != self.caller:
            raise PermissionDenied("no access")
        entries = store["entries"]
        if len(parts) == 1:
            if method == "DELETE":
                del self.stores[sid]
                return {}
            return {k: v for k, v in store.items() if k != "entries"}
        if parts[1] == "entries" and len(parts) == 2:
            if method == "POST":
                key = (body["actor_id"], body.get("session_id"), body["path"])
                if any(
                    (e["actor_id"], e.get("session_id"), e["path"]) == key for e in entries.values()
                ):
                    raise AlreadyExists("ALREADY_EXISTS")
                self._next += 1
                eid = f"e{self._next}"
                entries[eid] = {**body, "name": f"memory-stores/{sid}/entries/{eid}"}
                return entries[eid]
            found = [e for e in entries.values() if e["actor_id"] == query["actor_id"]]
            if query.get("path"):
                found = [e for e in found if e["path"] == query["path"]]
            if query.get("path_prefix"):
                found = [e for e in found if e["path"].startswith(query["path_prefix"])]
            return {"managed_memory_entries": found}
        if parts[1] == "entries:search":
            found = [e for e in entries.values() if e["actor_id"] == body["actor_id"]][
                : body["limit"]
            ]
            return {
                "results": [{"managed_memory_entry": e, "score": 1.0} for e in found],
                "managed_memory_entries": found,
            }
        eid = parts[2]
        if method == "PATCH":
            entries[eid].update(body)
            return entries[eid]
        if method == "DELETE":
            del entries[eid]
            return {}
        raise AssertionError(f"unexpected {method} {path}")


def configure(monkeypatch, cfg: mem.MemoryConfig) -> None:
    for key, value in cfg.to_env().items():
        monkeypatch.setenv(key, value)


async def tools_for(monkeypatch, api: FakeMemoryApi, cfg: mem.MemoryConfig) -> dict[str, Any]:
    configure(monkeypatch, cfg)
    worker = load_worker()
    return {t.name: t for t in await worker._open_memory_tools(api)}


async def call(tool, **args) -> dict:
    out = await tool.call(args)
    return json.loads(out if isinstance(out, str) else "".join(b.get("text", "") for b in out))


# ---------------------------------------------------------------- ids and config


def test_store_id_is_stable_valid_and_keyed():
    a = mem.memory_store_id(KEY, ALICE)
    assert a == mem.memory_store_id(KEY, ALICE)
    assert a != mem.memory_store_id(KEY, BOB)
    assert a != mem.memory_store_id(
        "other-key", ALICE
    )  # unpredictable without the deployment's key
    assert a.startswith("cma-memory-") and 3 <= len(a) <= 56 and ALICE not in a


@pytest.mark.parametrize("key, user", [("", ALICE), (KEY, "")])
def test_store_id_needs_a_key_and_a_user(key, user):
    with pytest.raises(ValueError):
        mem.memory_store_id(key, user)


def test_config_env_and_actor():
    cfg = mem.MemoryConfig.for_user(KEY, ALICE)
    assert cfg.actor_id == "databricks-user:1001"
    assert cfg.to_env() == {
        "CMA_MEMORY_STORE": cfg.store_id,
        "CMA_MEMORY_ACTOR": "databricks-user:1001",
        "CMA_MEMORY_OWNER": ALICE,
    }


def test_config_rejects_a_bad_store_id():
    with pytest.raises(ValueError):
        mem.MemoryConfig("Bad_Id", "databricks-user:1", "1")


def test_launcher_passes_memory_to_the_worker_env():
    cfg = mem.MemoryConfig.for_user(KEY, ALICE)
    launcher = DatabricksSandboxLauncher(
        workspace_client=MagicMock(), environment_id="env_1", environment_key="sk-env", memory=cfg
    )
    work = SimpleNamespace(id="work_1", data=SimpleNamespace(id="sesn_1"))
    env = launcher._worker_env(work, "sandboxes/x")  # type: ignore[arg-type]
    assert {k: env[k] for k in cfg.to_env()} == cfg.to_env()
    plain = DatabricksSandboxLauncher(
        workspace_client=MagicMock(), environment_id="env_1", environment_key="sk"
    )
    assert "CMA_MEMORY_STORE" not in plain._worker_env(work, "sandboxes/x")  # type: ignore[arg-type]


# ---------------------------------------------------------------- worker tools


def test_worker_tools_match_the_declarations(monkeypatch):
    worker = load_worker()
    assert worker.MEMORY_API == mem.MEMORY_API
    assert (
        worker.MEMORY_MAX_CONTENT == mem.MAX_CONTENT_CHARS
        and worker.MEMORY_MAX_RESULTS == mem.MAX_SEARCH_RESULTS
    )
    import asyncio

    tools = asyncio.run(
        tools_for(monkeypatch, FakeMemoryApi(ALICE), mem.MemoryConfig.for_user(KEY, ALICE))
    )
    declared = {d["name"]: d for d in mem.memory_custom_tools()}
    assert set(tools) == set(declared)
    for name, tool in tools.items():
        served = tool.to_dict()["input_schema"]
        assert set(served["properties"]) == set(declared[name]["input_schema"]["properties"]), name
        assert set(served.get("required", [])) == set(
            declared[name]["input_schema"].get("required", [])
        ), name


def test_memory_is_off_without_the_env(monkeypatch):
    for key in ("CMA_MEMORY_STORE", "CMA_MEMORY_ACTOR", "CMA_MEMORY_OWNER"):
        monkeypatch.delenv(key, raising=False)
    import asyncio

    assert asyncio.run(load_worker()._open_memory_tools(FakeMemoryApi(ALICE))) == []


@pytest.mark.asyncio
async def test_missing_store_is_created_and_owned_by_the_sandbox_identity(monkeypatch):
    cfg = mem.MemoryConfig.for_user(KEY, ALICE)
    api = FakeMemoryApi(ALICE)
    await tools_for(monkeypatch, api, cfg)
    assert api.stores[cfg.store_id]["owner_user_id"] == ALICE


@pytest.mark.asyncio
async def test_a_store_owned_by_someone_else_refuses_to_start(monkeypatch):
    cfg = mem.MemoryConfig.for_user(KEY, ALICE)

    class ReadableButBobs(FakeMemoryApi):
        def do(
            self, method: str, path: str, query: dict | None = None, body: dict | None = None
        ) -> Any:
            return {"name": "x", "owner_user_id": BOB}

    api = ReadableButBobs(ALICE)
    configure(monkeypatch, cfg)
    worker = load_worker()
    with pytest.raises(worker.MemoryStoreNotOwned):
        await worker._open_memory_tools(api)


@pytest.mark.asyncio
async def test_save_search_list_update_delete(monkeypatch):
    cfg = mem.MemoryConfig.for_user(KEY, ALICE)
    api = FakeMemoryApi(ALICE)
    tools = await tools_for(monkeypatch, api, cfg)
    saved = await call(
        tools["memory_save"], path="/prefs/style.md", content="tables", description="style"
    )
    assert saved == {"status": "created", "path": "/prefs/style.md"}
    again = await call(
        tools["memory_save"],
        path="/prefs/style.md",
        content="tables, no emoji",
        description="style",
    )
    assert again["status"] == "updated"
    entries = api.stores[cfg.store_id]["entries"]
    assert [e["content"] for e in entries.values()] == ["tables, no emoji"]
    assert all(e["actor_id"] == "databricks-user:1001" for e in entries.values())
    found = await call(tools["memory_search"], query="style", limit=99)
    assert found["status"] == "ok" and found["results"][0]["content"] == "tables, no emoji"
    listed = await call(tools["memory_list"], path_prefix="/prefs/")
    assert listed["memories"] == [
        {"path": "/prefs/style.md", "description": "style", "updated": None}
    ]
    assert (await call(tools["memory_delete"], path="/prefs/style.md"))["status"] == "deleted"
    assert (await call(tools["memory_delete"], path="/prefs/style.md"))["status"] == "not_found"


@pytest.mark.asyncio
async def test_the_model_cannot_choose_the_actor_or_the_store(monkeypatch):
    tools = await tools_for(
        monkeypatch, FakeMemoryApi(ALICE), mem.MemoryConfig.for_user(KEY, ALICE)
    )
    for tool in tools.values():
        properties = set(tool.to_dict()["input_schema"]["properties"])
        assert not properties & {"actor_id", "store", "store_id", "session_id"}, tool.name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["prefs.md", "/../etc/passwd", "/a/../b", "/a b", "/" + "x" * 300, ""]
)
async def test_bad_paths_are_refused(monkeypatch, path):
    api = FakeMemoryApi(ALICE)
    tools = await tools_for(monkeypatch, api, mem.MemoryConfig.for_user(KEY, ALICE))
    before = len(api.calls)
    result = await call(tools["memory_save"], path=path, content="x", description="d")
    assert result["status"] == "error" and len(api.calls) == before  # refused before any request


@pytest.mark.asyncio
async def test_oversized_content_is_refused(monkeypatch):
    tools = await tools_for(
        monkeypatch, FakeMemoryApi(ALICE), mem.MemoryConfig.for_user(KEY, ALICE)
    )
    result = await call(
        tools["memory_save"],
        path="/a.md",
        content="x" * (mem.MAX_CONTENT_CHARS + 1),
        description="d",
    )
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_api_unavailable_is_a_tool_error_not_a_crash(monkeypatch):
    api = FakeMemoryApi(ALICE)
    api.fail = NotFound("The Agent Memory Sessions preview is not enabled for this workspace")
    tools = await tools_for(monkeypatch, api, mem.MemoryConfig.for_user(KEY, ALICE))
    result = await call(tools["memory_search"], query="x")
    assert result["status"] == "error" and "not enabled" in result["error"]


@pytest.mark.asyncio
async def test_an_api_error_during_a_call_is_reported(monkeypatch):
    api = FakeMemoryApi(ALICE)
    tools = await tools_for(monkeypatch, api, mem.MemoryConfig.for_user(KEY, ALICE))
    api.fail = PermissionDenied("scope missing")
    result = await call(tools["memory_save"], path="/a.md", content="x", description="d")
    assert result["status"] == "error" and "scope missing" in result["error"]


# ---------------------------------------------------------------- App user mode


def test_user_launcher_gives_each_user_their_own_store():
    import service as svc
    import user_mode as um

    env = {
        "ANTHROPIC_ENVIRONMENT_ID": "env_S",
        "ANTHROPIC_ENVIRONMENT_KEY": "sk-env-S",
        "CMA_SANDBOX_ID_KEY": KEY,
    }
    service = svc.build_launcher(env, workspace_client=MagicMock())
    cfg = um.UserConfig(
        environment_id="env_U",
        environment_key="sk-env-U",
        anthropic_api_key="sk-ant",
        agents=("agent_1",),
        max_sessions_per_user=20,
        memory=True,
    )
    alice = um.user_launcher(service, cfg, MagicMock(), ALICE)
    bob = um.user_launcher(service, cfg, MagicMock(), BOB)
    assert alice.memory == mem.MemoryConfig.for_user(KEY, ALICE)
    assert (
        bob.memory is not None
        and alice.memory is not None
        and bob.memory.store_id != alice.memory.store_id
    )
    off = dataclasses.replace(cfg, memory=False)
    assert um.user_launcher(service, off, MagicMock(), ALICE).memory is None
    assert (
        um.user_launcher(service, cfg, MagicMock()).memory is None
    )  # the template: no user, no memory


def test_user_memory_flag_from_environment():
    import user_mode as um

    base = {
        "CMA_USER_ENVIRONMENT_ID": "env_U",
        "ANTHROPIC_USER_ENVIRONMENT_KEY": "k",
        "CMA_ANTHROPIC_API_KEY": "a",
        "CMA_USER_AGENTS": "agent_1",
    }
    assert um.UserConfig.from_environment(base, service_environment_id="env_S").memory is False
    on = um.UserConfig.from_environment(
        {**base, "CMA_USER_MEMORY": "1"}, service_environment_id="env_S"
    )
    assert on.memory is True


def test_the_routing_launcher_launches_with_the_bound_users_id():
    import signin
    import user_store as us

    store, tickets = us.MemoryStore(), signin.Tickets()
    store.bind(us.Binding("sesn_a", ALICE, "alice@example.com", "agent_1"))
    tickets.put("sesn_a", ALICE, MagicMock(), ttl_s=600)
    seen: list[str] = []

    def launcher_for(client, principal_id):
        seen.append(principal_id)
        return SimpleNamespace(launch=lambda work: SimpleNamespace(sandbox_name="sandboxes/x"))

    rl = signin.RoutingLauncher(
        template=MagicMock(),
        launcher_for=launcher_for,
        tickets=tickets,
        store=store,
        sessions_api=MagicMock(),
        agents=("agent_1",),
        app_url="https://app.example.com",
    )
    rl.launch(
        SimpleNamespace(id="sesn_a", secret=None, data=SimpleNamespace(id="sesn_a", type="session"))
    )
    assert seen == [ALICE]


def _user_mode(memory: bool, do=None):
    import service as svc
    import user_mode as um

    client = MagicMock()
    if do is not None:
        client.api_client.do.side_effect = do
    env = {
        "ANTHROPIC_ENVIRONMENT_ID": "env_S",
        "ANTHROPIC_ENVIRONMENT_KEY": "sk-env-S",
        "CMA_SANDBOX_ID_KEY": KEY,
    }
    service = svc.build_launcher(env, workspace_client=client)
    cfg = um.UserConfig(
        environment_id="env_U",
        environment_key="sk-env-U",
        anthropic_api_key="sk-ant",
        agents=("agent_1",),
        max_sessions_per_user=20,
        memory=memory,
    )
    return um.UserMode(
        config=cfg,
        identity=MagicMock(),
        store=MagicMock(),
        anthropic=MagicMock(),
        service_launcher=service,
    ), client


def test_memory_off_by_default_makes_no_call():
    user, client = _user_mode(memory=False)
    assert user.check_memory() == "off"
    client.api_client.do.assert_not_called()
    assert user.launcher_for(MagicMock(), ALICE).memory is None


def test_memory_on_and_usable():
    user, client = _user_mode(memory=True, do=lambda *a, **k: {"managed_memory_stores": []})
    assert user.check_memory() == "on"
    assert client.api_client.do.call_args.args[:2] == ("GET", mem.MEMORY_API)
    assert user.launcher_for(MagicMock(), ALICE).memory == mem.MemoryConfig.for_user(KEY, ALICE)


def test_memory_on_but_the_beta_is_off_serves_without_memory(caplog):
    def disabled(*args, **kwargs):
        raise NotFound(
            "The Agent Memory Sessions preview is not enabled for this workspace",
            error_code="FEATURE_DISABLED",
        )

    user, _ = _user_mode(memory=True, do=disabled)
    assert user.check_memory() == "unavailable (FEATURE_DISABLED)"
    assert user.launcher_for(MagicMock(), ALICE).memory is None
    assert "Agent Memory Sessions preview" in caplog.text


def test_status_reports_user_memory():
    from fastapi.testclient import TestClient

    import app as app_module

    user, _ = _user_mode(memory=False)
    service = SimpleNamespace(
        full_status=lambda: {"healthy": True, "dispatcher": {}},
        start=lambda: None,
        launcher=user.service_launcher,
    )

    async def stop():
        return None

    service.stop = stop
    user.store.ensure_schema = lambda: None
    user.start = lambda: None
    user.stop = stop

    import os

    os.environ["CMA_USER_MODE"] = "1"
    try:
        test_app = app_module.create_app(
            make_service=lambda: service, make_user_mode=lambda s: user
        )
    finally:
        del os.environ["CMA_USER_MODE"]
    with TestClient(test_app) as client:
        assert client.get("/api/status").json()["user_memory"] == "off"
