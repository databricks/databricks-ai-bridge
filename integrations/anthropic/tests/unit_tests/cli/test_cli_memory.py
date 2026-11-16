"""``run --memory`` and ``memory show|delete``: this profile's own Databricks memory store."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from cli_fakes import FakeDispatcher, FakeStats, FakeWorkspace
from databricks.sdk.errors import NotFound

from databricks_anthropic.cli import auth, config, doctor, main
from databricks_anthropic.cli.config import Secrets, Settings
from databricks_anthropic.sandbox.memory import MEMORY_API, MemoryConfig

ENV = "env_mem01"
KEY = "h" * 64
USER_ID = "4242"


class FakeApi:
    def __init__(
        self, stores: dict[str, dict] | None = None, fail: Exception | None = None
    ) -> None:
        self.stores = stores or {}
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def do(
        self, method: str, path: str, query: dict | None = None, body: dict | None = None
    ) -> Any:
        self.calls.append((method, path))
        if self.fail is not None:
            raise self.fail
        if path == MEMORY_API:
            return {"managed_memory_stores": []}
        store_id = path.removeprefix(MEMORY_API + "/").split("/")[0]
        if store_id not in self.stores:
            raise NotFound("no store")
        if method == "DELETE":
            del self.stores[store_id]
            return {}
        if path.endswith("/entries"):
            return {"managed_memory_entries": self.stores[store_id].get("entries", [])}
        return self.stores[store_id]


@pytest.fixture
def mem_env(tmp_path, monkeypatch):
    directory = tmp_path / "cfg"
    config.ensure_config_dir(directory)
    config.save_settings(directory / "default.toml", Settings(profile="dev", environment_id=ENV))
    config.save_secrets(
        directory / "default.secrets",
        Secrets(environment_key="sk-ant-oat01-MEM", sandbox_id_key=KEY),
    )
    workspace = FakeWorkspace()
    workspace.current_user = SimpleNamespace(
        me=lambda: SimpleNamespace(user_name="dev@example.com", id=USER_ID)
    )
    workspace.api_client = FakeApi()
    state: dict[str, Any] = {"dir": str(directory), "workspace": workspace, "launcher_kwargs": None}
    monkeypatch.setattr(auth, "login", lambda **kwargs: workspace)
    monkeypatch.setattr(doctor, "environment_api", lambda key, base: FakeStats(0).client())
    monkeypatch.setattr(main, "_pause", lambda s: None)
    monkeypatch.setattr(main, "DatabricksSandboxDispatcher", FakeDispatcher)

    real = main._launcher

    def launcher(ctx, w, args, **kwargs):
        state["launcher_kwargs"] = kwargs
        return real(ctx, w, args, **kwargs)

    monkeypatch.setattr(main, "_launcher", launcher)
    return state


def expected() -> MemoryConfig:
    return MemoryConfig.for_user(KEY, USER_ID)


def test_run_memory_passes_the_profiles_own_store(mem_env, capsys):
    assert main.main(["run", "--config-dir", mem_env["dir"], "--memory"]) == 0
    assert mem_env["launcher_kwargs"]["memory"] == expected()
    assert (
        f"memory: this identity's Databricks memory store {expected().store_id}"
        in capsys.readouterr().out
    )


def test_run_without_memory_passes_none(mem_env):
    assert main.main(["run", "--config-dir", mem_env["dir"]]) == 0
    assert mem_env["launcher_kwargs"]["memory"] is None


def test_run_memory_stops_with_the_fix_when_the_beta_is_off(mem_env):
    mem_env["workspace"].api_client = FakeApi(
        fail=NotFound("The Agent Memory Sessions preview is not enabled for this workspace")
    )
    with pytest.raises(SystemExit, match="Agent Memory Sessions"):
        main.main(["run", "--config-dir", mem_env["dir"], "--memory"])


def test_memory_show_before_the_store_exists(mem_env, capsys):
    assert main.main(["memory", "show", "--config-dir", mem_env["dir"]]) == 0
    assert "not created yet" in capsys.readouterr().out


def test_memory_show_lists_entries(mem_env, capsys):
    store = expected().store_id
    mem_env["workspace"].api_client = FakeApi(
        {
            store: {
                "owner_user_id": USER_ID,
                "entries": [{"path": "/prefs/a.md", "description": "style"}],
            }
        }
    )
    assert main.main(["memory", "show", "--config-dir", mem_env["dir"]]) == 0
    out = capsys.readouterr().out
    assert f"memory store {store}: 1 memories" in out and "/prefs/a.md" in out


def test_memory_delete_needs_yes_and_refuses_someone_elses_store(mem_env):
    store = expected().store_id
    api = FakeApi({store: {"owner_user_id": USER_ID}})
    mem_env["workspace"].api_client = api
    with pytest.raises(SystemExit, match="add --yes"):
        main.main(["memory", "delete", "--config-dir", mem_env["dir"]])
    assert store in api.stores
    api.stores[store]["owner_user_id"] = "someone-else"
    with pytest.raises(SystemExit, match="owned by someone else"):
        main.main(["memory", "delete", "--config-dir", mem_env["dir"], "--yes"])
    assert store in api.stores
    api.stores[store]["owner_user_id"] = USER_ID
    assert main.main(["memory", "delete", "--config-dir", mem_env["dir"], "--yes"]) == 0
    assert store not in api.stores
