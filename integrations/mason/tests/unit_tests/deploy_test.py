"""Unit tests for the deploy wrapper: app.yaml env injection, store reuse, deploy argv."""

from __future__ import annotations

import json
import pathlib
import types
from unittest import mock

import yaml
from click.testing import CliRunner

from databricks_mason import deploy as deploy_mod
from databricks_mason.errors import AgentCliError


def test_upsert_manifest_env_scaffolds_when_missing(tmp_path: pathlib.Path):
    scaffolded = deploy_mod._upsert_manifest_env(
        tmp_path, {"AGENT_MEMORY_STORE": "memory-stores/x"}
    )
    assert scaffolded is True
    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert {"name": "AGENT_MEMORY_STORE", "value": "memory-stores/x"} in doc["env"]
    assert "command" in doc  # placeholder written


def test_upsert_manifest_env_updates_existing(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["uvicorn", "app:app"],
                "env": [{"name": "AGENT_MEMORY_STORE", "value": "old"}],
            }
        )
    )
    scaffolded = deploy_mod._upsert_manifest_env(
        tmp_path, {"AGENT_MEMORY_STORE": "new", "AGENT_SESSION_STORE": "s"}
    )
    assert scaffolded is False
    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert doc["command"] == ["uvicorn", "app:app"]  # preserved
    by_name = {e["name"]: e["value"] for e in doc["env"]}
    assert by_name == {"AGENT_MEMORY_STORE": "new", "AGENT_SESSION_STORE": "s"}


def test_ensure_session_store_reuses_on_already_exists():
    client = mock.Mock()
    client.create_session_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.get_session_store.return_value = {"session_store_name": "s"}
    assert deploy_mod._ensure_session_store(client, "s") == {"session_store_name": "s"}


class _FakeClient:
    host = "https://ws"
    current_user = "me@example.com"

    def get_memory_store(self, name):
        return {"name": f"memory-stores/{name}"}

    def list_memory_stores(self, page_size=None, page_token=None):
        # One page; the store's resource name is an id distinct from its display name (as the real
        # API returns), so resolution must match on display_name, not id.
        return {
            "managed_memory_stores": [{"name": "memory-stores/mem-id-123", "display_name": "mem"}],
            "next_page_token": "",
        }

    def create_memory_store(self, display_name):
        # Create-if-not-exists is the deploy default; return the same id resolution would find.
        return {"name": "memory-stores/mem-id-123", "display_name": display_name}

    def get_session_store(self, name):
        return {"session_store_name": name}

    def create_session_store(self, name):
        return {"session_store_name": name}


class _FakeCtx:
    profile = "prof"
    output = "text"

    def client(self):
        return _FakeClient()


def test_deploy_drives_sync_and_apps_deploy(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--memory", "mem"],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    ws = "/Workspace/Users/me@example.com/mason_deployments/myapp"
    # uv.lock is excluded so the build resolves fresh against its own index (not the dev machine's).
    assert ["sync", str(src), ws, "--exclude", "uv.lock"] in calls
    assert ["apps", "deploy", "myapp", "--source-code-path", ws] in calls
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    # Display name "mem" resolves to store id memory-stores/mem-id-123; the runtime re-adds the
    # `memory-stores/` prefix, so the env var must carry the bare id.
    assert env["AGENT_MEMORY_STORE"] == "mem-id-123"


def test_deploy_sync_keeps_code_selected_tool_registry(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    registry = src / "agent" / "databricks_tools.py"
    registry.parent.mkdir()
    registry.write_text("DATABRICKS_TOOLS = ()\n")
    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    sync = next(args for args in calls if args[0] == "sync")
    assert sync[:3] == [
        "sync",
        str(src),
        "/Workspace/Users/me@example.com/mason_deployments/myapp",
    ]
    excluded = {sync[index + 1] for index, value in enumerate(sync[:-1]) if value == "--exclude"}
    assert "agent/databricks_tools.py" not in excluded
    assert registry.read_text() == "DATABRICKS_TOOLS = ()\n"


def test_first_deploy_waits_for_running_before_deploying(tmp_path: pathlib.Path, monkeypatch):
    # A brand-new app isn't RUNNING right after `apps create`; deploy must wait, or it races and
    # fails ("not in RUNNING state"). Verify create -> wait -> sync/deploy ordering.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[list[str]] = []
    monkeypatch.setattr(
        deploy_mod, "_deployment_exists", lambda a, p: False
    )  # app doesn't exist yet
    waited = {"called": False}
    monkeypatch.setattr(
        deploy_mod, "_wait_for_running", lambda name, profile: waited.__setitem__("called", True)
    )
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: calls.append(args)
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    assert ["apps", "create", "myapp"] in calls
    assert waited["called"], "must wait for the new app to be running before deploying"
    # the wait happens after create and before sync/deploy
    create_i = calls.index(["apps", "create", "myapp"])
    sync_i = next(i for i, a in enumerate(calls) if a[:1] == ["sync"])
    assert create_i < sync_i


def test_wait_for_running_returns_when_compute_active(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_app_compute_state", lambda name, p: "ACTIVE")
    deploy_mod._wait_for_running("app", "prof", timeout_s=1)  # returns without raising


def test_wait_for_running_times_out(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_app_compute_state", lambda name, p: "STARTING")
    monkeypatch.setattr(deploy_mod.time, "sleep", lambda s: None)  # don't actually wait
    try:
        deploy_mod._wait_for_running("app", "prof", timeout_s=0)
        raise AssertionError("expected AgentCliError on timeout")
    except AgentCliError:
        pass


def test_deploy_injects_shared_actor_for_managed_stores(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        [
            "myapp",
            "--source",
            str(src),
            "--memory",
            "mem",
            "--session",
            "sessions",
            "--actor-id",
            "alice",
        ],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    env = {
        entry["name"]: entry["value"]
        for entry in yaml.safe_load((src / "app.yaml").read_text())["env"]
    }
    assert env["AGENT_MEMORY_ACTOR_ID"] == "alice"
    assert env["AGENT_SESSION_ACTOR_ID"] == "alice"


def test_deploy_with_traces_injects_tracing_env(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        [
            "myapp",
            "--source",
            str(src),
            "--with-traces",
            "cat.schema",
            "--traces-experiment",
            "/Shared/x",
        ],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    doc = yaml.safe_load((src / "app.yaml").read_text())
    env = {e["name"]: e["value"] for e in doc["env"]}
    assert env["MLFLOW_TRACING_DESTINATION"] == "cat.schema"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "/Shared/x"


def test_resolve_memory_store_pages_at_100_and_matches_display_name():
    # The list API caps page_size at 100, so resolution must page (not request 1000) and match the
    # display name across pages.
    class _PagingClient:
        def __init__(self):
            self.calls = []

        def list_memory_stores(self, page_size=None, page_token=None):
            self.calls.append((page_size, page_token))
            if page_token is None:
                return {
                    "managed_memory_stores": [{"name": "memory-stores/a", "display_name": "other"}],
                    "next_page_token": "p2",
                }
            return {
                "managed_memory_stores": [{"name": "memory-stores/b", "display_name": "wanted"}],
                "next_page_token": "",
            }

    client = _PagingClient()
    store = deploy_mod._resolve_memory_store(client, "wanted")
    assert store is not None
    assert store["name"] == "memory-stores/b"  # found on page 2
    assert all(ps == 100 for ps, _ in client.calls)  # never exceeds the API cap
    assert [pt for _, pt in client.calls] == [None, "p2"]  # followed the page token


def test_resolve_memory_store_returns_none_when_absent():
    class _EmptyClient:
        def list_memory_stores(self, page_size=None, page_token=None):
            return {"managed_memory_stores": [], "next_page_token": ""}

    assert deploy_mod._resolve_memory_store(_EmptyClient(), "nope") is None


def test_memory_store_database_resolves_by_display_name():
    # The grant step derives the Lakebase db from the store; it must resolve by display name
    # (list+match), not get_memory_store (by id), or it 404s on the deploy flag's value.
    class _Client:
        def list_memory_stores(self, page_size=None, page_token=None):
            return {
                "managed_memory_stores": [
                    {
                        "name": "memory-stores/uuid-x",
                        "display_name": "mem",
                        "storage_backend": {
                            "backend_id": "projects/p/branches/production/databases/memory-uuidx"
                        },
                    }
                ],
                "next_page_token": "",
            }

    assert deploy_mod._memory_store_database(_Client(), "mem") == "memory-uuidx"


def test_deploy_no_create_stores_resolves_memory_by_display_name(
    tmp_path: pathlib.Path, monkeypatch
):
    # --no-create-stores path must resolve by display name (list+match), not get_memory_store (by id).
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--memory", "mem", "--no-create-stores"],
        obj=_FakeCtx(),
    )
    assert result.exit_code == 0, result.output
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    assert env["AGENT_MEMORY_STORE"] == "mem-id-123"  # resolved by display name, bare id


def test_deploy_creates_missing_stores_by_default(tmp_path: pathlib.Path, monkeypatch):
    # Create-if-not-exists is now the default (no flag): a referenced store is created via the API.
    created: list[str] = []
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    class _CreatingClient(_FakeClient):
        def create_memory_store(self, display_name):
            created.append(display_name)
            return {"name": "memory-stores/mem-id-123", "display_name": display_name}

    class _Ctx(_FakeCtx):
        def client(self):
            return _CreatingClient()

    result = CliRunner().invoke(
        deploy_mod.deploy, ["myapp", "--source", str(src), "--memory", "new-mem"], obj=_Ctx()
    )
    assert result.exit_code == 0, result.output
    assert created == ["new-mem"]  # created without any --create-stores flag


def test_deploy_accepts_short_store_flags(tmp_path: pathlib.Path, monkeypatch):
    # -m / -s are the short forms of --memory / --session.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "-m", "mem", "-s", "s"],
        obj=_FakeCtx(),
    )
    assert result.exit_code == 0, result.output
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    assert env["AGENT_MEMORY_STORE"] == "mem-id-123"
    assert env["AGENT_SESSION_STORE"] == "s"


def test_with_traces_defaults_the_experiment_per_app():
    # --with-traces alone must still set the experiment, or the agent ships tracing half-configured
    # (destination set, experiment missing) and silently disables it. The default is per-app, so
    # each agent's traces are isolated instead of piling into one shared experiment.
    env = deploy_mod.resolve_store_env(
        _FakeClient(),
        app="my-agent",
        memory_store=None,
        session_store=None,
        traces_destination="cat.schema",
        traces_experiment=None,
        create_stores=False,
    )
    assert env["MLFLOW_TRACING_DESTINATION"] == "cat.schema"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "/Users/me@example.com/mason-traces/my-agent"


def test_with_traces_explicit_experiment_wins_over_per_app():
    env = deploy_mod.resolve_store_env(
        _FakeClient(),
        app="my-agent",
        memory_store=None,
        session_store=None,
        traces_destination="cat.schema",
        traces_experiment="/Shared/custom",
        create_stores=False,
    )
    assert env["MLFLOW_EXPERIMENT_NAME"] == "/Shared/custom"


def _run_deploy(src, monkeypatch, extra_args):
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    return CliRunner().invoke(
        deploy_mod.deploy, ["myapp", "--source", str(src), *extra_args], obj=_FakeCtx()
    )


def test_deploy_injects_public_pypi_index_by_default(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    result = _run_deploy(src, monkeypatch, [])
    assert result.exit_code == 0, result.output
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    for name in ("PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX"):
        assert env[name] == "https://pypi.org/simple/"


def test_deploy_empty_pip_index_disables_override(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    result = _run_deploy(src, monkeypatch, ["--pip-index-url", ""])
    assert result.exit_code == 0, result.output
    doc = yaml.safe_load((src / "app.yaml").read_text())
    env = {e["name"]: e["value"] for e in (doc.get("env") or [])}
    assert "PIP_INDEX_URL" not in env  # empty -> no override, use the build's default index


class _JsonCtx:
    profile = "prof"
    output = "json"


def test_lifecycle_commands_honor_json_output(monkeypatch):
    # start/stop/delete must emit JSON (not the Rich success panel) under --output json.
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    for command, key, args in (
        (deploy_mod.deployments_start, "started", ["myapp"]),
        (deploy_mod.deployments_stop, "stopped", ["myapp", "--yes"]),  # destructive: needs --yes
        (deploy_mod.deployments_delete, "deleted", ["myapp", "--yes"]),
    ):
        result = CliRunner().invoke(command, args, obj=_JsonCtx())
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {key: "myapp"}
