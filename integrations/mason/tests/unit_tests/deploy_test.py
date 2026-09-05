"""Unit tests for the deploy wrapper: trace env injection, store validation, deploy argv."""

from __future__ import annotations

import json
import pathlib
import types
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from databricks_mason import deploy as deploy_mod
from databricks_mason.errors import AgentCliError


@pytest.fixture(autouse=True)
def _compute_active(monkeypatch):
    # `mason deploy` now waits for compute on every deploy; report ACTIVE so the wait returns
    # immediately. Tests that exercise _wait_for_running directly override _app_compute_state.
    monkeypatch.setattr(deploy_mod, "_app_compute_state", lambda name, profile: "ACTIVE")


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
    # Reused store -> created is False.
    assert deploy_mod._ensure_session_store(client, "s") == ({"session_store_name": "s"}, False)
    client.create_session_store.assert_called_once_with("s", retry_transient=True)


def test_ensure_session_store_reports_created():
    client = mock.Mock()
    client.create_session_store.return_value = {"session_store_name": "s"}
    assert deploy_mod._ensure_session_store(client, "s") == ({"session_store_name": "s"}, True)


def test_ensure_memory_store_reuses_on_already_exists():
    client = mock.Mock()
    client.create_memory_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.list_memory_stores.return_value = {
        "managed_memory_stores": [{"name": "memory-stores/mem-id-123", "display_name": "mem"}]
    }

    # Reused store -> created is False.
    assert deploy_mod._ensure_memory_store(client, "mem") == (
        {"name": "memory-stores/mem-id-123", "display_name": "mem"},
        False,
    )
    client.create_memory_store.assert_called_once_with("mem", retry_transient=True)


def test_ensure_memory_store_reports_created():
    client = mock.Mock()
    client.create_memory_store.return_value = {"name": "memory-stores/mem-id-123"}
    assert deploy_mod._ensure_memory_store(client, "mem") == (
        {"name": "memory-stores/mem-id-123"},
        True,
    )


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

    def create_memory_store(self, display_name, *, retry_transient=False):
        return {"name": "memory-stores/mem-id-123", "display_name": display_name}

    def get_session_store(self, name):
        return {"session_store_name": name}

    def create_session_store(self, name, *, retry_transient=False):
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
    _agent_toml(src, memory="mem")

    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: (
            calls.append(args) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src)],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    # Mason prefixes the app name with `mason-` so `deployments list` can find its own apps.
    ws = "/Workspace/Users/me@example.com/mason_deployments/mason-myapp"
    # uv.lock is excluded so the build resolves fresh against its own index (not the dev machine's).
    assert ["sync", str(src), ws, "--exclude", "uv.lock"] in calls
    assert ["apps", "deploy", "mason-myapp", "--source-code-path", ws] in calls
    # Stores are read from agent.toml at runtime, so deploy does NOT write store env into app.yaml.
    env_entries = yaml.safe_load((src / "app.yaml").read_text()).get("env") or []
    env = {e["name"]: e["value"] for e in env_entries}
    assert "AGENT_MEMORY_STORE" not in env


def test_deploy_creates_with_instance_count(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: False)
    monkeypatch.setattr(deploy_mod, "_wait_for_running", lambda name, profile: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: calls.append((args, kwargs))
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--instances", "2"],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    assert (
        [
            "apps",
            "create",
            "mason-myapp",
            "--compute-min-instances",
            "2",
            "--compute-max-instances",
            "2",
        ],
        {
            "capture": True,
            "action": "Could not create deployment 'mason-myapp'.",
        },
    ) in calls


def test_deploy_updates_existing_instance_count(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: calls.append((args, kwargs))
        or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--instances", "2"],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    update_args, update_kwargs = next(
        call for call in calls if call[0][:3] == ["apps", "create-update", "mason-myapp"]
    )
    assert update_kwargs == {
        "capture": True,
        "action": "Could not update deployment 'mason-myapp'.",
    }
    payload = json.loads(update_args[update_args.index("--json") + 1])
    assert payload == {
        "app": {"compute_min_instances": 2, "compute_max_instances": 2},
        "update_mask": "compute_min_instances,compute_max_instances",
    }


def test_deploy_rejects_instance_count_above_platform_limit():
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--instances", "6"],
        obj=_FakeCtx(),
    )

    assert result.exit_code != 0
    assert "6 is not in the range 1<=x<=5" in result.output


def test_deploy_help_exposes_instances_and_sticky_routing():
    result = CliRunner().invoke(deploy_mod.deploy, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--instances" in result.output
    assert "--min-instances" not in result.output
    assert "--max-instances" not in result.output
    assert "sticky routing" in result.output
    assert "__Host-databricks-app-router" in result.output
    assert "Databricks Apps instances" not in result.output


def test_deploy_renames_underlying_app_compute_output(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: False)
    monkeypatch.setattr(deploy_mod, "_wait_for_running", lambda name, profile: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: (
            calls.append((args, kwargs))
            or types.SimpleNamespace(
                returncode=0,
                stdout="App compute is starting\n" if args[:2] == ["apps", "create"] else "",
                stderr="",
            )
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    apps_calls = [call for call in calls if call[0][1] in ("create", "deploy")]
    assert [call[0][1] for call in apps_calls] == ["create", "deploy"]
    # create is still captured (to relabel its output); both carry an `action` so a failure is
    # reported in Mason's terms instead of echoing the raw `databricks apps` command.
    assert apps_calls[0][1] == {
        "capture": True,
        "action": "Could not create deployment 'mason-myapp'.",
    }
    assert apps_calls[1][1] == {"action": "Could not deploy 'mason-myapp'."}
    assert "Agent compute is starting" in result.output
    assert "App compute" not in result.output
    get_call = next(call for call in calls if call[0][:2] == ["apps", "get"])
    assert get_call[1] == {"capture": True, "check": False}


def test_deploy_reports_app_url(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        deploy_mod, "_app_url", lambda name, p: "https://myapp-123.databricksapps.com"
    )
    captured: dict = {}
    monkeypatch.setattr(deploy_mod.render, "emit_json", lambda data: captured.update(data))

    class _JsonCtx(_FakeCtx):
        output = "json"

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_JsonCtx())

    assert result.exit_code == 0, result.output
    assert captured["url"] == "https://myapp-123.databricksapps.com"


def test_deploy_sync_keeps_directly_edited_agent_manifest(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    (src / "agent.toml").write_text('schema_version = 1\n\n[agent]\nframework = "openai"\n')
    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: (
            calls.append(args) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    sync = next(args for args in calls if args[0] == "sync")
    assert sync[:3] == [
        "sync",
        str(src),
        "/Workspace/Users/me@example.com/mason_deployments/mason-myapp",
    ]
    excluded = {sync[index + 1] for index, value in enumerate(sync[:-1]) if value == "--exclude"}
    assert "agent.toml" not in excluded


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
        lambda args, profile, **kw: (
            calls.append(args) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    assert ["apps", "create", "mason-myapp"] in calls
    assert waited["called"], "must wait for the new app to be running before deploying"


def test_redeploy_waits_for_running_and_skips_create(tmp_path: pathlib.Path, monkeypatch):
    # An existing app is re-deployed: no `apps create` (it would error), but still wait for compute
    # so there's feedback and the app is ACTIVE before `apps deploy`.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[list[str]] = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)  # already exists
    waited = {"called": False}
    monkeypatch.setattr(
        deploy_mod, "_wait_for_running", lambda name, profile: waited.__setitem__("called", True)
    )
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: (
            calls.append(args) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    assert ["apps", "create", "mason-myapp"] not in calls  # never re-create an existing app
    assert waited["called"], "re-deploy must also wait for compute"


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


def test_deploy_does_not_write_store_env_to_app_yaml(tmp_path: pathlib.Path, monkeypatch):
    # Stores are read from agent.toml at runtime, so deploy writes no AGENT_*_STORE (nor actor) env.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src, memory="mem", session="sessions")

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src)],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    env_entries = yaml.safe_load((src / "app.yaml").read_text()).get("env") or []
    env = {entry["name"]: entry["value"] for entry in env_entries}
    assert "AGENT_MEMORY_STORE" not in env
    assert "AGENT_SESSION_STORE" not in env
    assert "AGENT_MEMORY_ACTOR_ID" not in env
    assert "AGENT_SESSION_ACTOR_ID" not in env


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


def test_deploy_validates_bound_memory_store_by_display_name(tmp_path: pathlib.Path, monkeypatch):
    # Deploy resolves the bound memory store by display name (list+match), not get_memory_store (by id).
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src, memory="mem")
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, p: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    # _FakeClient resolves display name "mem" via list+match; deploy succeeds.
    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())
    assert result.exit_code == 0, result.output


def test_deploy_errors_when_bound_store_missing(tmp_path: pathlib.Path, monkeypatch):
    # Stores are created by `mason ... bind`, not deploy; a bound-but-missing store fails with a hint.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src, memory="ghost")
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)

    class _EmptyClient(_FakeClient):
        def list_memory_stores(self, page_size=None, page_token=None):
            return {"managed_memory_stores": [], "next_page_token": ""}

    class _Ctx(_FakeCtx):
        def client(self):
            return _EmptyClient()

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_Ctx())
    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert "mason memory bind ghost" in result.output


def test_with_traces_defaults_the_experiment_per_app():
    # --with-traces alone must still set the experiment, or the agent ships tracing half-configured
    # (destination set, experiment missing) and silently disables it. The default is per-app, so
    # each agent's traces are isolated instead of piling into one shared experiment.
    env = deploy_mod.validate_stores_and_trace_env(
        _FakeClient(),
        app="my-agent",
        memory_store=None,
        session_store=None,
        traces_destination="cat.schema",
        traces_experiment=None,
    )
    assert env["MLFLOW_TRACING_DESTINATION"] == "cat.schema"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "/Users/me@example.com/mason-traces/my-agent"


def test_with_traces_explicit_experiment_wins_over_per_app():
    env = deploy_mod.validate_stores_and_trace_env(
        _FakeClient(),
        app="my-agent",
        memory_store=None,
        session_store=None,
        traces_destination="cat.schema",
        traces_experiment="/Shared/custom",
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


def _agent_toml(source: pathlib.Path, *, memory=None, session=None) -> None:
    text = 'schema_version = 1\n\n[agent]\nframework = "openai"\n'
    if memory:
        text += f'\n[memory_store]\nname = "{memory}"\n'
    if session:
        text += f'\n[session_store]\nname = "{session}"\n'
    (source / "agent.toml").write_text(text, encoding="utf-8")


def test_store_bindings_reads_agent_toml(tmp_path: pathlib.Path):
    _agent_toml(tmp_path, memory="bound-mem", session="bound-sess")
    assert deploy_mod.store_bindings(tmp_path) == ("bound-mem", "bound-sess")


def test_store_bindings_none_when_unbound(tmp_path: pathlib.Path):
    _agent_toml(tmp_path)  # scaffold with no store tables
    assert deploy_mod.store_bindings(tmp_path) == (None, None)


def test_store_bindings_ignores_missing_manifest(tmp_path: pathlib.Path):
    # No agent.toml -> no stores, never raises (so deploy/dev aren't blocked).
    assert deploy_mod.store_bindings(tmp_path) == (None, None)


def test_deploy_grants_bound_store(tmp_path: pathlib.Path, monkeypatch):
    # `mason sessions bind` then plain `mason deploy`: the binding must drive both the
    # app.yaml env AND the SP access grant, or the deployed app can't reach its durable store.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src, session="bound-sess")

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda name, profile: "sp-123")
    grant_args: dict = {}
    monkeypatch.setattr(
        deploy_mod,
        "_grant_store_access",
        lambda name, sp, user, session_store, memory_database, profile: (
            grant_args.update(sp=sp, session_store=session_store, memory_database=memory_database)
            or None
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    # The grant fired for the bound session store, resolved from agent.toml — no store env in app.yaml.
    assert grant_args == {"sp": "sp-123", "session_store": "bound-sess", "memory_database": None}
    env_entries = yaml.safe_load((src / "app.yaml").read_text()).get("env") or []
    assert "AGENT_SESSION_STORE" not in {e["name"] for e in env_entries}
