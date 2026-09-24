"""Unit tests for the deploy wrapper: trace env injection, store validation, deploy argv."""

from __future__ import annotations

import json
import pathlib
import types
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject, ToolSpec
from databricks_mason.apps_client import AppsClient
from databricks_mason.cli import deploy as deploy_mod
from databricks_mason.cli.tracing import MLflowTraceTables, ResolvedTraceExperiment
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import write_project_metadata
from databricks_mason.store_provisioner import StoreProvisioner

# The autouse fixture below stubs `get_or_create_trace_experiment` for deploy-command tests; capture the
# real function here so its own unit tests can exercise the actual logic.
_REAL_RESOLVE_TRACE = deploy_mod.get_or_create_trace_experiment


@pytest.fixture(autouse=True)
def _compute_active(monkeypatch):
    # `mason deploy` now waits for compute on every deploy; report ACTIVE so the wait returns
    # immediately. Tests that exercise wait_for_running directly override AppsClient.compute_state.
    monkeypatch.setattr(AppsClient, "compute_state", lambda self, name: "ACTIVE")


@pytest.fixture(autouse=True)
def _no_tracing_by_default(monkeypatch):
    # Tracing is on by default and would create an MLflow experiment (a live workspace op); stub the
    # provisioning off so non-tracing deploy tests stay hermetic. Tracing tests override this.
    # (The trace-resource reconcile is stubbed dir-wide by conftest.py's autouse fixture.)
    monkeypatch.setattr(deploy_mod, "get_or_create_trace_experiment", lambda *a, **k: None)


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


def test_upsert_manifest_env_preserves_unrelated_entries_and_replaces_value_from(
    tmp_path: pathlib.Path,
):
    unrelated = {"name": "USER_SECRET", "valueFrom": "user-secret-resource"}
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["uvicorn", "app:app"],
                "env": [
                    unrelated,
                    {"name": "AGENT_MEMORY_STORE", "valueFrom": "old-resource"},
                    "invalid-entry",
                ],
            }
        )
    )

    assert deploy_mod._upsert_manifest_env(tmp_path, {"AGENT_MEMORY_STORE": "new"}) is False

    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert doc["env"] == [unrelated, {"name": "AGENT_MEMORY_STORE", "value": "new"}]


def test_managed_runtime_store_is_an_internal_disabled_rollout_switch():
    assert deploy_mod._USE_MANAGED_RUNTIME_STORE is False


class _FakeClient:
    host = "https://ws"
    current_user = "me@example.com"

    def __init__(self):
        # Seeded with one pre-existing store ("mem", whose id differs from its display name as the
        # real API returns); created stores are appended so deploy's auto-create can then resolve them.
        self._memory_stores = [{"name": "memory-stores/mem-id-123", "display_name": "mem"}]

    def get_memory_store(self, name):
        return {"name": f"memory-stores/{name}"}

    def list_memory_stores(self, page_size=None, page_token=None):
        return {"managed_memory_stores": list(self._memory_stores), "next_page_token": ""}

    def create_memory_store(self, display_name, *, retry_transient=False):
        for existing in self._memory_stores:
            if existing.get("display_name") == display_name:
                raise AgentCliError(
                    f"Memory store '{display_name}' already exists", error_code="ALREADY_EXISTS"
                )
        store = {"name": f"memory-stores/{display_name}", "display_name": display_name}
        self._memory_stores.append(store)
        return store

    def get_session_store(self, name):
        return {"session_store_name": name}

    def create_session_store(self, name, *, retry_transient=False):
        return {"session_store_name": name}

    def create_runtime_store(self, store_id, sp, *, app_name, retry_transient=False):
        return {
            "name": f"runtime-stores/{store_id}",
            "owner": {"app": {"name": app_name, "service_principal_id": sp}},
            "storage_backend": {
                "lakebase": {
                    "project_id": "databricks-internal-custom-agents",
                    "branch": "projects/databricks-internal-custom-agents/branches/production",
                    "database_id": "runtime-agent-mason-myapp-550e8400-e29b-41d4-a716-446655440000",
                }
            },
        }

    def get_runtime_store(self, name):
        raise AgentCliError("absent", error_code="NOT_FOUND")

    def grant_session_store_permission(self, store, principal):
        return None

    def grant_memory_store_permission(self, store, principal):
        return None


class _FakeCtx:
    profile = "prof"
    output = "text"

    def client(self):
        return _FakeClient()


def _write_agent_manifest(
    source: pathlib.Path,
    *,
    framework: str = "langgraph",
    server: str = "mason",
    memory: str | None = None,
    session: str | None = None,
) -> None:
    body = f'schema_version = 1\n\n[agent]\nframework = "{framework}"\nserver = "{server}"\n'
    if memory:
        body += f'\n[memory_store]\nname = "{memory}"\n'
    if session:
        body += f'\n[session_store]\nname = "{session}"\n'
    (source / "agent.toml").write_text(body)


@pytest.mark.parametrize(
    ("framework", "template"),
    [
        ("langgraph", "custom-agent-langgraph"),
        ("openai", "custom-agent-openai"),
    ],
)
def test_deploy_rejects_custom_server_manifest_tools_before_mutation_or_network(
    tmp_path: pathlib.Path,
    framework: str,
    template: str,
):
    source = tmp_path / template
    source.mkdir()
    (source / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    project = AgentProject.create(source, framework=framework, server="custom")
    project.add_tool(ToolSpec.mcp("web", service="system.ai.web_search"))
    project.write()
    write_project_metadata(source, framework=framework, template=template)
    manifest = source / "agent.toml"
    before = manifest.read_text(encoding="utf-8")
    ctx = _FakeCtx()

    with (
        mock.patch.object(deploy_mod, "_databricks") as db,
        mock.patch.object(ctx, "client") as client,
    ):
        result = CliRunner().invoke(
            deploy_mod.deploy,
            ["custom", "--source", str(source)],
            obj=ctx,
        )

    assert result.exit_code != 0
    assert "require a Mason server template" in " ".join(result.output.split())
    assert manifest.read_text(encoding="utf-8") == before
    client.assert_not_called()
    db.assert_not_called()


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_deploy_surfaces_invalid_custom_server_manifest_before_mutation_or_network(
    tmp_path: pathlib.Path,
    framework: str,
):
    source = tmp_path / f"custom-agent-{framework}"
    source.mkdir()
    (source / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    manifest = source / "agent.toml"
    manifest.write_text(
        f'schema_version = 1\n\n[agent]\nframework = "{framework}"\nserver = "custom"\n'
        '\n[[tools]]\nid = "legacy"\nsource = { kind = "python", '
        'entrypoint = "agent.tools:legacy" }\n',
        encoding="utf-8",
    )
    before = manifest.read_text(encoding="utf-8")
    ctx = _FakeCtx()

    with (
        mock.patch.object(deploy_mod, "_databricks") as db,
        mock.patch.object(ctx, "client") as client,
    ):
        result = CliRunner().invoke(
            deploy_mod.deploy,
            ["custom", "--source", str(source)],
            obj=ctx,
        )

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "Python tools are code-first" in output
    assert "framework-native agent code" in output
    assert "remain active" not in output
    assert manifest.read_text(encoding="utf-8") == before
    client.assert_not_called()
    db.assert_not_called()


def test_deploy_drives_sync_and_apps_deploy(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src, memory="mem")

    calls: list[list[str]] = []
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
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
    # Mason prefixes the app name with `agent-mason-` so `deployments list` can find its own apps.
    ws = "/Workspace/Users/me@example.com/mason_deployments/agent-mason-myapp"
    # uv.lock is excluded so the build resolves fresh against its own index (not the dev machine's).
    assert ["sync", str(src), ws, "--exclude", "uv.lock"] in calls
    assert ["apps", "deploy", "agent-mason-myapp", "--source-code-path", ws] in calls
    # deploy injects the resolved memory-store id so the entries API (keyed by id) can be addressed.
    env_entries = yaml.safe_load((src / "app.yaml").read_text()).get("env") or []
    env = {e["name"]: e["value"] for e in env_entries}
    assert env.get("AGENT_MEMORY_STORE") == "mem-id-123"


def test_deploy_creates_with_instance_count(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: False)
    monkeypatch.setattr(AppsClient, "wait_for_running", lambda self, name, timeout_s=300: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: (
            calls.append((args, kwargs))
            or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
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
            "agent-mason-myapp",
            "--compute-min-instances",
            "2",
            "--compute-max-instances",
            "2",
        ],
        {
            "capture": True,
            "action": "Could not create deployment 'agent-mason-myapp'.",
        },
    ) in calls


def test_deploy_updates_existing_instance_count(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kwargs: (
            calls.append((args, kwargs))
            or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--instances", "2"],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    update_args, update_kwargs = next(
        call for call in calls if call[0][:3] == ["apps", "create-update", "agent-mason-myapp"]
    )
    assert update_kwargs == {
        "capture": True,
        "action": "Could not update deployment 'agent-mason-myapp'.",
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


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_deploy_custom_server_skips_runtime_store_provisioning_and_binding(
    tmp_path: pathlib.Path, monkeypatch, framework: str
) -> None:
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["x"],
                "env": [{"name": "USER_ENV", "value": "keep"}],
            }
        )
    )
    _write_agent_manifest(src, framework=framework, server="custom")

    create = mock.Mock(side_effect=AssertionError("custom server must not provision a store"))
    calls = []

    def fake_databricks(args, profile, **kwargs):
        calls.append(args)
        assert profile == "prof"
        return types.SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: False)
    monkeypatch.setattr(_FakeClient, "create_runtime_store", create)
    # the trace-resource reconcile issues its own create-update and is out of scope here
    monkeypatch.setattr(deploy_mod, "apply_trace_resources", mock.Mock(return_value=None))
    monkeypatch.setattr(deploy_mod, "_databricks", fake_databricks)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src)],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    create.assert_not_called()
    assert any(args[:3] == ["apps", "create", "agent-mason-myapp"] for args in calls)
    assert any(args[:3] == ["apps", "deploy", "agent-mason-myapp"] for args in calls)
    assert not any(args[:2] in (["apps", "update"], ["apps", "create-update"]) for args in calls)
    env = {
        entry["name"]: entry["value"]
        for entry in yaml.safe_load((src / "app.yaml").read_text())["env"]
    }
    assert env["USER_ENV"] == "keep"
    assert "DATABRICKS_MASON_RUNTIME_STORE_LAKEBASE_ENDPOINT" not in env
    assert "DATABRICKS_MASON_RUNTIME_STORE_SCHEMA" not in env


def test_deploy_mason_server_provisions_runtime_store(tmp_path: pathlib.Path, monkeypatch) -> None:
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(src)
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", True)

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(AppsClient, "service_principal", lambda *args: "sp-123")
    deployed_env = None

    def fake_databricks(args, profile, **kwargs):
        nonlocal deployed_env
        if args[:2] == ["apps", "deploy"]:
            manifest = yaml.safe_load((src / "app.yaml").read_text())
            deployed_env = {entry["name"]: entry["value"] for entry in manifest.get("env", [])}
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(deploy_mod, "_databricks", fake_databricks)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src)],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    env = {
        entry["name"]: entry["value"]
        for entry in yaml.safe_load((src / "app.yaml").read_text())["env"]
    }
    assert env[deploy_mod.RUNTIME_STORE_LAKEBASE_BRANCH_ENV] == (
        "projects/databricks-internal-custom-agents/branches/production"
    )
    assert env[deploy_mod.RUNTIME_STORE_DATABASE_ENV].startswith("runtime-agent-mason-myapp-")
    assert env[deploy_mod.RUNTIME_STORE_USERNAME_ENV] == "sp-123"
    assert deployed_env is not None
    assert deployed_env[deploy_mod.RUNTIME_STORE_LAKEBASE_BRANCH_ENV] == (
        "projects/databricks-internal-custom-agents/branches/production"
    )
    assert deploy_mod.RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV not in deployed_env
    assert deploy_mod.RUNTIME_STORE_SCHEMA_ENV not in deployed_env


def test_deploy_defaults_to_legacy_runtime_store(tmp_path: pathlib.Path, monkeypatch) -> None:
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["x"],
                "env": [{"name": "USER_ENV", "value": "keep"}],
            }
        )
    )
    _write_agent_manifest(src)
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)

    backend = deploy_mod.legacy_runtime_store.backend("agent-mason-myapp")
    events = []
    provision = mock.Mock(side_effect=lambda *args: (events.append("legacy-project"), backend)[1])
    attach = mock.Mock(side_effect=lambda *args: events.append("legacy-resource"))
    client = _FakeClient()
    create_managed = mock.Mock(side_effect=AssertionError("managed API must remain opt-in"))
    monkeypatch.setattr(client, "create_runtime_store", create_managed)
    monkeypatch.setattr(deploy_mod.legacy_runtime_store, "get_or_create_backend", provision)
    monkeypatch.setattr(deploy_mod, "apply_postgres_resources", attach)
    monkeypatch.setattr(
        AppsClient,
        "exists",
        lambda *args: (events.append("app-exists"), True)[1],
    )
    monkeypatch.setattr(
        AppsClient,
        "service_principal",
        mock.Mock(side_effect=AssertionError("legacy provisioning does not need an app SP lookup")),
    )
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, *rest, **kwargs: (
            events.append(args[0]) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=ctx)

    assert result.exit_code == 0, result.output
    provision.assert_called_once_with("agent-mason-myapp", "prof")
    attach.assert_called_once_with("agent-mason-myapp", [backend], "prof")
    create_managed.assert_not_called()
    assert events[:4] == ["legacy-project", "app-exists", "legacy-resource", "sync"]
    env = {
        entry["name"]: entry["value"]
        for entry in yaml.safe_load((src / "app.yaml").read_text())["env"]
    }
    assert env[deploy_mod.RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV] == backend.endpoint_path
    assert env[deploy_mod.RUNTIME_STORE_SCHEMA_ENV] == backend.schema
    assert env["USER_ENV"] == "keep"
    assert deploy_mod.RUNTIME_STORE_LAKEBASE_BRANCH_ENV not in env
    assert deploy_mod.RUNTIME_STORE_DATABASE_ENV not in env
    assert deploy_mod.RUNTIME_STORE_USERNAME_ENV not in env


@pytest.mark.parametrize("store_kind", ["session", "memory"])
def test_deploy_runtime_store_uses_dedicated_backend_with_managed_store(
    tmp_path: pathlib.Path, monkeypatch, store_kind: str
) -> None:
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(src, **{store_kind: "other-store"})
    events = []
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", True)
    client = _FakeClient()
    create = client.create_runtime_store

    def create_store(*args, **kwargs):
        events.append("runtime-store")
        return create(*args, **kwargs)

    monkeypatch.setattr(client, "create_runtime_store", create_store)
    monkeypatch.setattr(AppsClient, "exists", lambda *args: False)
    monkeypatch.setattr(AppsClient, "service_principal", lambda *args: "sp-123")
    monkeypatch.setattr(StoreProvisioner, "grant_store_access", lambda self, *a, **k: None)

    def fake_databricks(args, profile, **kwargs):
        assert args[:1] != ["postgres"]
        assert args[:2] != ["apps", "update"]
        if args[:2] == ["apps", "create"]:
            events.append("app-created")
        if args[:2] == ["apps", "deploy"]:
            env = {
                entry["name"]: entry["value"]
                for entry in yaml.safe_load((src / "app.yaml").read_text())["env"]
            }
            assert env[deploy_mod.RUNTIME_STORE_DATABASE_ENV].startswith(
                "runtime-agent-mason-myapp-"
            )
            assert env[deploy_mod.RUNTIME_STORE_DATABASE_ENV] != "other-store"
            assert env[deploy_mod.RUNTIME_STORE_USERNAME_ENV] == "sp-123"
            assert (
                env[deploy_mod.RUNTIME_STORE_LAKEBASE_BRANCH_ENV]
                == "projects/databricks-internal-custom-agents/branches/production"
            )
            assert deploy_mod.RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV not in env
            assert deploy_mod.RUNTIME_STORE_SCHEMA_ENV not in env
            events.append("app-deployed")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(deploy_mod, "_databricks", fake_databricks)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)
    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=ctx)

    assert result.exit_code == 0, result.output
    assert events == ["app-created", "runtime-store", "app-deployed"]


def test_deploy_renames_underlying_app_compute_output(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: False)
    monkeypatch.setattr(AppsClient, "wait_for_running", lambda self, name, timeout_s=300: None)
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
        "action": "Could not create deployment 'agent-mason-myapp'.",
    }
    assert apps_calls[1][1] == {"action": "Could not deploy 'agent-mason-myapp'."}
    assert "Agent compute is starting" in result.output
    assert "App compute" not in result.output
    get_call = next(call for call in calls if call[0][:2] == ["apps", "get"])
    assert get_call[1] == {"capture": True, "check": False}


def test_deploy_reports_app_url(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        AppsClient, "url", lambda self, name: "https://myapp-123.databricksapps.com"
    )
    captured: dict = {}
    monkeypatch.setattr(deploy_mod.render, "emit_json", lambda data: captured.update(data))

    class _JsonCtx(_FakeCtx):
        output = "json"

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_JsonCtx())

    assert result.exit_code == 0, result.output
    assert captured["url"] == "https://myapp-123.databricksapps.com"


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize("server,chat_ui", [("mason", True), ("mason", False), ("custom", False)])
def test_deploy_recommends_invoking_deployed_agent(
    tmp_path: pathlib.Path,
    monkeypatch,
    framework: str,
    server: str,
    chat_ui: bool,
):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(src, framework=framework, server=server)
    if chat_ui:
        (src / "runtime").mkdir()
        (src / "runtime" / "ui.py").write_text("# chat UI\n")
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", True)

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(AppsClient, "service_principal", lambda *args: "sp-123")
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
    commands = [line for line in result.output.splitlines() if line.startswith("mason endpoint")]
    assert len(commands) == 1, result.output
    command = commands[0]
    path = "/api/invocations" if server == "mason" else "/invocations"
    assert f"mason endpoint invoke agent-mason-myapp --path {path} --json " in command
    assert "│" not in command
    assert ("$(uuidgen)" in command) is (server == "mason")
    panel, example = result.output.split("Invoke with Mason\n")
    assert panel.splitlines()[-1].startswith("╰")
    assert example.splitlines() == [command]
    for existing_command in ("mason deployments get", "mason deployments logs"):
        assert any(line.startswith("│") and existing_command in line for line in panel.splitlines())
    assert "Runtime Store" not in result.output
    assert "runtime-agent-mason-myapp-550e8400-e29b-41d4-a716-446655440000" not in result.output
    env = {
        entry["name"]: entry["value"]
        for entry in yaml.safe_load((src / "app.yaml").read_text()).get("env", [])
    }
    if server == "mason":
        # Hiding the display field must not disable the deployed runtime's backend.
        assert env["DATABRICKS_MASON_RUNTIME_STORE_LAKEBASE_BRANCH"] == (
            "projects/databricks-internal-custom-agents/branches/production"
        )


def test_deploy_sync_keeps_directly_edited_agent_manifest(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    (src / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "custom"\n'
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
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
        "/Workspace/Users/me@example.com/mason_deployments/agent-mason-myapp",
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
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: False)  # app doesn't exist yet
    waited = {"called": False}
    monkeypatch.setattr(
        AppsClient,
        "wait_for_running",
        lambda self, name, timeout_s=300: waited.__setitem__("called", True),
    )
    monkeypatch.setattr(AppsClient, "service_principal", lambda self, name: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: (
            calls.append(args) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    assert ["apps", "create", "agent-mason-myapp"] in calls
    assert waited["called"], "must wait for the new app to be running before deploying"


def test_redeploy_waits_for_running_and_skips_create(tmp_path: pathlib.Path, monkeypatch):
    # An existing app is re-deployed: no `apps create` (it would error), but still wait for compute
    # so there's feedback and the app is ACTIVE before `apps deploy`.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    calls: list[list[str]] = []
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)  # already exists
    waited = {"called": False}
    monkeypatch.setattr(
        AppsClient,
        "wait_for_running",
        lambda self, name, timeout_s=300: waited.__setitem__("called", True),
    )
    monkeypatch.setattr(AppsClient, "service_principal", lambda self, name: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: (
            calls.append(args) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    assert ["apps", "create", "agent-mason-myapp"] not in calls  # never re-create an existing app
    assert waited["called"], "re-deploy must also wait for compute"


def test_deploy_injects_store_env(tmp_path: pathlib.Path, monkeypatch):
    # The runtime reads stores from env, never agent.toml: deploy wires the resolved memory id
    # (AGENT_MEMORY_STORE — the entries API is keyed by id) and the session name (AGENT_SESSION_STORE)
    # into app.yaml.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(src, memory="mem", session="sessions")
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", True)

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(AppsClient, "service_principal", lambda *args: "sp-123")

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    env_entries = yaml.safe_load((src / "app.yaml").read_text()).get("env") or []
    env = {entry["name"]: entry["value"] for entry in env_entries}
    assert env["AGENT_MEMORY_STORE"] == "mem-id-123"  # _FakeClient resolves "mem" -> mem-id-123
    assert env["AGENT_SESSION_STORE"] == "sessions"


def test_deploy_wires_tracing_env_and_grants_experiment_resource(
    tmp_path: pathlib.Path, monkeypatch
):
    # Tracing is on by default: deploy wires the two MLflow env vars (id + workspace) into app.yaml
    # and grants the app's SP write access by declaring the experiment as an app resource.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "get_or_create_trace_experiment",
        lambda *a, **k: deploy_mod.ResolvedTraceExperiment("exp-42", MLflowTraceTables()),
    )
    granted: dict = {}
    monkeypatch.setattr(
        deploy_mod,
        "apply_trace_resources",
        lambda app, experiment_id, tables, profile: granted.update(
            app=app, experiment_id=experiment_id, tables=tables
        ),
    )
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    assert env["MLFLOW_EXPERIMENT_ID"] == "exp-42"
    assert env["MLFLOW_TRACKING_URI"] == "databricks"
    # the experiment is granted to the app's SP as an app resource (no manual SQL grant); a managed
    # experiment carries no UC tables
    assert granted == {"app": "agent-mason-myapp", "experiment_id": "exp-42", "tables": []}
    assert "Deployed without tracing" not in result.output  # bound -> no unbound notice


def test_deploy_grants_uc_trace_tables_for_uc_backed_experiment(tmp_path, monkeypatch):
    # A bound experiment that resolves UC-backed: deploy grants the experiment (CAN_EDIT) AND MODIFY
    # on its UC OTEL tables in ONE trace-resource write - the experiment grant alone does not
    # propagate to the UC tables the app exports traces to.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    tables = MLflowTraceTables(spans="cat.schema.otel_spans", logs="cat.schema.otel_logs")
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "get_or_create_trace_experiment",
        lambda *a, **k: deploy_mod.ResolvedTraceExperiment("exp-uc", tables),
    )
    trace_grant = mock.Mock(return_value=None)
    monkeypatch.setattr(deploy_mod, "apply_trace_resources", trace_grant)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    # deploy hands the grant the (kind, table) pairs from the resolved tables
    trace_grant.assert_called_once_with("agent-mason-myapp", "exp-uc", tables.otel_tables(), "prof")
    env = {e["name"]: e["value"] for e in yaml.safe_load((src / "app.yaml").read_text())["env"]}
    assert env["MLFLOW_EXPERIMENT_ID"] == "exp-uc"
    # the success line is the same for UC and managed experiments (they converge)
    out = " ".join(result.output.replace("│", " ").split())
    assert "granted to agent runtime service principal" in out


def test_deploy_grants_managed_experiment_with_no_uc_tables(tmp_path, monkeypatch):
    # A managed experiment has no UC tables: the single trace-resource grant gets an empty table
    # tuple (which also converges away any stale table resources from a prior UC binding).
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "get_or_create_trace_experiment",
        lambda *a, **k: deploy_mod.ResolvedTraceExperiment("exp-42", MLflowTraceTables()),
    )
    trace_grant = mock.Mock(return_value=None)
    monkeypatch.setattr(deploy_mod, "apply_trace_resources", trace_grant)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    trace_grant.assert_called_once_with("agent-mason-myapp", "exp-42", [], "prof")


def test_deploy_proceeds_when_trace_grant_fails(tmp_path, monkeypatch):
    # The trace grant is best-effort: a failure surfaces as next-step guidance but never aborts
    # the deploy.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    tables = MLflowTraceTables(spans="cat.schema.otel_spans")
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "get_or_create_trace_experiment",
        lambda *a, **k: deploy_mod.ResolvedTraceExperiment("exp-uc", tables),
    )
    monkeypatch.setattr(
        deploy_mod,
        "apply_trace_resources",
        mock.Mock(return_value="denied: needs MANAGE on the catalog"),
    )
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output  # deploy still succeeded
    # the panel wraps long lines behind │ borders; strip them so wrapped phrases still match
    out = " ".join(result.output.replace("│", " ").split())
    assert "needs write access to its trace experiment" in out  # grant-failure guidance shown
    assert "denied: needs MANAGE on the catalog" in out  # the cause is surfaced


def test_deploy_reconciles_trace_resources_even_when_unbound(tmp_path, monkeypatch):
    # Unbound (mason tracing unbind): deploy still reconciles the mason-owned trace set - passing
    # experiment_id=None prunes stale mason-trace-* resources left by a previously bound deploy.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    # the autouse fixture already stubs get_or_create_trace_experiment -> None (unbound)
    trace_grant = mock.Mock(return_value=None)
    monkeypatch.setattr(deploy_mod, "apply_trace_resources", trace_grant)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    trace_grant.assert_called_once_with("agent-mason-myapp", None, [], "prof")


def test_deploy_rebind_uc_to_uc_reconciles_to_the_new_table_set(tmp_path, monkeypatch):
    # UC -> UC rebind: the redeploy resolves the NEW experiment's tables and reconciles to them, so
    # the old experiment's mason-trace-table-* resources are dropped in the same write (convergence
    # itself is apply_trace_resources' job; here we guard that deploy passes the new set through).
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    # the app currently carries the OLD experiment's 3 table resources (documentation only - the
    # app state itself is apply_trace_resources' concern, stubbed here)
    old_tables = MLflowTraceTables(
        spans="old.schema.otel_spans",
        logs="old.schema.otel_logs",
        annotations="old.schema.otel_annotations",
    )
    new_tables = MLflowTraceTables(spans="new.schema.otel_spans", logs="new.schema.otel_logs")
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "get_or_create_trace_experiment",
        lambda *a, **k: deploy_mod.ResolvedTraceExperiment("exp-new", new_tables),
    )
    trace_grant = mock.Mock(return_value=None)
    monkeypatch.setattr(deploy_mod, "apply_trace_resources", trace_grant)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    assert len(old_tables.otel_tables()) != len(new_tables.otel_tables())  # the table count changed
    trace_grant.assert_called_once_with(
        "agent-mason-myapp", "exp-new", new_tables.otel_tables(), "prof"
    )


def test_deploy_proceeds_when_tracing_provisioning_raises(tmp_path: pathlib.Path, monkeypatch):
    # Tracing provisioning is best-effort: a non-AgentCliError (e.g. MLflow/network) must not abort
    # the deploy — it proceeds without tracing.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    def _boom(*a, **k):
        raise RuntimeError("mlflow create_experiment blew up")

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(deploy_mod, "get_or_create_trace_experiment", _boom)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())
    assert result.exit_code == 0, result.output  # deploy still succeeded
    env_entries = yaml.safe_load((src / "app.yaml").read_text()).get("env") or []
    assert not any(e["name"].startswith("MLFLOW") for e in env_entries)  # tracing skipped
    out = " ".join(result.output.split())
    assert "Deployed without tracing" in out and "mason tracing bind" in out  # guidance shown
    assert "mlflow create_experiment blew up" in out  # the cause is surfaced


def test_deploy_notifies_when_tracing_unbound(tmp_path: pathlib.Path, monkeypatch):
    # Unbound tracing deploys silently otherwise; surface a next-step so the developer knows (in case
    # it wasn't intended) and can enable it. The autouse fixture stubs resolve -> None (unbound).
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "Deployed without tracing" in out
    assert "mason tracing bind" in out  # points at the (parameter-free) enable command
    assert "Tracing setup failed" not in out  # unbound is not an error, so no cause suffix


def test_deploy_resolves_existing_memory_store_by_display_name(tmp_path: pathlib.Path, monkeypatch):
    # deploy reconciles the declared store; when it already exists it is resolved by display name
    # (list+match, not get_memory_store which keys on resource id) and its id is injected into app.yaml.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src, memory="mem")
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(AppsClient, "service_principal", lambda self, name: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    # _FakeClient resolves "mem" via list+match and returns id mem-id-123; deploy succeeds.
    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())
    assert result.exit_code == 0, result.output
    env_entries = yaml.safe_load((src / "app.yaml").read_text()).get("env") or []
    env = {e["name"]: e["value"] for e in env_entries}
    assert env.get("AGENT_MEMORY_STORE") == "mem-id-123"


def test_deploy_creates_missing_declared_store(tmp_path: pathlib.Path, monkeypatch):
    # A declared-but-missing store is created on deploy (not an error); agent.toml is never rewritten.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src, memory="ghost")
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())
    assert result.exit_code == 0, result.output
    assert "Created memory store 'ghost'" in result.output


def test_mlflow_tracing_config_binds_experiment_by_id_and_workspace():
    # The agent binding is exactly two env vars: the workspace (destination) and the experiment id.
    assert deploy_mod.mlflow_tracing_config("exp-9").env() == {
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_EXPERIMENT_ID": "exp-9",
    }


def test_resolve_trace_experiment_none_when_unbound(tmp_path: pathlib.Path, monkeypatch):
    # No experiment_name bound -> tracing is off; nothing is created and None is returned (no default
    # fallback). A legacy [tracing] disabled key is simply ignored.
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "mason"\n'
    )
    called: list[str] = []
    monkeypatch.setattr(
        deploy_mod,
        "create_experiment_idempotent",
        lambda profile, client, name: called.append(name) or "should-not-happen",
    )
    assert _REAL_RESOLVE_TRACE(tmp_path, _FakeClient(), None) is None
    assert called == []  # unbound -> no experiment provisioned


def test_resolve_trace_experiment_get_or_creates_bound_name(tmp_path: pathlib.Path, monkeypatch):
    # A bound experiment_name is get-or-created in the current workspace; the name (not an id) stays
    # in agent.toml, so nothing is pinned back.
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "mason"\n'
        '\n[tracing]\nexperiment_name = "/Shared/mason_traces/bound"\n'
    )
    created: dict = {}
    monkeypatch.setattr(
        deploy_mod,
        "create_experiment_idempotent",
        lambda profile, client, name: created.update(name=name)
        or ResolvedTraceExperiment("id-b", MLflowTraceTables()),
    )
    assert _REAL_RESOLVE_TRACE(tmp_path, _FakeClient(), None) == deploy_mod.ResolvedTraceExperiment(
        "id-b", MLflowTraceTables()
    )
    assert created["name"] == "/Shared/mason_traces/bound"
    from databricks_mason.agent_project import AgentProject

    project = AgentProject.load(tmp_path)
    assert project.trace_experiment_name == "/Shared/mason_traces/bound"  # name kept


def test_resolve_trace_experiment_get_or_creates_by_name_each_run(
    tmp_path: pathlib.Path, monkeypatch
):
    # Nothing is pinned back, so each run re-resolves the bound name and get-or-creates it (idempotent).
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "mason"\n'
        '\n[tracing]\nexperiment_name = "/Shared/mason_traces/bound"\n'
    )
    calls: list[str] = []
    monkeypatch.setattr(
        deploy_mod,
        "create_experiment_idempotent",
        lambda profile, client, name: calls.append(name)
        or ResolvedTraceExperiment("made-id", MLflowTraceTables()),
    )
    assert _REAL_RESOLVE_TRACE(tmp_path, _FakeClient(), None) == deploy_mod.ResolvedTraceExperiment(
        "made-id", MLflowTraceTables()
    )
    assert _REAL_RESOLVE_TRACE(tmp_path, _FakeClient(), None) == deploy_mod.ResolvedTraceExperiment(
        "made-id", MLflowTraceTables()
    )
    assert calls == ["/Shared/mason_traces/bound", "/Shared/mason_traces/bound"]


def _run_deploy(src, monkeypatch, extra_args):
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
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


class _JsonCtx(_FakeCtx):
    output = "json"


def test_lifecycle_commands_honor_json_output(monkeypatch):
    monkeypatch.setattr(AppsClient, "service_principal", lambda *args: "sp-123")
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


def _agent_toml(
    source: pathlib.Path,
    *,
    server: str = "custom",
    memory=None,
    session=None,
    experiment=None,
    deployment_name=None,
) -> None:
    text = f'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "{server}"\n'
    if deployment_name:
        text += f'deployment_name = "{deployment_name}"\n'
    if memory:
        text += f'\n[memory_store]\nname = "{memory}"\n'
    if session:
        text += f'\n[session_store]\nname = "{session}"\n'
    if experiment:
        text += f'\n[tracing]\nexperiment_name = "{experiment}"\n'
    (source / "agent.toml").write_text(text, encoding="utf-8")


def test_resource_bindings_reads_agent_toml(tmp_path: pathlib.Path):
    _agent_toml(
        tmp_path, memory="bound-mem", session="bound-sess", experiment="/Shared/mason_traces/x"
    )
    assert deploy_mod.resource_bindings(tmp_path) == (
        "bound-mem",
        "bound-sess",
        "/Shared/mason_traces/x",
    )


def test_resource_bindings_none_when_unbound(tmp_path: pathlib.Path):
    _agent_toml(tmp_path)  # scaffold with no resource tables
    assert deploy_mod.resource_bindings(tmp_path) == (None, None, None)


def test_resource_bindings_ignores_missing_manifest(tmp_path: pathlib.Path):
    # No agent.toml -> nothing bound, never raises (so deploy/dev aren't blocked).
    assert deploy_mod.resource_bindings(tmp_path) == (None, None, None)


def test_deploy_writes_deployment_name_to_toml(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src)  # a project with no deployment_name yet

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    assert AgentProject.load(src).deployment_name == "myapp"  # persisted for later deploys


def test_deploy_reads_deployment_name_from_toml_when_omitted(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src, deployment_name="stored")

    calls: list[list[str]] = []
    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: (
            calls.append(args) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    ws = "/Workspace/Users/me@example.com/mason_deployments/agent-mason-stored"
    assert ["apps", "deploy", "agent-mason-stored", "--source-code-path", ws] in calls


def test_deploy_without_name_or_toml_errors(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))  # no agent.toml

    called: list = []
    monkeypatch.setattr(deploy_mod, "_databricks", lambda *a, **k: called.append(a))

    result = CliRunner().invoke(deploy_mod.deploy, ["--source", str(src)], obj=_FakeCtx())

    assert result.exit_code != 0
    assert "No deployment name" in result.output
    assert called == []  # errored before shelling out to `databricks apps`


def test_deploy_creates_declared_but_missing_store_without_writing_agent_toml(
    tmp_path, monkeypatch
):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    # Include deployment_name so deploy's name-persist write doesn't change the file.
    _agent_toml(src, memory="declared-mem", session="declared-sess", deployment_name="myapp")
    before = (src / "agent.toml").read_text()

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    assert "Created memory store 'declared-mem'" in result.output
    assert (src / "agent.toml").read_text() == before  # deploy never rewrites the manifest


def test_deploy_grants_bound_store(tmp_path: pathlib.Path, monkeypatch):
    # `mason sessions bind` then plain `mason deploy`: the binding must drive both the
    # app.yaml env AND the SP access grant, or the deployed app can't reach its durable store.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src, session="bound-sess")

    monkeypatch.setattr(AppsClient, "exists", lambda self, name: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(AppsClient, "service_principal", lambda self, name: "sp-123")
    grant_args: dict = {}
    monkeypatch.setattr(
        StoreProvisioner,
        "grant_store_access",
        lambda self, sp, session_store, memory_store: (
            grant_args.update(sp=sp, session_store=session_store, memory_store=memory_store) or None
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    # The grant fired for the bound session store, and its name was wired into app.yaml as
    # AGENT_SESSION_STORE. AGENT_MEMORY_STORE is absent because no memory store is declared.
    assert grant_args == {"sp": "sp-123", "session_store": "bound-sess", "memory_store": None}
    env_entries = yaml.safe_load((src / "app.yaml").read_text()).get("env") or []
    env = {e["name"]: e["value"] for e in env_entries}
    assert env["AGENT_SESSION_STORE"] == "bound-sess"
    assert "AGENT_MEMORY_STORE" not in env


# ---------------------------------------------------------------------------
# DeployOrchestrator unit tests - exercise the DI seam directly with fakes
# ---------------------------------------------------------------------------


def _make_orchestrator(*, runner=None, apps=None, stores=None, client=None):
    """Build a minimal DeployOrchestrator with injectable fakes."""
    import types as _types

    fake_client = client or _FakeClient()
    fake_runner = runner or (
        lambda args, profile, **kw: _types.SimpleNamespace(returncode=0, stdout="", stderr="")
    )

    class _FakeApps:
        def exists(self, name):
            return True

        def wait_for_running(self, name, timeout_s=300):
            pass

        def service_principal(self, name):
            return "sp-fake-123"

        def url(self, name):
            return None

    class _FakeStores:
        def grant_store_access(self, sp, session_store, memory_store):
            return None

    fake_stores = stores or _FakeStores()
    return deploy_mod.DeployOrchestrator(
        client_factory=lambda: fake_client,
        apps=apps or _FakeApps(),
        stores_factory=lambda _c: fake_stores,
        profile="prof",
        output="text",
        runner=fake_runner,
    )


def test_orchestrator_grant_access_succeeds_when_sp_found():
    # _grant_access returns (True, None, None) when stores are declared and SP is resolvable.
    from databricks_mason.cli.tracing import MLflowTraceTables

    granted: dict = {}

    class _AppsWithSP:
        def service_principal(self, name):
            return "sp-real"

        def url(self, name):
            return None

    class _Stores:
        def grant_store_access(self, sp, session_store, memory_store):
            granted.update(sp=sp, session=session_store, memory=memory_store)
            return None

    orch = _make_orchestrator(apps=_AppsWithSP(), stores=_Stores())
    # Seed _stores so _grant_access can use it (run() normally does this after _authorize).
    orch._stores = _Stores()
    grants_stores, grant_error, trace_grant_error = orch._grant_access(
        "agent-mason-myapp", "mem-store", "sess-store", None, MLflowTraceTables()
    )

    assert grants_stores is True
    assert grant_error is None
    assert trace_grant_error is None
    assert granted == {"sp": "sp-real", "session": "sess-store", "memory": "mem-store"}


def test_orchestrator_grant_access_surfaces_error_when_sp_none():
    # _grant_access sets grant_error when service_principal returns None; deploy still continues.
    from databricks_mason.cli.tracing import MLflowTraceTables

    class _AppsNoSP:
        def service_principal(self, name):
            return None

        def url(self, name):
            return None

    orch = _make_orchestrator(apps=_AppsNoSP())
    grants_stores, grant_error, trace_grant_error = orch._grant_access(
        "agent-mason-myapp", "mem-store", None, None, MLflowTraceTables()
    )

    assert grants_stores is True  # memory_store is truthy -> grant was attempted
    assert grant_error is not None
    assert "service principal" in grant_error


def test_orchestrator_ensure_app_issues_create_when_not_exists():
    # _ensure_app calls runner with ["apps", "create", name, ...] when app does not yet exist.
    import types as _types

    runner_calls: list[list[str]] = []

    def fake_runner(args, profile, **kw):
        runner_calls.append(args)
        return _types.SimpleNamespace(returncode=0, stdout="", stderr="")

    class _AppsNotExists:
        def exists(self, name):
            return False

        def wait_for_running(self, name, timeout_s=300):
            pass

    orch = _make_orchestrator(runner=fake_runner, apps=_AppsNotExists())
    orch._ensure_app("agent-mason-myapp", None, None, [])

    create_calls = [c for c in runner_calls if c[:2] == ["apps", "create"]]
    assert len(create_calls) == 1
    assert create_calls[0][2] == "agent-mason-myapp"


def test_orchestrator_ensure_app_skips_create_when_exists():
    # _ensure_app does NOT call runner with apps create when the app already exists.
    import types as _types

    runner_calls: list[list[str]] = []

    def fake_runner(args, profile, **kw):
        runner_calls.append(args)
        return _types.SimpleNamespace(returncode=0, stdout="", stderr="")

    class _AppsExists:
        def exists(self, name):
            return True

        def wait_for_running(self, name, timeout_s=300):
            pass

    orch = _make_orchestrator(runner=fake_runner, apps=_AppsExists())
    orch._ensure_app("agent-mason-myapp", None, None, [])

    create_calls = [c for c in runner_calls if c[:2] == ["apps", "create"]]
    assert create_calls == []  # no create issued for an already-existing app
