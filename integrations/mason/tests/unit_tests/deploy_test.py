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
from databricks_mason.agent_project import AgentProject, ToolSpec
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import write_project_metadata

# The autouse fixture below stubs `resolve_trace_experiment_id` for deploy-command tests; capture the
# real function here so its own unit tests can exercise the actual logic.
_REAL_RESOLVE_TRACE = deploy_mod.resolve_trace_experiment_id


@pytest.fixture(autouse=True)
def _compute_active(monkeypatch):
    # `mason deploy` now waits for compute on every deploy; report ACTIVE so the wait returns
    # immediately. Tests that exercise _wait_for_running directly override _app_compute_state.
    monkeypatch.setattr(deploy_mod, "_app_compute_state", lambda name, profile: "ACTIVE")


@pytest.fixture(autouse=True)
def _no_tracing_by_default(monkeypatch):
    # Tracing is on by default and would create an MLflow experiment (a live workspace op); stub the
    # provisioning off so non-tracing deploy tests stay hermetic. Tracing tests override this.
    monkeypatch.setattr(deploy_mod, "resolve_trace_experiment_id", lambda *a, **k: None)


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

    def __init__(self):
        # Seeded with one pre-existing store ("mem", whose id differs from its display name as the
        # real API returns); created stores are appended so deploy's auto-create can then resolve them.
        self._memory_stores = [{"name": "memory-stores/mem-id-123", "display_name": "mem"}]

    def get_memory_store(self, name):
        return {"name": f"memory-stores/{name}"}

    def list_memory_stores(self, page_size=None, page_token=None):
        return {"managed_memory_stores": list(self._memory_stores), "next_page_token": ""}

    def create_memory_store(self, display_name, *, retry_transient=False):
        store = {"name": f"memory-stores/{display_name}", "display_name": display_name}
        self._memory_stores.append(store)
        return store

    def get_session_store(self, name):
        return {"session_store_name": name}

    def create_session_store(self, name, *, retry_transient=False):
        return {"session_store_name": name}


class _FakeCtx:
    profile = "prof"
    output = "text"

    def client(self):
        return _FakeClient()


def _mark_template(source: pathlib.Path, template: str) -> None:
    config = source / ".mason" / "project.toml"
    config.parent.mkdir(parents=True)
    config.write_text(f'schema_version = 1\nframework = "langgraph"\ntemplate = "{template}"\n')


def _write_agent_manifest(
    source: pathlib.Path,
    *,
    durability: bool = False,
    memory: str | None = None,
    session: str | None = None,
) -> None:
    body = 'schema_version = 1\n\n[agent]\nframework = "langgraph"\n'
    if memory:
        body += f'\n[memory_store]\nname = "{memory}"\n'
    if session:
        body += f'\n[session_store]\nname = "{session}"\n'
    if durability:
        body += "\n[durability]\nenabled = true\n"
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
    project = AgentProject.create(source, framework=framework)
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


@pytest.mark.parametrize(
    ("framework", "template"),
    [
        ("langgraph", "custom-agent-langgraph"),
        ("openai", "custom-agent-openai"),
    ],
)
def test_deploy_surfaces_invalid_custom_server_manifest_before_mutation_or_network(
    tmp_path: pathlib.Path,
    framework: str,
    template: str,
):
    source = tmp_path / template
    source.mkdir()
    (source / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    manifest = source / "agent.toml"
    manifest.write_text(
        f'schema_version = 1\n\n[agent]\nframework = "{framework}"\n'
        '\n[[tools]]\nid = "legacy"\nsource = { kind = "python", '
        'entrypoint = "agent.tools:legacy" }\n',
        encoding="utf-8",
    )
    write_project_metadata(source, framework=framework, template=template)
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


def test_deploy_non_durable_template_does_not_enable_runtime_store(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _mark_template(src, "agent-langgraph")
    _write_agent_manifest(src)

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod.lakebase_durability_store,
        "get_or_create_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not provision")),
    )
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

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
    assert "DATABRICKS_MASON_RUNTIME_ENDPOINT" not in env


def test_deploy_durability_binding_uses_dedicated_backend_with_session_store(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(src, durability=True, session="sessions")
    selected = deploy_mod.lakebase_durability_store.backend("mason-myapp")
    session_backend = deploy_mod.session_store_access.backend("sessions")
    events = []

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod.lakebase_durability_store,
        "get_or_create_backend",
        lambda app, profile, create: selected,
    )
    monkeypatch.setattr(
        deploy_mod.session_store_access,
        "backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not use session store for durability")
        ),
    )
    monkeypatch.setattr(
        deploy_mod,
        "apply_postgres_resources",
        lambda app, backends, profile: events.append(("attach", backends)) or None,
    )
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda app, profile: "sp")
    monkeypatch.setattr(deploy_mod, "_grant_store_access", lambda *args, **kwargs: None)

    def fake_databricks(args, profile, **kwargs):
        if args[:2] == ["apps", "deploy"]:
            events.append(("deploy", args))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(deploy_mod, "_databricks", fake_databricks)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--no-create-stores"],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    assert [event[0] for event in events] == ["attach", "deploy"]
    backend = events[0][1][0]
    assert backend == selected
    assert backend.database != "sessions"
    assert backend.resource_name != session_backend.resource_name
    assert backend.schema == deploy_mod.lakebase_durability_store.get_lakebase_schema("mason-myapp")
    assert backend.tables == ()
    env = {
        entry["name"]: entry["value"]
        for entry in yaml.safe_load((src / "app.yaml").read_text())["env"]
    }
    assert env["DATABRICKS_MASON_RUNTIME_ENDPOINT"] == backend.endpoint_path
    assert env["DATABRICKS_MASON_RUNTIME_SCHEMA"] == (
        deploy_mod.lakebase_durability_store.get_lakebase_schema("mason-myapp")
    )


def test_deploy_durability_binding_does_not_reuse_memory_store(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(src, durability=True, memory="mem")
    selected = deploy_mod.lakebase_durability_store.backend("mason-myapp")
    events = []

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "_memory_store_database", lambda client, store: "memory-db")
    monkeypatch.setattr(
        deploy_mod.lakebase_durability_store,
        "get_or_create_backend",
        lambda app, profile, create: selected,
    )
    monkeypatch.setattr(
        deploy_mod,
        "apply_postgres_resources",
        lambda app, backends, profile: events.append(("attach", backends)) or None,
    )
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda app, profile: "sp")
    monkeypatch.setattr(deploy_mod, "_grant_store_access", lambda *args, **kwargs: None)

    def fake_databricks(args, profile, **kwargs):
        if args[:2] == ["apps", "deploy"]:
            events.append(("deploy", args))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(deploy_mod, "_databricks", fake_databricks)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["myapp", "--source", str(src), "--no-create-stores"],
        obj=_FakeCtx(),
    )

    assert result.exit_code == 0, result.output
    assert [event[0] for event in events] == ["attach", "deploy"]
    assert events[0][1] == [selected]


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


def test_deploy_wires_tracing_env_and_grants_experiment_resource(
    tmp_path: pathlib.Path, monkeypatch
):
    # Tracing is on by default: deploy wires the two MLflow env vars (id + workspace) into app.yaml
    # and grants the app's SP write access by declaring the experiment as an app resource.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "resolve_trace_experiment_id", lambda *a, **k: "exp-42")
    granted: dict = {}
    monkeypatch.setattr(
        deploy_mod,
        "apply_experiment_resource",
        lambda app, experiment_id, profile: granted.update(app=app, experiment_id=experiment_id),
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
    # the experiment is granted to the app's SP as an app resource (no manual SQL grant)
    assert granted == {"app": "mason-myapp", "experiment_id": "exp-42"}


def test_deploy_keys_experiment_on_source_dir_name_not_prefixed(
    tmp_path: pathlib.Path, monkeypatch
):
    # dev keys the experiment on the source dir name; deploy must match it (NOT the mason-prefixed
    # deployment name), so dev and deploy trace to the same per-project experiment.
    src = tmp_path / "my-agent"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    captured: dict = {}
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "resolve_trace_experiment_id",
        lambda source, app, client, profile: captured.update(app=app) or None,
    )
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(
        deploy_mod.deploy, ["my-agent", "--source", str(src)], obj=_FakeCtx()
    )
    assert result.exit_code == 0, result.output
    assert captured["app"] == "my-agent"  # source dir name, not "mason-my-agent"


def test_deploy_proceeds_when_tracing_provisioning_raises(tmp_path: pathlib.Path, monkeypatch):
    # Tracing provisioning is best-effort: a non-AgentCliError (e.g. MLflow/network) must not abort
    # the deploy — it proceeds without tracing.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))

    def _boom(*a, **k):
        raise RuntimeError("mlflow create_experiment blew up")

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(deploy_mod, "resolve_trace_experiment_id", _boom)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())
    assert result.exit_code == 0, result.output  # deploy still succeeded
    env_entries = yaml.safe_load((src / "app.yaml").read_text()).get("env") or []
    assert not any(e["name"].startswith("MLFLOW") for e in env_entries)  # tracing skipped


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


def test_mlflow_tracing_config_binds_experiment_by_id_and_workspace():
    # The agent binding is exactly two env vars: the workspace (destination) and the experiment id.
    assert deploy_mod.mlflow_tracing_config("exp-9").env() == {
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_EXPERIMENT_ID": "exp-9",
    }


def test_resolve_trace_experiment_none_when_disabled(tmp_path: pathlib.Path):
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\n\n[tracing]\ndisabled = true\n'
    )
    assert _REAL_RESOLVE_TRACE(tmp_path, "app", _FakeClient(), None) is None


def test_resolve_trace_experiment_uses_pinned_id_without_creating(
    tmp_path: pathlib.Path, monkeypatch
):
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\n\n[tracing]\nexperiment_id = "pinned-1"\n'
    )
    # A pinned id is used directly — no experiment creation.
    monkeypatch.setattr(
        deploy_mod, "create_experiment_idempotent", lambda *a, **k: pytest.fail("should not create")
    )
    assert _REAL_RESOLVE_TRACE(tmp_path, "app", _FakeClient(), None) == "pinned-1"


def test_resolve_trace_experiment_creates_per_project_default(tmp_path: pathlib.Path, monkeypatch):
    (tmp_path / "agent.toml").write_text('schema_version = 1\n\n[agent]\nframework = "openai"\n')
    created: dict = {}
    monkeypatch.setattr(
        deploy_mod,
        "create_experiment_idempotent",
        lambda profile, client, name: created.update(name=name) or "made-id",
    )
    exp_id = _REAL_RESOLVE_TRACE(tmp_path, "my-agent", _FakeClient(), None)
    assert exp_id == "made-id"
    assert created["name"] == "/Users/me@example.com/mason-traces/my-agent"
    # First run pins the resolved default into agent.toml so later runs reuse it by id.
    from databricks_mason.agent_project import AgentProject

    assert AgentProject.load(tmp_path).trace_experiment_id == "made-id"


def test_resolve_trace_experiment_reuses_pinned_default_on_second_run(
    tmp_path: pathlib.Path, monkeypatch
):
    (tmp_path / "agent.toml").write_text('schema_version = 1\n\n[agent]\nframework = "openai"\n')
    calls: list[str] = []
    monkeypatch.setattr(
        deploy_mod,
        "create_experiment_idempotent",
        lambda profile, client, name: calls.append(name) or "made-id",
    )
    # First run creates + pins; the second reads the pinned id straight from agent.toml.
    assert _REAL_RESOLVE_TRACE(tmp_path, "my-agent", _FakeClient(), None) == "made-id"
    assert _REAL_RESOLVE_TRACE(tmp_path, "my-agent", _FakeClient(), None) == "made-id"
    assert calls == ["/Users/me@example.com/mason-traces/my-agent"]  # created only once


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


def _agent_toml(source: pathlib.Path, *, memory=None, session=None, deployment_name=None) -> None:
    text = 'schema_version = 1\n\n[agent]\nframework = "openai"\n'
    if deployment_name:
        text += f'deployment_name = "{deployment_name}"\n'
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


def test_deploy_writes_deployment_name_to_toml(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src)  # a project with no deployment_name yet

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
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
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: (
            calls.append(args) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    ws = "/Workspace/Users/me@example.com/mason_deployments/mason-stored"
    assert ["apps", "deploy", "mason-stored", "--source-code-path", ws] in calls


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


def test_ensure_default_stores_creates_and_binds_unbound(tmp_path: pathlib.Path):
    _agent_toml(tmp_path)  # a project with no stores bound
    project = AgentProject.load(tmp_path)

    deploy_mod._ensure_default_stores(project, "foo", _FakeClient())

    assert project.memory_store == "foo-memory"
    assert project.session_store == "foo-session"
    reloaded = AgentProject.load(tmp_path)  # persisted for later deploys
    assert reloaded.memory_store == "foo-memory"
    assert reloaded.session_store == "foo-session"


def test_ensure_default_stores_leaves_a_bound_store_alone(tmp_path: pathlib.Path):
    _agent_toml(tmp_path, memory="existing-mem")  # memory bound, session unbound
    project = AgentProject.load(tmp_path)

    deploy_mod._ensure_default_stores(project, "foo", _FakeClient())

    assert project.memory_store == "existing-mem"  # untouched
    assert project.session_store == "foo-session"  # only the missing one filled in


def test_deploy_auto_creates_default_stores(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src)  # no stores bound

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(deploy_mod.deploy, ["myapp", "--source", str(src)], obj=_FakeCtx())

    assert result.exit_code == 0, result.output
    project = AgentProject.load(src)
    assert project.memory_store == "myapp-memory"
    assert project.session_store == "myapp-session"


def test_deploy_no_create_stores_leaves_stores_unbound(tmp_path: pathlib.Path, monkeypatch):
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _agent_toml(src)  # no stores bound

    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda a, p: True)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, profile, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(
        deploy_mod.deploy, ["myapp", "--source", str(src), "--no-create-stores"], obj=_FakeCtx()
    )

    assert result.exit_code == 0, result.output
    project = AgentProject.load(src)
    assert project.memory_store is None
    assert project.session_store is None


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
