"""Unit tests for the deploy CLI adapter and its resource/client boundaries.

The deployment workflow used to live in this command module. It is now owned by ``DeployService``
and its clients/provisioners (covered by ``deploy_service_test.py`` and resource-specific tests),
so this file focuses on the CLI adapter, manifest contract, and client behavior observable there.
"""

from __future__ import annotations

import json
import pathlib
import types
from dataclasses import replace
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.conversation_store_client import (
    MemoryStoreClient,
    SessionStoreClient,
)
from databricks_agentbricks.clients.tracing_client import (
    MLflowTraceTables,
    ResolvedTraceExperiment,
    TracingClient,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.projects.app_manifest import AppManifest
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.services.deploy_service import (
    DeployRequest,
    DeployResult,
    ToolAccessSummary,
)
from databricks_agentbricks.services.deployment.config import (
    _DEFAULT_PIP_INDEX_URL,
    _USE_MANAGED_RUNTIME_STORE,
    TRACES_EXPERIMENT_ID_ENV,
    TRACES_TRACKING_URI_ENV,
    mlflow_tracing_config,
)
from databricks_agentbricks.services.deployment.names import DeploymentName


class _FakeApiClient:
    host = "https://workspace.example"
    current_user = "me@example.com"


class _Ctx:
    profile = "prof"
    output = "text"

    def __init__(self, *, output: str = "text"):
        self.output = output
        self.api_client_provider = types.SimpleNamespace(get=lambda: _FakeApiClient())


def _result(*, deployment: str = "agent-bricks-demo") -> DeployResult:
    return DeployResult(
        deployment=deployment,
        source="/tmp/agent",
        url="https://demo.example",
        workspace_path="/Workspace/demo",
        env={"KEEP": "value"},
        client_host="https://workspace.example",
        memory_store=None,
        session_store=None,
        trace_experiment_id=None,
        uc_trace_tables=[],
        trace_setup_error=None,
        trace_grant_error=None,
        memory_grant_error=None,
        session_grant_error=None,
        memory_grant_attempted=False,
        session_grant_attempted=False,
        tool_access=None,
        created_app_yaml=False,
        pip_index_url=None,
        instance_count=None,
        uses_runtime_api=True,
    )


class _DeployApi:
    """Small fake for the external Agent Bricks API used by concrete provisioners."""

    host = "https://workspace.example"
    current_user = "me@example.com"
    workspace_client = object()

    def __init__(self, events: list[tuple], *, fail_memory_grant: bool = False):
        self.events = events
        self.fail_memory_grant = fail_memory_grant

    def create_memory_store(self, display_name, *, retry_transient=False):
        self.events.append(("memory.create", display_name, retry_transient))
        return {"name": "memory-stores/memory-id", "display_name": display_name}

    def create_session_store(self, name, *, retry_transient=False):
        self.events.append(("session.create", name, retry_transient))
        return {"name": name}

    def grant_memory_store_permission(self, resource_name, principal):
        self.events.append(("memory.grant", resource_name, principal))
        if self.fail_memory_grant:
            raise AgentCliError("memory grant denied")

    def grant_session_store_permission(self, resource_name, principal):
        self.events.append(("session.grant", resource_name, principal))

    def create_runtime_store(self, app, principal, *, app_name, retry_transient=False):
        self.events.append(("runtime.create", app, principal, app_name, retry_transient))
        return {
            "name": f"runtime-stores/{app}",
            "owner": {"app": {"name": app, "service_principal_id": principal}},
            "storage_backend": {
                "lakebase": {
                    "branch": "projects/runtime/branches/production",
                    "database_id": "runtime-db",
                }
            },
        }


class _DeployRunner:
    """Fake `databricks` runner that records argv and serves an Apps JSON payload."""

    def __init__(self, events: list[tuple], *, exists: bool = False):
        self.events = events
        self.exists = exists
        self.resources: list[dict] = []

    def __call__(self, args, profile, **kwargs):
        args = list(args)
        self.events.append(("apps", tuple(args)))
        if args[:2] == ["apps", "get"]:
            if len(args) == 3:
                return types.SimpleNamespace(
                    returncode=0 if self.exists else 1,
                    stdout="" if self.exists else "",
                    stderr="" if self.exists else "not found",
                )
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "compute_status": {"state": "ACTIVE"},
                        "service_principal_client_id": "sp-123",
                        "url": "https://agent.example",
                        "resources": self.resources,
                    }
                ),
                stderr="",
            )
        if args[:2] == ["apps", "create-update"]:
            payload = json.loads(args[args.index("--json") + 1])
            if "resources" in payload.get("app", {}):
                self.resources = payload["app"]["resources"]
        if args[:2] == ["apps", "create"]:
            return types.SimpleNamespace(returncode=0, stdout="App compute", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _real_deploy_context(api: _DeployApi, *, output: str = "text"):
    return types.SimpleNamespace(
        profile="prof",
        output=output,
        api_client_provider=types.SimpleNamespace(get=lambda: api),
    )


def test_manifest_env_scaffolds_when_missing(tmp_path: pathlib.Path):
    scaffolded = AppManifest.upsert_env_file(tmp_path, {"AGENT_MEMORY_STORE": "memory-stores/x"})

    assert scaffolded is True
    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert {"name": "AGENT_MEMORY_STORE", "value": "memory-stores/x"} in doc["env"]
    assert "command" in doc


def test_manifest_env_updates_existing_and_preserves_command(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["uvicorn", "app:app"],
                "env": [{"name": "AGENT_MEMORY_STORE", "value": "old"}],
            }
        )
    )

    scaffolded = AppManifest.upsert_env_file(
        tmp_path, {"AGENT_MEMORY_STORE": "new", "AGENT_SESSION_STORE": "s"}
    )

    assert scaffolded is False
    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert doc["command"] == ["uvicorn", "app:app"]
    assert {entry["name"]: entry["value"] for entry in doc["env"]} == {
        "AGENT_MEMORY_STORE": "new",
        "AGENT_SESSION_STORE": "s",
    }


def test_manifest_env_drops_value_from_invalid_entries_and_removals(tmp_path: pathlib.Path):
    unrelated = {"name": "USER_SECRET", "valueFrom": "user-secret-resource"}
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["x"],
                "env": [
                    unrelated,
                    {"name": "AGENT_MEMORY_STORE", "valueFrom": "old-resource"},
                    "invalid-entry",
                    {"name": "MLFLOW_EXPERIMENT_ID", "value": "old"},
                ],
            }
        )
    )

    AppManifest.upsert_env_file(
        tmp_path,
        {"AGENT_MEMORY_STORE": "new"},
        removals=["MLFLOW_EXPERIMENT_ID"],
    )

    doc = yaml.safe_load((tmp_path / "app.yaml").read_text())
    assert doc["env"] == [unrelated, {"name": "AGENT_MEMORY_STORE", "value": "new"}]


def test_manifest_removal_only_does_not_scaffold_missing_file(tmp_path: pathlib.Path):
    assert AppManifest.upsert_env_file(tmp_path, {}, removals=["MLFLOW_EXPERIMENT_ID"]) is False
    assert not (tmp_path / "app.yaml").exists()


def test_managed_runtime_store_is_the_internal_default():
    assert _USE_MANAGED_RUNTIME_STORE is True


def test_memory_store_client_reuses_existing_store_and_pages_by_display_name():
    api = mock.Mock()
    api.create_memory_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    api.list_memory_stores.side_effect = [
        {
            "managed_memory_stores": [{"name": "memory-stores/other", "display_name": "other"}],
            "next_page_token": "next",
        },
        {
            "managed_memory_stores": [
                {"name": "memory-stores/mem-id-123", "display_name": "wanted"}
            ],
            "next_page_token": "",
        },
    ]
    provider = mock.Mock(spec=ApiClientProvider)
    provider.get.return_value = api

    store, created = MemoryStoreClient(provider).ensure("wanted")

    assert store["name"] == "memory-stores/mem-id-123"
    assert created is False
    assert [call.kwargs for call in api.list_memory_stores.call_args_list] == [
        {"page_size": 100, "page_token": None},
        {"page_size": 100, "page_token": "next"},
    ]


def test_memory_store_client_reports_permission_hint():
    api = mock.Mock()
    api.create_memory_store.side_effect = AgentCliError("denied", error_code="PERMISSION_DENIED")
    provider = mock.Mock(spec=ApiClientProvider)
    provider.get.return_value = api

    with pytest.raises(AgentCliError, match="permission to create memory store 'mem'") as exc:
        MemoryStoreClient(provider).ensure("mem")

    assert exc.value.hint is not None and "workspace admin" in exc.value.hint


def test_memory_store_client_distinguishes_inaccessible_existing_store():
    api = mock.Mock()
    api.create_memory_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    api.list_memory_stores.return_value = {"managed_memory_stores": [], "next_page_token": ""}
    provider = mock.Mock(spec=ApiClientProvider)
    provider.get.return_value = api

    with pytest.raises(AgentCliError, match="already exists but you don't have access"):
        MemoryStoreClient(provider).ensure("mem")


def test_session_store_client_reuses_existing_store():
    api = mock.Mock()
    api.create_session_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    api.get_session_store.return_value = {"name": "sessions"}
    provider = mock.Mock(spec=ApiClientProvider)
    provider.get.return_value = api

    store, created = SessionStoreClient(provider).ensure("sessions")

    assert store == {"name": "sessions"}
    assert created is False
    api.create_session_store.assert_called_once_with("sessions", retry_transient=True)


def test_session_store_client_reports_permission_hint():
    api = mock.Mock()
    api.create_session_store.side_effect = AgentCliError("denied", error_code="PERMISSION_DENIED")
    provider = mock.Mock(spec=ApiClientProvider)
    provider.get.return_value = api

    with pytest.raises(AgentCliError, match="permission to create session store 'sessions'"):
        SessionStoreClient(provider).ensure("sessions")


def test_deploy_help_exposes_instance_count_and_routing_key():
    result = CliRunner().invoke(deploy_mod.deploy, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--instances" in result.output
    assert "--min-instances" not in result.output
    assert "--max-instances" not in result.output
    assert "sticky routing" in result.output
    assert "X-Routing-Key" in result.output


def test_deploy_rejects_instance_count_above_platform_limit():
    result = CliRunner().invoke(deploy_mod.deploy, ["demo", "--instances", "6"], obj=_Ctx())

    assert result.exit_code != 0
    assert "6 is not in the range 1<=x<=5" in result.output


def test_deploy_cli_passes_request_to_service_and_presents_json(tmp_path: pathlib.Path):
    service = mock.Mock()
    service.deploy.return_value = _result()
    ctx = _Ctx(output="json")

    with mock.patch.object(deploy_mod, "build_deploy_service", return_value=service):
        result = CliRunner().invoke(
            deploy_mod.deploy,
            [
                "demo",
                "--source",
                str(tmp_path),
                "--pip-index-url",
                "https://pypi.org/simple/",
                "--workspace-path",
                "/Workspace/agents/demo",
                "--instances",
                "2",
                "--allow-user-scope-update",
            ],
            obj=ctx,
        )

    assert result.exit_code == 0, result.output
    request = service.deploy.call_args.args[0]
    assert request == DeployRequest(
        name="demo",
        source=str(tmp_path),
        pip_index_url="https://pypi.org/simple/",
        workspace_path="/Workspace/agents/demo",
        instance_count=2,
        allow_user_scope_update=True,
    )
    assert json.loads(result.output)["deployment"] == "agent-bricks-demo"


def test_deploy_json_preserves_additive_tool_access_contract(tmp_path: pathlib.Path):
    service = mock.Mock()
    service.deploy.return_value = replace(
        _result(),
        tool_access=ToolAccessSummary(app_resources=1, uc_grants=2, workspace_grants=3),
    )

    with mock.patch.object(deploy_mod, "build_deploy_service", return_value=service):
        result = CliRunner().invoke(
            deploy_mod.deploy,
            ["demo", "--source", str(tmp_path)],
            obj=_Ctx(output="json"),
        )

    assert result.exit_code == 0, result.output
    tool_access = json.loads(result.output)["tool_access"]
    assert tool_access == {
        "app_resources": 1,
        "uc_grants": 2,
        "workspace_grants": 3,
        "direct_resources_only": True,
        "uc_workspace_grants_additive": True,
    }
    assert "direct_grants_additive" not in tool_access


def test_build_deploy_service_composes_current_clients_without_opening_api_client():
    ctx = _Ctx()

    service = deploy_mod.build_deploy_service(ctx)

    assert service.__class__.__name__ == "DeployService"
    assert service._apps_client.__class__ is AppsClient
    assert service._api_client_provider is ctx.api_client_provider
    assert service._memory_store_provisioner.__class__.__name__ == "MemoryStoreProvisioner"
    assert service._session_store_provisioner.__class__.__name__ == "SessionStoreProvisioner"
    assert service._tracing_provisioner.__class__.__name__ == "TracingProvisioner"


def test_apps_client_formats_create_and_fixed_scale_update():
    calls = []

    def runner(args, profile, **kwargs):
        calls.append((args, profile, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="created", stderr="")

    client = AppsClient("prof", runner=runner)
    assert client.create("agent-bricks-demo", 2) == "created"
    assert client.create_update_instances("agent-bricks-demo", 3) == "created"

    assert calls[0][0] == [
        "apps",
        "create",
        "agent-bricks-demo",
        "--compute-min-instances",
        "2",
        "--compute-max-instances",
        "2",
    ]
    update = json.loads(calls[1][0][-1])
    assert update == {
        "app": {"compute_min_instances": 3, "compute_max_instances": 3},
        "update_mask": "compute_min_instances,compute_max_instances",
    }


def test_apps_client_wait_for_active_returns_when_compute_is_active():
    calls = []

    def runner(args, profile, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"compute_status": {"state": "ACTIVE"}}),
            stderr="",
        )

    AppsClient("prof", runner=runner).wait_for_active("agent-bricks-demo", timeout_s=1)

    assert calls == [["apps", "get", "agent-bricks-demo", "-o", "json"]]


def test_apps_client_wait_for_active_times_out(monkeypatch):
    client = AppsClient(
        "prof",
        runner=lambda *args, **kwargs: types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"compute_status": {"state": "STARTING"}}),
            stderr="",
        ),
    )
    monkeypatch.setattr("databricks_agentbricks.clients.apps_client.time.monotonic", lambda: 1)
    monkeypatch.setattr("databricks_agentbricks.clients.apps_client.time.sleep", lambda _: None)

    with pytest.raises(AgentCliError, match="did not reach a running state"):
        client.wait_for_active("agent-bricks-demo", timeout_s=0)


def test_tracing_config_renders_workspace_environment():
    assert mlflow_tracing_config("exp-9").env() == {
        TRACES_TRACKING_URI_ENV: "databricks",
        TRACES_EXPERIMENT_ID_ENV: "exp-9",
    }


def test_tracing_client_reconciles_bound_experiment_resource():
    updates = []

    def runner(args, profile, **kwargs):
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(
                returncode=0, stdout=json.dumps({"resources": []}), stderr=""
            )
        updates.append(json.loads(args[-1]))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    apps = AppsClient("prof", runner=runner)
    client = TracingClient(mock.Mock(spec=ApiClientProvider), apps, "prof")
    assert client.reconcile_app_resources("agent-bricks-demo", "exp-1", []) is None

    assert updates == [
        {
            "app": {
                "resources": [
                    {
                        "name": "agentbricks-trace-experiment",
                        "experiment": {"experiment_id": "exp-1", "permission": "CAN_EDIT"},
                    }
                ]
            },
            "update_mask": "resources",
        }
    ]


def test_project_resolver_reads_bindings_and_missing_project(tmp_path: pathlib.Path):
    project = AgentProject.create(
        tmp_path,
        framework="langgraph",
        server="agentbricks",
        memory_store="memory",
        session_store="sessions",
        experiment_name="/Shared/traces",
    )
    project.write()
    resolver = ProjectResolver()

    assert resolver.resource_bindings(tmp_path) == ("memory", "sessions", "/Shared/traces")
    missing = tmp_path / "missing"
    missing.mkdir()
    assert resolver.resource_bindings(missing) == (None, None, None)


def test_project_resolver_requires_a_name_when_manifest_has_none(tmp_path: pathlib.Path):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.write()

    with pytest.raises(AgentCliError, match="No deployment name given"):
        ProjectResolver().resolve_deployment_name(project, None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("demo", "demo"), ("agent-bricks-demo", "agent-bricks-demo")],
)
def test_deployment_name_validates_at_service_boundary(raw: str, expected: str):
    assert DeploymentName(raw) == expected


def test_real_cli_deploy_reconciles_resources_and_rolls_out_in_order(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = AgentProject.create(
        tmp_path,
        framework="langgraph",
        server="agentbricks",
        memory_store="memory",
        session_store="sessions",
        experiment_name="/Shared/traces",
    )
    project.write()
    events: list[tuple] = []
    api = _DeployApi(events, fail_memory_grant=True)
    runner = _DeployRunner(events)
    monkeypatch.setattr(deploy_mod, "_databricks", runner)

    def resolve_trace(profile, client, name):
        events.append(("trace.ensure", profile, name))
        return ResolvedTraceExperiment("exp-42", MLflowTraceTables())

    monkeypatch.setattr(
        "databricks_agentbricks.clients.tracing_client.create_experiment_idempotent",
        resolve_trace,
    )

    result = CliRunner().invoke(
        deploy_mod.deploy,
        [
            "demo",
            "--source",
            str(tmp_path),
            "--instances",
            "2",
            "--pip-index-url",
            "https://packages.example/simple",
        ],
        obj=_real_deploy_context(api),
    )

    assert result.exit_code == 0, result.output
    assert "Deployed agent 'agent-bricks-demo'" in result.output
    assert "memory grant denied" in result.output

    env = {
        entry["name"]: entry["value"]
        for entry in AppManifest.parse_lenient((tmp_path / "app.yaml").read_text()).raw_env()
    }
    assert env == {
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_EXPERIMENT_ID": "exp-42",
        "AGENT_MEMORY_STORE": "memory-id",
        "AGENT_SESSION_STORE": "sessions",
        "PIP_INDEX_URL": "https://packages.example/simple",
        "UV_INDEX_URL": "https://packages.example/simple",
        "UV_DEFAULT_INDEX": "https://packages.example/simple",
        "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LAKEBASE_BRANCH": "projects/runtime/branches/production",
        "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_DATABASE": "runtime-db",
        "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_USERNAME": "sp-123",
        "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_SCHEMA": "databricks_agentkit_runtime",
    }

    app_calls = [list(event[1]) for event in events if event[0] == "apps"]
    assert next(call for call in app_calls if call[:2] == ["apps", "create"]) == [
        "apps",
        "create",
        "agent-bricks-demo",
        "--compute-min-instances",
        "2",
        "--compute-max-instances",
        "2",
    ]
    assert next(call for call in app_calls if call[0] == "sync") == [
        "sync",
        str(tmp_path),
        "/Workspace/Users/me@example.com/agentbricks_deployments/agent-bricks-demo",
        "--exclude",
        "uv.lock",
    ]
    assert next(call for call in app_calls if call[:2] == ["apps", "deploy"]) == [
        "apps",
        "deploy",
        "agent-bricks-demo",
        "--source-code-path",
        "/Workspace/Users/me@example.com/agentbricks_deployments/agent-bricks-demo",
    ]

    def event_index(predicate):
        return next(index for index, event in enumerate(events) if predicate(event))

    create_index = event_index(
        lambda event: event
        == ("runtime.create", "agent-bricks-demo", "sp-123", "agent-bricks-demo", True)
    )
    sync_index = event_index(lambda event: event[0] == "apps" and event[1][0] == "sync")
    deploy_index = event_index(
        lambda event: event[0] == "apps" and event[1][:2] == ("apps", "deploy")
    )
    trace_update_index = event_index(
        lambda event: event[0] == "apps" and event[1][:2] == ("apps", "create-update")
    )
    memory_grant_index = event_index(lambda event: event[0] == "memory.grant")
    assert create_index < sync_index < deploy_index < trace_update_index
    assert deploy_index < memory_grant_index
    assert ("memory.create", "memory", True) in events
    assert ("session.create", "sessions", True) in events
    assert ("trace.ensure", "prof", "/Shared/traces") in events

    trace_payload = json.loads(
        next(call[-1] for call in app_calls if call[:2] == ["apps", "create-update"])
    )
    assert trace_payload["app"]["resources"][0]["experiment"] == {
        "experiment_id": "exp-42",
        "permission": "CAN_EDIT",
    }


@pytest.mark.parametrize(
    ("pip_index", "expected_index"),
    [(None, _DEFAULT_PIP_INDEX_URL), ("", None)],
    ids=["default-pypi", "empty-disables-override"],
)
def test_real_cli_deploy_custom_server_persists_name_and_skips_runtime_store(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    pip_index: str | None,
    expected_index: str | None,
):
    project = AgentProject.create(tmp_path, framework="langgraph", server="custom")
    project.write()
    events: list[tuple] = []
    api = _DeployApi(events)
    runner = _DeployRunner(events)
    monkeypatch.setattr(deploy_mod, "_databricks", runner)
    args = ["demo", "--source", str(tmp_path)]
    if pip_index is not None:
        args.extend(["--pip-index-url", pip_index])

    result = CliRunner().invoke(deploy_mod.deploy, args, obj=_real_deploy_context(api))

    assert result.exit_code == 0, result.output
    assert AgentProject.load(tmp_path).deployment_name == "demo"
    assert not any(
        event[0] in {"memory.create", "session.create", "runtime.create", "trace.ensure"}
        for event in events
    )
    if expected_index is None:
        assert not (tmp_path / "app.yaml").exists()
    else:
        env = {
            entry["name"]: entry["value"]
            for entry in AppManifest.parse_lenient((tmp_path / "app.yaml").read_text()).raw_env()
        }
        assert env["PIP_INDEX_URL"] == expected_index
        assert env["UV_INDEX_URL"] == expected_index
        assert env["UV_DEFAULT_INDEX"] == expected_index
    assert all(
        not key.startswith("DATABRICKS_AGENTBRICKS_RUNTIME_STORE_")
        for key in (
            {
                entry["name"]: entry["value"]
                for entry in AppManifest.parse_lenient(
                    (tmp_path / "app.yaml").read_text()
                ).raw_env()
            }
            if (tmp_path / "app.yaml").exists()
            else {}
        )
    )


def test_real_cli_deploy_without_agent_manifest_still_rolls_out_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "app.yaml").write_text("command: [python, app.py]\n")
    events: list[tuple] = []
    api = _DeployApi(events)
    runner = _DeployRunner(events)
    monkeypatch.setattr(deploy_mod, "_databricks", runner)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["standalone", "--source", str(tmp_path), "--pip-index-url", ""],
        obj=_real_deploy_context(api),
    )

    assert result.exit_code == 0, result.output
    assert "Deployed agent 'agent-bricks-standalone'" in result.output
    assert not (tmp_path / "agent.toml").exists()
    assert not any(
        event[0] in {"memory.create", "session.create", "runtime.create", "trace.ensure"}
        for event in events
    )
    app_calls = [list(event[1]) for event in events if event[0] == "apps"]
    assert any(call[:2] == ["apps", "create"] for call in app_calls)
    assert any(call[0] == "sync" for call in app_calls)
    assert any(call[:2] == ["apps", "deploy"] for call in app_calls)


def test_deployments_get_uses_real_client_and_presents_json_and_text(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[tuple] = []
    runner = _DeployRunner(events, exists=True)
    monkeypatch.setattr(deploy_mod, "_databricks", runner)
    api = _DeployApi(events)

    json_result = CliRunner().invoke(
        deploy_mod.deployments_get,
        ["agent-bricks-demo"],
        obj=_real_deploy_context(api, output="json"),
    )
    text_result = CliRunner().invoke(
        deploy_mod.deployments_get,
        ["agent-bricks-demo"],
        obj=_real_deploy_context(api, output="text"),
    )

    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output)["url"] == "https://agent.example"
    assert text_result.exit_code == 0, text_result.output
    assert "Agent Deployment" in text_result.output
    assert "https://agent.example" in text_result.output
    assert sum(event[0] == "apps" and event[1][:2] == ("apps", "get") for event in events) >= 2


def test_real_cli_unbind_prunes_trace_manifest_and_app_resources(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = AgentProject.create(tmp_path, framework="langgraph", server="custom")
    project.write()
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["python", "app.py"],
                "env": [
                    {"name": "KEEP", "value": "yes"},
                    {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
                    {"name": "MLFLOW_EXPERIMENT_ID", "value": "stale-exp"},
                ],
            }
        )
    )
    events: list[tuple] = []
    api = _DeployApi(events)
    runner = _DeployRunner(events, exists=True)
    runner.resources = [
        {"name": "user-owned", "secret": {}},
        {"name": "agentbricks-trace-experiment", "experiment": {"experiment_id": "stale-exp"}},
        {"name": "agentbricks-trace-spans", "uc_securable": {"permission": "MODIFY"}},
    ]
    monkeypatch.setattr(deploy_mod, "_databricks", runner)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(tmp_path), "--pip-index-url", ""],
        obj=_real_deploy_context(api),
    )

    assert result.exit_code == 0, result.output
    env = {
        entry["name"]: entry["value"]
        for entry in AppManifest.parse_lenient((tmp_path / "app.yaml").read_text()).raw_env()
    }
    assert env == {"KEEP": "yes"}
    assert runner.resources == [{"name": "user-owned", "secret": {}}]
    trace_update = next(
        list(event[1])
        for event in events
        if event[0] == "apps"
        and event[1][:2] == ("apps", "create-update")
        and "resources" in json.loads(event[1][-1]).get("app", {})
    )
    assert json.loads(trace_update[-1]) == {
        "app": {"resources": [{"name": "user-owned", "secret": {}}]},
        "update_mask": "resources",
    }


def test_real_cli_redeploy_scales_existing_app_without_create(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = AgentProject.create(tmp_path, framework="langgraph", server="custom")
    project.set_deployment_name("demo")
    project.write()
    events: list[tuple] = []
    api = _DeployApi(events)
    runner = _DeployRunner(events, exists=True)
    monkeypatch.setattr(deploy_mod, "_databricks", runner)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["--source", str(tmp_path), "--instances", "3", "--pip-index-url", ""],
        obj=_real_deploy_context(api),
    )

    assert result.exit_code == 0, result.output
    app_calls = [list(event[1]) for event in events if event[0] == "apps"]
    assert not any(call[:2] == ["apps", "create"] for call in app_calls)
    scale_call = next(call for call in app_calls if call[:2] == ["apps", "create-update"])
    assert json.loads(scale_call[-1]) == {
        "app": {"compute_min_instances": 3, "compute_max_instances": 3},
        "update_mask": "compute_min_instances,compute_max_instances",
    }
    assert any(call[0] == "sync" for call in app_calls)
    assert any(call[:2] == ["apps", "deploy"] for call in app_calls)


def test_real_cli_missing_name_fails_before_external_mutation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = AgentProject.create(tmp_path, framework="langgraph", server="custom")
    project.write()
    events: list[tuple] = []
    api = _DeployApi(events)
    runner = _DeployRunner(events)
    monkeypatch.setattr(deploy_mod, "_databricks", runner)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["--source", str(tmp_path)],
        obj=_real_deploy_context(api),
    )

    assert result.exit_code != 0
    assert "No deployment name given" in result.output
    assert events == []
