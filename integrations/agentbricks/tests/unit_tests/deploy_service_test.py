"""Unit coverage for the framework-agnostic deployment service."""

from __future__ import annotations

import pathlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.app_auth_client import AppAuthClient, AppAuthResult
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.deployment.names import DeploymentName
from databricks_agentbricks.deployment.provisioners import (
    AppProvisioner,
    GrantOutcome,
    ManifestPatch,
    MemoryStoreProvisioner,
    MemoryStoreState,
    RuntimeStoreProvisioner,
    RuntimeStoreState,
    SessionStoreProvisioner,
    SessionStoreState,
    TracingProvisioner,
    TracingState,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject, ToolSpec
from databricks_agentbricks.projects.app_manifest import AppManifest
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deploy_service import DeployRequest, DeployService
from databricks_agentbricks.trace_tables import TraceTable, TraceTableKind


@pytest.fixture
def deployment_fixture(tmp_path: pathlib.Path) -> SimpleNamespace:
    """Build a completely local service with typed phase results and no workspace calls."""
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    resolver = Mock(spec=ProjectResolver)
    resolver.load.return_value = project
    resolver.resolve_deployment_name.return_value = "demo"
    resolver.resource_bindings.return_value = (None, None, None)

    auth = Mock(spec=AppAuthClient)
    auth.ensure_user_auth.return_value = AppAuthResult(
        required=False, app_reconciled=False, app_existed=None
    )
    apps = Mock(spec=AppsClient)
    apps.exists.return_value = False
    apps.get_app_url.return_value = "https://demo.example"
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = SimpleNamespace(
        host="https://workspace.example", current_user="james@example.com"
    )

    memory = Mock(spec=MemoryStoreProvisioner)
    memory.reconcile.return_value = MemoryStoreState(None, ManifestPatch({}))
    memory.grant.return_value = GrantOutcome.skipped()
    session = Mock(spec=SessionStoreProvisioner)
    session.reconcile.return_value = SessionStoreState(None, ManifestPatch({}))
    session.grant.return_value = GrantOutcome.skipped()
    tracing = Mock(spec=TracingProvisioner)
    tracing.reconcile.return_value = TracingState(None, (), None, ManifestPatch({}))
    tracing.grant.return_value = GrantOutcome.skipped()
    runtime = Mock(spec=RuntimeStoreProvisioner)
    runtime.reconcile.return_value = RuntimeStoreState(False, None, ManifestPatch({}))
    runtime.after_app_ready.return_value = ManifestPatch({})

    app = Mock(spec=AppProvisioner)
    app.resolve_workspace_path.return_value = "/Workspace/demo"
    reporter = Mock(spec=Reporter)
    reporter.status.return_value = nullcontext()
    reporter.progress.return_value = nullcontext()
    service = DeployService(
        project_resolver=resolver,
        apps_client=apps,
        api_client_provider=provider,
        app_provisioner=app,
        app_auth_client=auth,
        memory_store_provisioner=memory,
        session_store_provisioner=session,
        tracing_provisioner=tracing,
        runtime_store_provisioner=runtime,
        reporter=reporter,
    )
    return SimpleNamespace(
        project=project,
        resolver=resolver,
        auth=auth,
        apps=apps,
        provider=provider,
        app=app,
        memory=memory,
        session=session,
        tracing=tracing,
        runtime=runtime,
        reporter=reporter,
        service=service,
    )


def _request(
    source: pathlib.Path,
    *,
    name: str | None = "demo",
    pip_index_url: str | None = None,
    workspace_path: str | None = None,
    instance_count: int | None = None,
    allow_user_scope_update: bool = False,
) -> DeployRequest:
    return DeployRequest(
        name=name,
        source=str(source),
        pip_index_url=pip_index_url,
        workspace_path=workspace_path,
        instance_count=instance_count,
        allow_user_scope_update=allow_user_scope_update,
    )


def test_deploy_passes_returned_state_between_phases(tmp_path: pathlib.Path) -> None:
    """The service owns phase order and state; collaborators need no live workspace client."""
    project = AgentProject.create(
        tmp_path,
        framework="langgraph",
        server="agentbricks",
        memory_store="memory",
        session_store="session",
        experiment_name="/Shared/trace",
    )
    resolver = Mock(spec=ProjectResolver)
    resolver.load.return_value = project
    resolver.resolve_deployment_name.return_value = "demo"
    resolver.resource_bindings.return_value = ("memory", "session", "/Shared/trace")

    auth = Mock(spec=AppAuthClient)
    auth.ensure_user_auth.return_value = AppAuthResult(
        required=False, app_reconciled=False, app_existed=None
    )
    apps = Mock(spec=AppsClient)
    apps.exists.return_value = False
    apps.get_app_url.return_value = "https://demo.example"
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = SimpleNamespace(host="https://workspace.example")

    memory_state = MemoryStoreState("memory", ManifestPatch({"AGENT_MEMORY_STORE": "mem-id"}))
    session_state = SessionStoreState("session", ManifestPatch({"AGENT_SESSION_STORE": "session"}))
    tracing_state = TracingState("42", (), None, ManifestPatch({"MLFLOW_EXPERIMENT_ID": "42"}))
    runtime_state = RuntimeStoreState(True, None, ManifestPatch({}))
    memory = Mock(spec=MemoryStoreProvisioner)
    memory.reconcile.return_value = memory_state
    memory.grant.return_value = GrantOutcome(attempted=True, error=None)
    session = Mock(spec=SessionStoreProvisioner)
    session.reconcile.return_value = session_state
    session.grant.return_value = GrantOutcome(attempted=True, error="grant denied")
    tracing = Mock(spec=TracingProvisioner)
    tracing.reconcile.return_value = tracing_state
    tracing.grant.return_value = GrantOutcome(attempted=True, error=None)
    runtime = Mock(spec=RuntimeStoreProvisioner)
    runtime.reconcile.return_value = runtime_state
    runtime.after_app_ready.return_value = ManifestPatch({"RUNTIME_STORE_DATABASE": "db-id"})

    app = Mock(spec=AppProvisioner)
    app.resolve_workspace_path.return_value = "/Workspace/demo"

    def check_manifest_before_app_ready(*args: object, **kwargs: object) -> None:
        manifest = AppManifest.parse_lenient((tmp_path / "app.yaml").read_text())
        env = {entry["name"]: entry["value"] for entry in manifest.raw_env()}
        assert env == {
            "MLFLOW_EXPERIMENT_ID": "42",
            "AGENT_MEMORY_STORE": "mem-id",
            "AGENT_SESSION_STORE": "session",
        }

    app.ensure_app_ready.side_effect = check_manifest_before_app_ready
    phases = Mock()
    for name, method in (
        ("memory_reconcile", memory.reconcile),
        ("session_reconcile", session.reconcile),
        ("tracing_reconcile", tracing.reconcile),
        ("runtime_reconcile", runtime.reconcile),
        ("app_ready", app.ensure_app_ready),
        ("runtime_after_app", runtime.after_app_ready),
        ("workspace_path", app.resolve_workspace_path),
        ("deploy_source", app.deploy_source),
        ("memory_grant", memory.grant),
        ("session_grant", session.grant),
        ("tracing_grant", tracing.grant),
    ):
        phases.attach_mock(method, name)

    service = DeployService(
        project_resolver=resolver,
        apps_client=apps,
        api_client_provider=provider,
        app_provisioner=app,
        app_auth_client=auth,
        memory_store_provisioner=memory,
        session_store_provisioner=session,
        tracing_provisioner=tracing,
        runtime_store_provisioner=runtime,
        reporter=Mock(spec=Reporter),
    )
    result = service.deploy(
        DeployRequest(
            name="demo",
            source=str(tmp_path),
            pip_index_url=None,
            workspace_path=None,
            instance_count=None,
            allow_user_scope_update=False,
        )
    )

    assert [call[0] for call in phases.mock_calls] == [
        "memory_reconcile",
        "session_reconcile",
        "tracing_reconcile",
        "runtime_reconcile",
        "app_ready",
        "runtime_after_app",
        "workspace_path",
        "deploy_source",
        "memory_grant",
        "session_grant",
        "tracing_grant",
    ]
    assert runtime.after_app_ready.call_args.args[1] is runtime_state
    assert memory.grant.call_args.args[1] is memory_state
    assert session.grant.call_args.args[1] is session_state
    assert tracing.grant.call_args.args[1] is tracing_state
    provider.get.assert_called_once_with()
    assert result.deployment == "agent-bricks-demo"
    assert result.workspace_path == "/Workspace/demo"
    assert result.env["RUNTIME_STORE_DATABASE"] == "db-id"
    assert result.session_grant_error == "grant denied"
    assert result.created_app_yaml is True
    assert result.uses_runtime_api is True


def test_preflight_failure_keeps_lazy_client_closed(deployment_fixture, tmp_path: pathlib.Path):
    deployment_fixture.resolver.load.side_effect = AgentCliError("bad agent manifest")

    with pytest.raises(AgentCliError, match="bad agent manifest"):
        deployment_fixture.service.deploy(_request(tmp_path))

    deployment_fixture.auth.ensure_user_auth.assert_not_called()
    deployment_fixture.provider.get.assert_not_called()


def test_auth_failure_keeps_lazy_client_closed(deployment_fixture, tmp_path: pathlib.Path):
    deployment_fixture.auth.ensure_user_auth.side_effect = AgentCliError("auth preflight failed")

    with pytest.raises(AgentCliError, match="auth preflight failed"):
        deployment_fixture.service.deploy(_request(tmp_path))

    deployment_fixture.provider.get.assert_not_called()


def test_resolve_prefixes_name_and_persists_it_before_auth(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.resolver.resolve_deployment_name.return_value = "first-run"
    prepared = deployment_fixture.service._prepare_deployment(
        tmp_path,
        _request(tmp_path, name="first-run", instance_count=2, allow_user_scope_update=True),
    )

    assert prepared.name == DeploymentName("agent-bricks-first-run")
    assert prepared.deployment_exists is None
    assert deployment_fixture.project.deployment_name == "first-run"
    assert 'deployment_name = "first-run"' in (tmp_path / "agent.toml").read_text()
    deployment_fixture.auth.ensure_user_auth.assert_called_once_with(
        DeploymentName("agent-bricks-first-run"),
        deployment_fixture.project,
        allow_existing_app_update=True,
        instance_count=2,
    )


def test_invalid_resolved_name_fails_before_auth_or_client(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.resolver.resolve_deployment_name.return_value = "bad/name"

    with pytest.raises(AgentCliError, match="Invalid deployment name"):
        deployment_fixture.service.deploy(_request(tmp_path, name=None))

    deployment_fixture.auth.ensure_user_auth.assert_not_called()
    deployment_fixture.provider.get.assert_not_called()


def test_preflight_managed_tool_failure_is_before_workspace_client(
    deployment_fixture, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    deployment_fixture.project.add_tool(
        ToolSpec.mcp("search", service="system.ai.web_search", auth="app")
    )
    require_support = Mock(side_effect=AgentCliError("unsupported managed tool"))
    monkeypatch.setattr(
        "databricks_agentbricks.services.deploy_service.require_managed_tool_support",
        require_support,
    )

    with pytest.raises(AgentCliError, match="unsupported managed tool"):
        deployment_fixture.service.deploy(_request(tmp_path))

    require_support.assert_called_once_with(tmp_path)
    deployment_fixture.auth.ensure_user_auth.assert_not_called()
    deployment_fixture.provider.get.assert_not_called()


def test_omitted_name_reuses_recorded_existing_app(deployment_fixture, tmp_path: pathlib.Path):
    deployment_fixture.project.set_deployment_name("demo")
    deployment_fixture.apps.exists.return_value = True

    prepared = deployment_fixture.service._prepare_deployment(
        tmp_path, _request(tmp_path, name=None)
    )

    assert prepared.name == DeploymentName("agent-bricks-demo")
    assert prepared.deployment_exists is True
    deployment_fixture.apps.exists.assert_called_once_with(DeploymentName("agent-bricks-demo"))


def test_required_auth_reports_note_and_uses_auth_app_state(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.auth.ensure_user_auth.return_value = AppAuthResult(
        required=True, app_reconciled=True, app_existed=True
    )

    prepared = deployment_fixture.service._prepare_deployment(
        tmp_path, _request(tmp_path, allow_user_scope_update=True, instance_count=3)
    )

    assert prepared.app_reconciled_by_auth is True
    assert prepared.deployment_exists is True
    deployment_fixture.reporter.note.assert_called_once()
    deployment_fixture.apps.exists.assert_not_called()


@pytest.mark.parametrize(
    ("auth_result", "app_exists", "expected_exists", "expected_reconciled", "instances"),
    [
        (AppAuthResult(False, False, None), False, False, False, 2),
        (AppAuthResult(False, False, None), True, True, False, None),
        (AppAuthResult(True, True, True), True, True, True, 4),
    ],
    ids=["create", "redeploy", "auth-reconciled"],
)
def test_deploy_passes_create_redeploy_and_scale_facts_to_app(
    deployment_fixture,
    tmp_path: pathlib.Path,
    auth_result: AppAuthResult,
    app_exists: bool,
    expected_exists: bool,
    expected_reconciled: bool,
    instances: int | None,
):
    deployment_fixture.auth.ensure_user_auth.return_value = auth_result
    deployment_fixture.apps.exists.return_value = app_exists

    result = deployment_fixture.service.deploy(_request(tmp_path, instance_count=instances))

    context = deployment_fixture.app.ensure_app_ready.call_args.args[0]
    assert context.deployment_exists is expected_exists
    deployment_fixture.app.ensure_app_ready.assert_called_once_with(
        context,
        app_reconciled_by_auth=expected_reconciled,
        instance_count=instances,
    )
    deployment_fixture.auth.ensure_user_auth.assert_called_once_with(
        DeploymentName("agent-bricks-demo"),
        deployment_fixture.project,
        allow_existing_app_update=False,
        instance_count=instances,
    )
    if auth_result.app_existed is None:
        deployment_fixture.apps.exists.assert_called_once_with(DeploymentName("agent-bricks-demo"))
    else:
        deployment_fixture.apps.exists.assert_not_called()
    assert result.instance_count == instances


def test_deploy_merges_manifest_env_and_writes_late_runtime_patch(
    deployment_fixture, tmp_path: pathlib.Path
):
    (tmp_path / "app.yaml").write_text(
        """command: [python, app.py]
env:
  - name: KEEP
    value: old
  - name: REMOVE_EARLY
    value: stale
  - name: REMOVE_LATE
    value: stale
"""
    )
    deployment_fixture.resolver.resource_bindings.return_value = ("memory", "session", "trace")
    deployment_fixture.auth.ensure_user_auth.return_value = AppAuthResult(
        required=False, app_reconciled=False, app_existed=True
    )
    deployment_fixture.memory.reconcile.return_value = MemoryStoreState(
        "memory", ManifestPatch({"MEMORY": "memory-id"}, ("REMOVE_EARLY",))
    )
    deployment_fixture.session.reconcile.return_value = SessionStoreState(
        "session", ManifestPatch({"SESSION": "session-name"})
    )
    deployment_fixture.tracing.reconcile.return_value = TracingState(
        "experiment-id",
        (TraceTable(TraceTableKind.SPANS, "catalog.schema.traces"),),
        None,
        ManifestPatch({"TRACE": "trace-id"}),
    )
    deployment_fixture.runtime.reconcile.return_value = RuntimeStoreState(
        True,
        None,
        ManifestPatch({"RUNTIME": "early"}),
    )
    late_patch = ManifestPatch({"RUNTIME_LATE": "late"}, ("REMOVE_LATE",))
    deployment_fixture.runtime.after_app_ready.return_value = late_patch
    phases: list[str] = []

    def check_initial_manifest(*args: object, **kwargs: object) -> None:
        phases.append("app-ready")
        env = {
            entry["name"]: entry["value"]
            for entry in AppManifest.parse_lenient((tmp_path / "app.yaml").read_text()).raw_env()
        }
        assert "RUNTIME_LATE" not in env
        assert env["PIP_INDEX_URL"] == "https://packages.example/simple"
        assert "REMOVE_EARLY" not in env

    deployment_fixture.app.ensure_app_ready.side_effect = check_initial_manifest

    def add_late_patch(*args: object, **kwargs: object) -> ManifestPatch:
        phases.append("runtime-after-app")
        return late_patch

    deployment_fixture.runtime.after_app_ready.side_effect = add_late_patch
    deployment_fixture.app.resolve_workspace_path.side_effect = (
        lambda *args: phases.append("workspace") or "/Workspace/custom"
    )

    result = deployment_fixture.service.deploy(
        _request(
            tmp_path,
            pip_index_url="https://packages.example/simple",
            workspace_path="/Workspace/custom",
            instance_count=3,
        )
    )

    final_env = {
        entry["name"]: entry["value"]
        for entry in AppManifest.parse_lenient((tmp_path / "app.yaml").read_text()).raw_env()
    }
    assert phases == ["app-ready", "runtime-after-app", "workspace"]
    assert final_env == {
        "KEEP": "old",
        "MEMORY": "memory-id",
        "SESSION": "session-name",
        "TRACE": "trace-id",
        "RUNTIME": "early",
        "PIP_INDEX_URL": "https://packages.example/simple",
        "UV_INDEX_URL": "https://packages.example/simple",
        "UV_DEFAULT_INDEX": "https://packages.example/simple",
        "RUNTIME_LATE": "late",
    }
    assert result.env["RUNTIME_LATE"] == "late"
    assert result.workspace_path == "/Workspace/custom"
    assert result.created_app_yaml is False
    deployment_fixture.app.ensure_app_ready.assert_called_once_with(
        deployment_fixture.app.ensure_app_ready.call_args.args[0],
        app_reconciled_by_auth=False,
        instance_count=3,
    )
    deployment_fixture.runtime.after_app_ready.assert_called_once_with(
        deployment_fixture.app.ensure_app_ready.call_args.args[0],
        deployment_fixture.runtime.reconcile.return_value,
    )


def test_deploy_with_no_manifest_changes_does_not_scaffold_app_yaml(
    deployment_fixture, tmp_path: pathlib.Path
):
    result = deployment_fixture.service.deploy(_request(tmp_path))

    assert result.created_app_yaml is False
    assert not (tmp_path / "app.yaml").exists()
    deployment_fixture.provider.get.assert_called_once_with()


def test_deploy_retains_grant_errors_and_attempt_flags(deployment_fixture, tmp_path: pathlib.Path):
    deployment_fixture.resolver.resource_bindings.return_value = ("memory", "session", "trace")
    deployment_fixture.memory.reconcile.return_value = MemoryStoreState("memory", ManifestPatch({}))
    deployment_fixture.session.reconcile.return_value = SessionStoreState(
        "session", ManifestPatch({})
    )
    deployment_fixture.tracing.reconcile.return_value = TracingState(
        "experiment-id", (), "trace setup failed", ManifestPatch({})
    )
    deployment_fixture.memory.grant.return_value = GrantOutcome(True, "memory denied")
    deployment_fixture.session.grant.return_value = GrantOutcome(False, None)
    deployment_fixture.tracing.grant.return_value = GrantOutcome(True, "trace denied")

    result = deployment_fixture.service.deploy(_request(tmp_path))

    assert result.trace_setup_error == "trace setup failed"
    assert result.trace_grant_error == "trace denied"
    assert result.memory_grant_error == "memory denied"
    assert result.session_grant_error is None
    assert result.memory_grant_attempted is True
    assert result.session_grant_attempted is False


def test_deploy_without_agent_project_still_deploys_named_source(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.resolver.load.return_value = None
    deployment_fixture.resolver.resolve_deployment_name.return_value = "standalone"

    result = deployment_fixture.service.deploy(_request(tmp_path, name="standalone"))

    assert result.deployment == "agent-bricks-standalone"
    assert result.uses_runtime_api is False
    deployment_fixture.auth.ensure_user_auth.assert_called_once_with(
        DeploymentName("agent-bricks-standalone"),
        None,
        allow_existing_app_update=False,
        instance_count=None,
    )


def test_list_deployments_filters_non_agent_apps(deployment_fixture):
    agent_one = {"name": "agent-bricks-one", "state": "ACTIVE"}
    agent_two = {"name": "agent-bricks-two", "state": "STOPPED"}
    deployment_fixture.apps.list_all.return_value = [
        agent_one,
        {"name": "unrelated-app"},
        {"name": ""},
        {"name": None},
        {"state": "ACTIVE"},
        agent_two,
    ]

    assert deployment_fixture.service.list_deployments() == [agent_one, agent_two]
    deployment_fixture.apps.list_all.assert_called_once_with()


def test_lifecycle_verbs_delegate_to_apps_client(deployment_fixture):
    name = DeploymentName("agent-bricks-demo")
    payload = {"name": str(name), "url": "https://demo.example"}
    deployment_fixture.apps.get.return_value = payload

    assert deployment_fixture.service.get(name) is payload
    assert deployment_fixture.service.logs(name) is None
    assert deployment_fixture.service.start(name) is None
    assert deployment_fixture.service.stop(name) is None

    deployment_fixture.apps.get.assert_called_once_with(name)
    deployment_fixture.apps.stream_logs.assert_called_once_with(name)
    deployment_fixture.apps.start.assert_called_once_with(name)
    deployment_fixture.apps.stop.assert_called_once_with(name)


@pytest.mark.parametrize("managed", [False, True])
def test_deletes_runtime_store_data_reflects_provisioner(deployment_fixture, managed: bool):
    deployment_fixture.runtime.manages_persistent_data.return_value = managed

    assert deployment_fixture.service.deletes_runtime_store_data() is managed
    deployment_fixture.runtime.manages_persistent_data.assert_called_once_with()


def test_delete_unmanaged_runtime_store_deletes_app_only(deployment_fixture):
    name = DeploymentName("agent-bricks-demo")
    deployment_fixture.runtime.manages_persistent_data.return_value = False

    deployment_fixture.service.delete(name)

    deployment_fixture.runtime.delete_managed.assert_not_called()
    deployment_fixture.reporter.status.assert_not_called()
    deployment_fixture.apps.delete.assert_called_once_with(name)


def test_delete_managed_runtime_store_before_app(deployment_fixture):
    name = DeploymentName("agent-bricks-demo")
    events: list[str] = []
    deployment_fixture.runtime.manages_persistent_data.return_value = True
    deployment_fixture.runtime.delete_managed.side_effect = lambda value: events.append(
        f"runtime:{value}"
    )
    deployment_fixture.apps.delete.side_effect = lambda value: events.append(f"app:{value}")

    deployment_fixture.service.delete(name)

    assert events == [f"runtime:{name}", f"app:{name}"]
    deployment_fixture.reporter.status.assert_called_once_with("Deleting Runtime Store…")
    deployment_fixture.runtime.delete_managed.assert_called_once_with(name)
    deployment_fixture.apps.delete.assert_called_once_with(name)


def test_delete_keeps_app_when_managed_cleanup_fails(deployment_fixture):
    name = DeploymentName("agent-bricks-demo")
    deployment_fixture.runtime.manages_persistent_data.return_value = True
    deployment_fixture.runtime.delete_managed.side_effect = AgentCliError("cleanup failed")

    with pytest.raises(AgentCliError, match="cleanup failed"):
        deployment_fixture.service.delete(name)

    deployment_fixture.runtime.delete_managed.assert_called_once_with(name)
    deployment_fixture.apps.delete.assert_not_called()
