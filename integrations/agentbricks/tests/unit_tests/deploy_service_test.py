"""Unit coverage for the framework-agnostic deployment service."""

from __future__ import annotations

import os
import pathlib
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import requests
from databricks.sdk.errors import (
    BadRequest,
    InternalError,
    NotFound,
    PermissionDenied,
    ResourceDoesNotExist,
    Unauthenticated,
)
from databricks.sdk.service.apps import App, AppDeployment
from delete_test_fixtures import fake_workspace_client

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.apps_user_auth_client import (
    AppAuthPlan,
    AppUserScopeUpdatePlan,
)
from databricks_agentbricks.clients.tracing_client import TraceTable, TraceTableKind
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject, ToolSpec
from databricks_agentbricks.projects.app_manifest import AppManifest
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deploy_service import DeployRequest, DeployService
from databricks_agentbricks.services.deployment import app_provisioner as app_provisioner_mod
from databricks_agentbricks.services.deployment import deploy_state
from databricks_agentbricks.services.deployment.app_provisioner import (
    AppDeleteWaitResult,
    AppProvisioner,
    recorded_source_code_path,
)
from databricks_agentbricks.services.deployment.deploy_state import DeployStateTracker
from databricks_agentbricks.services.deployment.model_binding_provisioner import (
    ModelBindingProvisioner,
)
from databricks_agentbricks.services.deployment.names import DeploymentName, fit_base_name
from databricks_agentbricks.services.deployment.provisioners import (
    GrantOutcome,
    ManifestPatch,
    MemoryStoreProvisioner,
    MemoryStoreState,
    ProjectContext,
    ResourceContext,
    RuntimeStoreProvisioner,
    RuntimeStoreState,
    SessionStoreProvisioner,
    SessionStoreState,
    TracingProvisioner,
    TracingState,
)
from databricks_agentbricks.services.deployment.tool_access import ToolAccessPlan
from databricks_agentbricks.services.deployment.tool_access_provisioner import ToolAccessProvisioner
from databricks_agentkit.runtime.tool_manifest import MEMORY_STORE_ENV, SESSION_STORE_ENV


def _auth_plan(
    *,
    existing_scopes: tuple[str, ...] | None = None,
    scopes: tuple[str, ...] = ("ai-gateway",),
    apps: Mock | None = None,
) -> AppAuthPlan:
    """Build a small request-user plan for service and App provisioner tests."""
    return AppAuthPlan(
        scope_update=AppUserScopeUpdatePlan(
            apps=apps if apps is not None else Mock(),
            name="agent-bricks-demo",
            existing_scopes=existing_scopes,
            scopes=scopes,
        )
    )


def _no_auth_plan() -> AppAuthPlan:
    return AppAuthPlan(scope_update=None)


def _app_context(*, deployment_exists: bool) -> ResourceContext:
    return ResourceContext(
        project=ProjectContext(
            source_dir=pathlib.Path("."),
            name=DeploymentName("agent-bricks-demo"),
            agent_project=None,
        ),
        memory_store=None,
        session_store=None,
        experiment_name=None,
        deployment_exists=deployment_exists,
    )


@pytest.fixture
def deployment_fixture(tmp_path: pathlib.Path) -> SimpleNamespace:
    """Build a completely local service with typed phase results and no workspace calls."""
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    resolver = Mock(spec=ProjectResolver)
    resolver.load.return_value = project
    resolver.resolve_deployment_name.return_value = "demo"
    resolver.resource_bindings.return_value = (None, None, None)

    apps = Mock(spec=AppsClient)
    apps.exists.return_value = False
    apps.get_service_principal.return_value = "sp-123"
    apps.get_app_url.return_value = "https://demo.example"
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = SimpleNamespace(
        host="https://workspace.example",
        current_user="james@example.com",
        workspace_client=object(),
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
    tool_access = Mock(spec=ToolAccessProvisioner)
    tool_access.plan.return_value = ToolAccessPlan()

    app = Mock(spec=AppProvisioner)
    app.prepare_app_auth.return_value = _no_auth_plan()
    app.resolve_workspace_path.return_value = "/Workspace/demo"
    reporter = Mock(spec=Reporter)
    reporter.status.return_value = nullcontext()
    reporter.progress.return_value = nullcontext()
    service = DeployService(
        project_resolver=resolver,
        apps_client=apps,
        api_client_provider=provider,
        app_provisioner=app,
        memory_store_provisioner=memory,
        session_store_provisioner=session,
        tracing_provisioner=tracing,
        runtime_store_provisioner=runtime,
        tool_access_provisioner=tool_access,
        model_binding_provisioner=ModelBindingProvisioner(provider, reporter),
        reporter=reporter,
    )
    return SimpleNamespace(
        project=project,
        resolver=resolver,
        apps=apps,
        provider=provider,
        app=app,
        memory=memory,
        session=session,
        tracing=tracing,
        runtime=runtime,
        tool_access=tool_access,
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

    apps = Mock(spec=AppsClient)
    apps.exists.return_value = False
    apps.get_service_principal.return_value = "sp-123"
    apps.get_app_url.return_value = "https://demo.example"
    provider = Mock(spec=ApiClientProvider)
    workspace_client = object()
    provider.get.return_value = SimpleNamespace(
        host="https://workspace.example", workspace_client=workspace_client
    )

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
    tool_access = Mock(spec=ToolAccessProvisioner)
    tool_plan = ToolAccessPlan(app_resources=({"name": "tool"},))
    tool_access.plan.return_value = tool_plan

    app = Mock(spec=AppProvisioner)
    auth_plan = _no_auth_plan()
    app.prepare_app_auth.return_value = auth_plan
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
        ("tool_access_plan", tool_access.plan),
        ("tool_access_reconcile", tool_access.reconcile_before_rollout),
        ("workspace_path", app.resolve_workspace_path),
        ("deploy_source", app.deploy_source),
        ("tool_access_finalize", tool_access.finalize_after_rollout),
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
        memory_store_provisioner=memory,
        session_store_provisioner=session,
        tracing_provisioner=tracing,
        runtime_store_provisioner=runtime,
        tool_access_provisioner=tool_access,
        model_binding_provisioner=ModelBindingProvisioner(provider, Mock(spec=Reporter)),
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
        "tool_access_plan",
        "tool_access_reconcile",
        "workspace_path",
        "deploy_source",
        "tool_access_finalize",
        "memory_grant",
        "session_grant",
        "tracing_grant",
    ]
    assert runtime.after_app_ready.call_args.args[1] is runtime_state
    tool_access.plan.assert_called_once_with(project.tools)
    assert tool_access.reconcile_before_rollout.call_args.args == (
        DeploymentName("agent-bricks-demo"),
        tool_plan,
        workspace_client,
    )
    tool_access.finalize_after_rollout.assert_called_once_with(
        DeploymentName("agent-bricks-demo"), tool_plan
    )
    assert memory.grant.call_args.args == (memory_state, "sp-123")
    assert session.grant.call_args.args == (session_state, "sp-123")
    assert tracing.grant.call_args.args[1] is tracing_state
    apps.get_service_principal.assert_called_once_with(DeploymentName("agent-bricks-demo"))
    app.prepare_app_auth.assert_called_once_with(
        DeploymentName("agent-bricks-demo"),
        project,
        allow_existing_app_update=False,
    )
    assert app.ensure_app_ready.call_args.kwargs["auth_plan"] is auth_plan
    provider.get.assert_called_once_with()
    assert result.deployment == "agent-bricks-demo"
    assert result.workspace_path == "/Workspace/demo"
    assert result.env["RUNTIME_STORE_DATABASE"] == "db-id"
    assert result.session_grant_error == "grant denied"
    assert result.tool_access is not None and result.tool_access.app_resources == 1
    assert result.created_app_yaml is True
    assert result.uses_runtime_api is True


def test_preflight_failure_keeps_lazy_client_closed(deployment_fixture, tmp_path: pathlib.Path):
    deployment_fixture.resolver.load.side_effect = AgentCliError("bad agent manifest")

    with pytest.raises(AgentCliError, match="bad agent manifest"):
        deployment_fixture.service.deploy(_request(tmp_path))

    deployment_fixture.app.prepare_app_auth.assert_not_called()
    deployment_fixture.provider.get.assert_not_called()


def test_auth_failure_keeps_lazy_client_closed(deployment_fixture, tmp_path: pathlib.Path):
    deployment_fixture.app.prepare_app_auth.side_effect = AgentCliError("auth preflight failed")

    with pytest.raises(AgentCliError, match="auth preflight failed"):
        deployment_fixture.service.deploy(_request(tmp_path))

    deployment_fixture.provider.get.assert_not_called()


def test_resolve_prefixes_name_and_persists_it_before_auth(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.resolver.resolve_deployment_name.return_value = "first-run"
    prepared = deployment_fixture.service.prepare(
        _request(tmp_path, name="first-run", instance_count=2, allow_user_scope_update=True),
    )

    assert prepared.name == DeploymentName("agent-bricks-first-run")
    assert prepared.deployment_exists is None
    assert deployment_fixture.project.deployment_name == "first-run"
    assert 'deployment_name = "first-run"' in (tmp_path / "agent.toml").read_text()
    deployment_fixture.app.prepare_app_auth.assert_called_once_with(
        DeploymentName("agent-bricks-first-run"),
        deployment_fixture.project,
        allow_existing_app_update=True,
    )


def test_invalid_resolved_name_fails_before_auth_or_client(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.resolver.resolve_deployment_name.return_value = "bad/name"

    with pytest.raises(AgentCliError, match="Invalid deployment name"):
        deployment_fixture.service.deploy(_request(tmp_path, name=None))

    deployment_fixture.app.prepare_app_auth.assert_not_called()
    deployment_fixture.provider.get.assert_not_called()


@pytest.mark.parametrize("requested", [None, "caj-langgraph-agent"])
def test_too_long_name_fails_before_auth_or_client_with_a_suggestion(
    deployment_fixture, tmp_path: pathlib.Path, requested: str | None
):
    deployment_fixture.resolver.resolve_deployment_name.return_value = "caj-langgraph-agent"

    with pytest.raises(AgentCliError, match=r"too long \(32 > 30\)") as excinfo:
        deployment_fixture.service.deploy(_request(tmp_path, name=requested))

    assert excinfo.value.hint == (
        "Names can be up to 17 characters (Agent Bricks adds 'agent-bricks-', and app names cap "
        f"at 30). Try: agentbricks deploy {fit_base_name('caj-langgraph-agent')}"
    )
    deployment_fixture.app.prepare_app_auth.assert_not_called()
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
    deployment_fixture.app.prepare_app_auth.assert_not_called()
    deployment_fixture.provider.get.assert_not_called()


def test_omitted_name_reuses_recorded_existing_app(deployment_fixture, tmp_path: pathlib.Path):
    deployment_fixture.project.set_deployment_name("demo")
    deployment_fixture.apps.exists.return_value = True

    prepared = deployment_fixture.service.prepare(_request(tmp_path, name=None))

    assert prepared.name == DeploymentName("agent-bricks-demo")
    assert prepared.deployment_exists is True
    deployment_fixture.apps.exists.assert_called_once_with(DeploymentName("agent-bricks-demo"))


def test_required_auth_reports_note_and_uses_auth_app_state(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.app.prepare_app_auth.return_value = _auth_plan(
        existing_scopes=("ai-gateway",)
    )

    prepared = deployment_fixture.service.prepare(
        _request(tmp_path, allow_user_scope_update=True, instance_count=3)
    )

    assert prepared.auth_plan.required is True
    assert prepared.auth_plan.app_existed is True
    assert prepared.deployment_exists is True
    deployment_fixture.reporter.note.assert_called_once()
    deployment_fixture.apps.exists.assert_not_called()


@pytest.mark.parametrize(
    ("auth_plan", "app_exists", "expected_exists", "instances"),
    [
        (_no_auth_plan(), False, False, 2),
        (_no_auth_plan(), True, True, None),
        (_auth_plan(existing_scopes=("ai-gateway",)), True, True, 4),
    ],
    ids=["create", "redeploy", "auth-reconciled"],
)
def test_deploy_passes_create_redeploy_and_scale_facts_to_app(
    deployment_fixture,
    tmp_path: pathlib.Path,
    auth_plan: AppAuthPlan,
    app_exists: bool,
    expected_exists: bool,
    instances: int | None,
):
    deployment_fixture.app.prepare_app_auth.return_value = auth_plan
    deployment_fixture.apps.exists.return_value = app_exists

    result = deployment_fixture.service.deploy(_request(tmp_path, instance_count=instances))

    context = deployment_fixture.app.ensure_app_ready.call_args.args[0]
    assert context.deployment_exists is expected_exists
    deployment_fixture.app.ensure_app_ready.assert_called_once_with(
        context,
        auth_plan=auth_plan,
        instance_count=instances,
    )
    deployment_fixture.app.prepare_app_auth.assert_called_once_with(
        DeploymentName("agent-bricks-demo"),
        deployment_fixture.project,
        allow_existing_app_update=False,
    )
    if auth_plan.app_existed is None:
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
    deployment_fixture.app.prepare_app_auth.return_value = _no_auth_plan()
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
        auth_plan=deployment_fixture.app.prepare_app_auth.return_value,
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


def test_declarative_unbind_prunes_stale_store_env_from_existing_manifest(
    deployment_fixture, tmp_path: pathlib.Path
) -> None:
    (tmp_path / "app.yaml").write_text(
        "command: ['python', 'app.py']\n"
        "env:\n"
        f"  - name: {MEMORY_STORE_ENV}\n"
        "    value: stale-memory-id\n"
        f"  - name: {SESSION_STORE_ENV}\n"
        "    value: stale-session-name\n"
        "  - name: KEEP\n"
        "    value: preserve-me\n"
    )
    memory_client = Mock()
    session_client = Mock()
    service = DeployService(
        project_resolver=deployment_fixture.resolver,
        apps_client=deployment_fixture.apps,
        api_client_provider=deployment_fixture.provider,
        app_provisioner=deployment_fixture.app,
        memory_store_provisioner=MemoryStoreProvisioner(memory_client, deployment_fixture.reporter),
        session_store_provisioner=SessionStoreProvisioner(
            session_client, deployment_fixture.reporter
        ),
        tracing_provisioner=deployment_fixture.tracing,
        runtime_store_provisioner=deployment_fixture.runtime,
        tool_access_provisioner=deployment_fixture.tool_access,
        model_binding_provisioner=ModelBindingProvisioner(
            deployment_fixture.provider, deployment_fixture.reporter
        ),
        reporter=deployment_fixture.reporter,
    )

    result = service.deploy(_request(tmp_path))

    env = {
        entry["name"]: entry["value"]
        for entry in AppManifest.parse_lenient((tmp_path / "app.yaml").read_text()).raw_env()
    }
    assert env == {"KEEP": "preserve-me"}
    assert result.memory_grant_attempted is False
    assert result.session_grant_attempted is False
    memory_client.assert_not_called()
    session_client.assert_not_called()


def test_deploy_does_not_resolve_service_principal_without_bound_stores(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.service.deploy(_request(tmp_path))

    deployment_fixture.apps.get_service_principal.assert_not_called()
    deployment_fixture.memory.grant.assert_called_once_with(
        deployment_fixture.memory.reconcile.return_value, None
    )
    deployment_fixture.session.grant.assert_called_once_with(
        deployment_fixture.session.reconcile.return_value, None
    )


def test_tool_access_failure_stops_before_source_rollout(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.tool_access.reconcile_before_rollout.side_effect = AgentCliError(
        "tool grant denied"
    )

    with pytest.raises(AgentCliError, match="tool grant denied"):
        deployment_fixture.service.deploy(_request(tmp_path))

    deployment_fixture.app.deploy_source.assert_not_called()
    deployment_fixture.tool_access.finalize_after_rollout.assert_not_called()


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
    deployment_fixture.apps.get_service_principal.assert_called_once_with(
        DeploymentName("agent-bricks-demo")
    )
    deployment_fixture.memory.grant.assert_called_once_with(
        deployment_fixture.memory.reconcile.return_value, "sp-123"
    )
    deployment_fixture.session.grant.assert_called_once_with(
        deployment_fixture.session.reconcile.return_value, "sp-123"
    )


def test_memory_store_grant_uses_reconciled_resource_name_and_shared_sp():
    client = Mock()
    client.reconcile.return_value = SimpleNamespace(
        created=False,
        store_id="memory-id",
        resource_name="memory-stores/memory-id",
    )
    client.grant.return_value = None
    reporter = Mock(spec=Reporter)
    reporter.status.return_value = nullcontext()
    provisioner = MemoryStoreProvisioner(client, reporter)

    state = provisioner.reconcile(
        ResourceContext(
            project=ProjectContext(pathlib.Path("."), DeploymentName("agent-bricks-demo"), None),
            memory_store="friendly-name",
            session_store=None,
            experiment_name=None,
        )
    )
    outcome = provisioner.grant(state, "sp-123")

    assert state.resource_name == "memory-stores/memory-id"
    assert state.manifest.env["AGENT_MEMORY_STORE"] == "memory-id"
    assert outcome == GrantOutcome(attempted=True, error=None)
    client.grant.assert_called_once_with("memory-stores/memory-id", "sp-123")


def _app_provisioner_for_test() -> SimpleNamespace:
    apps = Mock(spec=AppsClient)
    apps.create.return_value = "App compute"
    reporter = Mock(spec=Reporter)
    reporter.progress.return_value = nullcontext()
    return SimpleNamespace(
        apps=apps,
        reporter=reporter,
        provisioner=AppProvisioner(
            apps,
            Mock(spec=ApiClientProvider),
            Mock(),
            reporter,
        ),
    )


@pytest.mark.parametrize(("deployment_exists", "instances"), [(False, None), (True, 2)])
def test_app_provisioner_reports_cli_output_on_stderr(deployment_exists, instances):
    # The App the CLI prints is progress; stdout carries only the `--output json` payload.
    harness = _app_provisioner_for_test()
    harness.apps.create_update_instances.return_value = "App compute"

    harness.provisioner.ensure_app_ready(
        _app_context(deployment_exists=deployment_exists),
        auth_plan=_no_auth_plan(),
        instance_count=instances,
    )

    harness.reporter.note.assert_called_once_with("Agent compute")
    harness.reporter.echo.assert_not_called()


def test_app_provisioner_new_auth_uses_scoped_sdk_create():
    harness = _app_provisioner_for_test()
    sdk_apps = Mock()
    sdk_apps.get.return_value = SimpleNamespace(
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=True,
    )
    auth_plan = _auth_plan(apps=sdk_apps, existing_scopes=None)

    harness.provisioner.ensure_app_ready(
        _app_context(deployment_exists=False), auth_plan=auth_plan, instance_count=2
    )

    sdk_apps.create.assert_called_once()
    created = sdk_apps.create.call_args.args[0]
    assert created.name == "agent-bricks-demo"
    assert created.user_api_scopes == ["ai-gateway"]
    assert created.forward_user_access_token is True
    assert created.compute_min_instances == 2
    assert created.compute_max_instances == 2
    harness.apps.create.assert_not_called()
    harness.apps.create_update_instances.assert_not_called()
    harness.apps.wait_for_active.assert_called_once_with(DeploymentName("agent-bricks-demo"))


def test_app_provisioner_existing_auth_uses_masked_sdk_update():
    harness = _app_provisioner_for_test()
    sdk_apps = Mock()
    existing = SimpleNamespace(
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway", "sql"],
        forward_user_access_token=True,
    )
    converged = SimpleNamespace(
        user_api_scopes=["ai-gateway", "sql"],
        effective_user_api_scopes=["ai-gateway", "sql"],
        forward_user_access_token=True,
    )
    sdk_apps.get.side_effect = [existing, converged]
    auth_plan = _auth_plan(
        apps=sdk_apps,
        existing_scopes=("ai-gateway",),
        scopes=("ai-gateway", "sql"),
    )

    harness.provisioner.ensure_app_ready(
        _app_context(deployment_exists=True), auth_plan=auth_plan, instance_count=3
    )

    sdk_apps.create.assert_not_called()
    sdk_apps.create_update.assert_called_once()
    assert sdk_apps.create_update.call_args.args == ("agent-bricks-demo",)
    assert sdk_apps.create_update.call_args.kwargs["update_mask"] == (
        "user_api_scopes,compute_min_instances,compute_max_instances"
    )
    update = sdk_apps.create_update.call_args.kwargs["app"]
    assert update.user_api_scopes == ["ai-gateway", "sql"]
    assert update.forward_user_access_token is None
    harness.apps.create.assert_not_called()
    harness.apps.create_update_instances.assert_not_called()
    harness.apps.wait_for_active.assert_called_once_with(DeploymentName("agent-bricks-demo"))


@pytest.mark.parametrize(
    ("deployment_exists", "instance_count", "method"),
    [(False, 2, "create"), (True, 3, "create_update_instances")],
)
def test_app_provisioner_no_auth_uses_cli_create_or_scale(
    deployment_exists: bool, instance_count: int, method: str
):
    harness = _app_provisioner_for_test()

    harness.provisioner.ensure_app_ready(
        _app_context(deployment_exists=deployment_exists),
        auth_plan=_no_auth_plan(),
        instance_count=instance_count,
    )

    getattr(harness.apps, method).assert_called_once_with(
        DeploymentName("agent-bricks-demo"), instance_count
    )
    other = "create_update_instances" if method == "create" else "create"
    getattr(harness.apps, other).assert_not_called()
    harness.apps.wait_for_active.assert_called_once_with(DeploymentName("agent-bricks-demo"))


def test_deploy_without_agent_project_still_deploys_named_source(
    deployment_fixture, tmp_path: pathlib.Path
):
    deployment_fixture.resolver.load.return_value = None
    deployment_fixture.resolver.resolve_deployment_name.return_value = "standalone"

    result = deployment_fixture.service.deploy(_request(tmp_path, name="standalone"))

    assert result.deployment == "agent-bricks-standalone"
    assert result.uses_runtime_api is False
    assert result.tool_access is None
    deployment_fixture.tool_access.plan.assert_not_called()
    deployment_fixture.tool_access.reconcile_before_rollout.assert_not_called()
    deployment_fixture.app.prepare_app_auth.assert_called_once_with(
        DeploymentName("agent-bricks-standalone"),
        None,
        allow_existing_app_update=False,
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


_DEFAULT_SOURCE = "/Workspace/Users/user@example.com/agentbricks_deployments/agent-bricks-demo"


def _existing_app(deployment_fixture, *, source: str | None = _DEFAULT_SOURCE) -> None:
    deployment = AppDeployment(source_code_path=source) if source else None
    deployment_fixture.app.find.return_value = App(
        name="agent-bricks-demo", active_deployment=deployment, service_principal_client_id="sp-1"
    )
    deployment_fixture.app.delete_source.return_value = True
    deployment_fixture.app.wait_until_deleted.return_value = AppDeleteWaitResult()


def _delete(deployment_fixture, *, delete_source: bool = True, name: str = "agent-bricks-demo"):
    service = deployment_fixture.service
    return service.delete(service.delete_target(DeploymentName(name)), delete_source=delete_source)


def test_delete_unmanaged_runtime_store_deletes_app_only(deployment_fixture):
    name = DeploymentName("agent-bricks-demo")
    _existing_app(deployment_fixture)
    deployment_fixture.runtime.manages_persistent_data.return_value = False

    result = _delete(deployment_fixture)

    deployment_fixture.runtime.delete_managed.assert_not_called()
    deployment_fixture.reporter.status.assert_called_once_with(
        "Waiting for the deployment to be deleted…"
    )
    deployment_fixture.apps.delete.assert_called_once_with(name)
    assert result.still_deleting is False


def test_delete_managed_runtime_store_before_app(deployment_fixture):
    name = DeploymentName("agent-bricks-demo")
    _existing_app(deployment_fixture)
    events: list[str] = []
    deployment_fixture.runtime.manages_persistent_data.return_value = True
    deployment_fixture.runtime.delete_managed.side_effect = lambda value, sp: events.append(
        f"runtime:{value}:{sp}"
    )
    deployment_fixture.apps.delete.side_effect = lambda value: events.append(f"app:{value}")
    deployment_fixture.app.wait_until_deleted.side_effect = lambda value: (
        events.append(f"gone:{value}") or AppDeleteWaitResult()
    )
    deployment_fixture.app.delete_source.side_effect = lambda path: (
        events.append(f"source:{path}") or True
    )

    _delete(deployment_fixture)

    # The folder goes only once the App is gone, so a stalled delete keeps its source.
    assert events == [
        f"runtime:{name}:sp-1",
        f"app:{name}",
        f"gone:{name}",
        f"source:{_DEFAULT_SOURCE}",
    ]
    assert deployment_fixture.reporter.status.call_args_list[0].args == ("Deleting Runtime Store…",)


def test_delete_keeps_app_when_managed_cleanup_fails(deployment_fixture):
    name = DeploymentName("agent-bricks-demo")
    _existing_app(deployment_fixture)
    deployment_fixture.runtime.manages_persistent_data.return_value = True
    deployment_fixture.runtime.delete_managed.side_effect = AgentCliError("cleanup failed")

    with pytest.raises(AgentCliError, match="cleanup failed"):
        _delete(deployment_fixture)

    deployment_fixture.runtime.delete_managed.assert_called_once_with(name, "sp-1")
    deployment_fixture.apps.delete.assert_not_called()


def test_delete_of_a_missing_deployment_deletes_nothing(deployment_fixture):
    deployment_fixture.app.find.return_value = None

    with pytest.raises(AgentCliError, match="does not exist"):
        _delete(deployment_fixture)

    deployment_fixture.runtime.delete_managed.assert_not_called()
    deployment_fixture.apps.delete.assert_not_called()
    deployment_fixture.app.delete_source.assert_not_called()


def test_delete_fails_when_the_app_cannot_be_read(deployment_fixture):
    deployment_fixture.app.find.side_effect = AgentCliError("Could not check whether ...")

    with pytest.raises(AgentCliError, match="Could not check"):
        _delete(deployment_fixture)

    deployment_fixture.apps.delete.assert_not_called()
    deployment_fixture.app.delete_source.assert_not_called()


def test_delete_without_a_recorded_deployment_deletes_no_folder(deployment_fixture):
    _existing_app(deployment_fixture, source=None)
    deployment_fixture.runtime.manages_persistent_data.return_value = False

    result = _delete(deployment_fixture)

    assert result.source_path is None
    assert result.source_folder_status == "not_recorded"
    deployment_fixture.app.delete_source.assert_not_called()
    deployment_fixture.apps.delete.assert_called_once_with("agent-bricks-demo")


def test_delete_keeps_the_source_folder_unless_asked(deployment_fixture):
    _existing_app(deployment_fixture)
    deployment_fixture.runtime.manages_persistent_data.return_value = False

    result = _delete(deployment_fixture, delete_source=False)

    assert result.source_path == _DEFAULT_SOURCE
    assert result.source_folder_status == "kept"
    assert result.confirmed is True
    deployment_fixture.app.delete_source.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [_DEFAULT_SOURCE, "/Workspace/Users/user@example.com/my-agent", "/Workspace/Shared/agent"],
)
def test_delete_source_deletes_whatever_folder_the_app_recorded(deployment_fixture, source):
    # The user asked for it, so the folder needn't be deploy's default to be deleted.
    _existing_app(deployment_fixture, source=source)
    deployment_fixture.runtime.manages_persistent_data.return_value = False

    result = _delete(deployment_fixture)

    assert result.source_folder_status == "deleted"
    deployment_fixture.app.delete_source.assert_called_once_with(source)


def test_delete_reports_a_source_folder_it_could_not_delete(deployment_fixture):
    _existing_app(deployment_fixture)
    deployment_fixture.runtime.manages_persistent_data.return_value = False
    deployment_fixture.app.delete_source.side_effect = AgentCliError("permission denied")

    result = _delete(deployment_fixture)

    # The App is already gone, so the folder's failure is reported, not raised.
    deployment_fixture.apps.delete.assert_called_once_with("agent-bricks-demo")
    assert result.source_folder_status == "failed"
    assert result.source_folder_error == "permission denied"


def test_delete_reports_a_source_folder_that_was_already_gone(deployment_fixture):
    _existing_app(deployment_fixture)
    deployment_fixture.runtime.manages_persistent_data.return_value = False
    deployment_fixture.app.delete_source.return_value = False

    result = _delete(deployment_fixture)

    assert result.source_folder_status == "already_gone"
    assert result.source_folder_error is None


@pytest.mark.parametrize(
    ("wait", "still_deleting"),
    [(AppDeleteWaitResult(timed_out=True), True), (AppDeleteWaitResult(error="expired"), False)],
)
def test_delete_keeps_the_folder_when_the_app_is_not_confirmed_gone(
    deployment_fixture, wait, still_deleting
):
    _existing_app(deployment_fixture)
    deployment_fixture.runtime.manages_persistent_data.return_value = False
    deployment_fixture.app.wait_until_deleted.return_value = wait

    result = _delete(deployment_fixture)

    assert result.still_deleting is still_deleting
    assert result.wait_error == wait.error
    assert result.confirmed is False
    assert result.source_folder_status == "kept_app_not_confirmed"
    deployment_fixture.app.delete_source.assert_not_called()


def test_delete_interrupted_during_the_wait_keeps_the_folder_and_says_so(deployment_fixture):
    _existing_app(deployment_fixture)
    deployment_fixture.runtime.manages_persistent_data.return_value = False
    deployment_fixture.app.wait_until_deleted.side_effect = KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _delete(deployment_fixture)

    deployment_fixture.app.delete_source.assert_not_called()
    (message,) = deployment_fixture.reporter.note.call_args.args
    assert f"databricks workspace delete {_DEFAULT_SOURCE} --recursive" in message


@pytest.mark.parametrize(
    ("active", "pending", "expected"),
    [
        ("/a", None, "/a"),
        (None, "/p", "/p"),
        ("/a", "/p", "/p"),  # a pending rollout is newer than the active one
        (None, None, None),
    ],
)
def test_recorded_source_code_path_reads_the_newest_deployment(active, pending, expected):
    app = App(
        name="app",
        active_deployment=AppDeployment(source_code_path=active) if active else None,
        pending_deployment=AppDeployment(source_code_path=pending) if pending else None,
    )

    assert recorded_source_code_path(app) == expected


def _app_provisioner(workspace_client) -> AppProvisioner:
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = SimpleNamespace(workspace_client=workspace_client)
    return AppProvisioner(Mock(spec=AppsClient), provider, Mock(), Mock(spec=Reporter))


def test_find_returns_none_only_when_the_app_does_not_exist():
    w = fake_workspace_client()
    provisioner = _app_provisioner(w)
    app = provisioner.find(DeploymentName("agent-bricks-demo"))
    assert app is not None and app.name == "agent-bricks-demo"

    w.remove_app()
    assert provisioner.find(DeploymentName("agent-bricks-demo")) is None


@pytest.mark.parametrize("error", [Unauthenticated("expired"), InternalError("boom")])
def test_find_raises_when_it_cannot_tell(error):
    w = Mock()
    w.apps.get.side_effect = error

    with pytest.raises(AgentCliError, match="Could not check whether deployment"):
        _app_provisioner(w).find(DeploymentName("agent-bricks-demo"))


def test_wait_until_deleted_retries_a_check_that_fails(monkeypatch):
    monkeypatch.setattr(app_provisioner_mod.time, "sleep", lambda _s: None)
    w = Mock()
    w.apps.get.side_effect = [
        InternalError("blip"),
        TimeoutError("Timed out after 0:01:00"),
        requests.ConnectionError("connection reset"),
        App(name="app"),
        NotFound("gone"),
    ]

    assert _app_provisioner(w).wait_until_deleted(DeploymentName("app")) == AppDeleteWaitResult()
    assert w.apps.get.call_count == 5


_TOKEN_REFRESH_FAILED = ValueError("oauth2: token expired and refresh token is not set")
# What a `databricks auth login` profile's credential provider raises when its refresh fails.
_CLI_LOGIN_EXPIRED = OSError("cannot get access token: refresh token is invalid")
# What the OIDC and Model Serving credential providers raise.
_OTHER_LOGIN_FAILURES = [Exception("Cannot get oidc token"), RuntimeError("no invokers token")]


@pytest.mark.parametrize(
    "error",
    [
        Unauthenticated("expired"),
        PermissionDenied("no"),
        BadRequest("bad"),
        _TOKEN_REFRESH_FAILED,
        _CLI_LOGIN_EXPIRED,
        *_OTHER_LOGIN_FAILURES,
    ],
)
def test_wait_until_deleted_stops_at_once_on_an_error_that_wont_clear(monkeypatch, error):
    monkeypatch.setattr(app_provisioner_mod.time, "sleep", lambda _s: None)
    w = Mock()
    w.apps.get.side_effect = error

    wait = _app_provisioner(w).wait_until_deleted(DeploymentName("app"))

    assert wait == AppDeleteWaitResult(error=str(error))
    assert w.apps.get.call_count == 1


def test_wait_until_deleted_reports_running_out():
    w = fake_workspace_client()

    wait = _app_provisioner(w).wait_until_deleted(DeploymentName("app"), timeout_s=0)

    assert wait == AppDeleteWaitResult(timed_out=True)


def test_wait_until_deleted_running_out_keeps_the_last_failed_check():
    w = Mock()
    w.apps.get.side_effect = InternalError("boom")

    wait = _app_provisioner(w).wait_until_deleted(DeploymentName("app"), timeout_s=0)

    assert wait == AppDeleteWaitResult(timed_out=True, error="boom")


@pytest.mark.parametrize(
    "error", [_TOKEN_REFRESH_FAILED, _CLI_LOGIN_EXPIRED, *_OTHER_LOGIN_FAILURES]
)
def test_find_reports_a_failed_login(error):
    w = Mock()
    w.apps.get.side_effect = error

    with pytest.raises(AgentCliError, match="Could not check whether deployment"):
        _app_provisioner(w).find(DeploymentName("agent-bricks-demo"))


def test_delete_source_deletes_the_folder_recursively():
    w = Mock()

    assert _app_provisioner(w).delete_source("/Workspace/x") is True
    w.workspace.delete.assert_called_once_with("/Workspace/x", recursive=True)


def test_delete_source_of_a_missing_folder_is_not_an_error():
    w = Mock()
    w.workspace.delete.side_effect = ResourceDoesNotExist("Path (/Workspace/x) doesn't exist.")

    assert _app_provisioner(w).delete_source("/Workspace/x") is False


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("Timed out after 0:01:00"),
        requests.ConnectionError("connection reset"),
        _TOKEN_REFRESH_FAILED,
        _CLI_LOGIN_EXPIRED,
        *_OTHER_LOGIN_FAILURES,
    ],
)
def test_delete_source_network_or_login_failure_is_reported_not_a_traceback(error):
    w = Mock()
    w.workspace.delete.side_effect = error

    with pytest.raises(AgentCliError, match="Could not delete the uploaded source folder"):
        _app_provisioner(w).delete_source("/Workspace/x")


def test_delete_source_failure_carries_the_reason():
    w = Mock()
    w.workspace.delete.side_effect = PermissionDenied("User cannot delete /Workspace/x")

    with pytest.raises(AgentCliError, match="User cannot delete"):
        _app_provisioner(w).delete_source("/Workspace/x")


# --- state file and wait -------------------------------------------------------------------


class _Clock:
    """Wall-clock stand-in: ``sleep`` advances time and runs an optional per-sleep hook."""

    def __init__(self, start: float = 1_800_000_000.0, on_sleep=None) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self.on_sleep = on_sleep

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(self)


def _service_with(fixture, clock: _Clock, **extra) -> DeployService:
    """The fixture's service, rebuilt with a fake clock (and any extra collaborators)."""
    return DeployService(
        project_resolver=fixture.resolver,
        apps_client=fixture.apps,
        api_client_provider=fixture.provider,
        app_provisioner=fixture.app,
        memory_store_provisioner=fixture.memory,
        session_store_provisioner=fixture.session,
        tracing_provisioner=fixture.tracing,
        runtime_store_provisioner=fixture.runtime,
        tool_access_provisioner=fixture.tool_access,
        model_binding_provisioner=ModelBindingProvisioner(fixture.provider, fixture.reporter),
        reporter=fixture.reporter,
        clock=clock,
        sleep=clock.sleep,
        **extra,
    )


def _notes(fixture) -> list[str]:
    return [call.args[0] for call in fixture.reporter.note.call_args_list]


def test_deploy_records_success_through_the_state_tracker(
    deployment_fixture, tmp_path: pathlib.Path
):
    tracker = DeployStateTracker(result_payload=lambda result: {"url": result.url})
    service = _service_with(deployment_fixture, _Clock(), state_tracker=tracker)

    service.deploy(_request(tmp_path))

    state = deploy_state.read_state(deploy_state.state_path(tmp_path, "agent-bricks-demo"))
    assert state is not None
    assert state["status"] == "succeeded"
    assert state["result"] == {"url": "https://demo.example"}


def test_deploy_records_failure_and_reraises(deployment_fixture, tmp_path: pathlib.Path):
    deployment_fixture.app.deploy_source.side_effect = AgentCliError(
        "Could not deploy 'agent-bricks-demo'.", hint="app crashed"
    )
    tracker = DeployStateTracker(result_payload=lambda result: {})
    service = _service_with(deployment_fixture, _Clock(), state_tracker=tracker)

    with pytest.raises(AgentCliError):
        service.deploy(_request(tmp_path))

    state = deploy_state.read_state(deploy_state.state_path(tmp_path, "agent-bricks-demo"))
    assert state is not None
    assert state["status"] == "failed"
    assert state["error"] == "Could not deploy 'agent-bricks-demo'.\napp crashed"


def test_prepare_refuses_a_running_deploy_before_any_change(
    deployment_fixture, tmp_path: pathlib.Path
):
    clock = _Clock()
    deploy_state.write_state(
        deploy_state.state_path(tmp_path, "agent-bricks-demo"),
        {
            "pid": os.getppid(),
            "status": "running",
            "phase_started_at": deploy_state.timestamp(clock.now),
        },
    )

    with pytest.raises(AgentCliError, match="already running") as raised:
        _service_with(deployment_fixture, clock).prepare(_request(tmp_path))

    assert raised.value.hint == "run `agentbricks deployments wait agent-bricks-demo`"
    deployment_fixture.app.prepare_app_auth.assert_not_called()
    assert deployment_fixture.project.deployment_name is None


def test_wait_follows_the_local_record_until_it_succeeds(deployment_fixture, tmp_path):
    path = deploy_state.state_path(tmp_path, "agent-bricks-demo")
    start = 1_800_000_000.0

    def advance(clock: _Clock) -> None:
        elapsed = clock.now - start
        state: dict[str, Any] = {
            "pid": os.getpid(),
            "started_at": deploy_state.timestamp(start),
            "status": "running",
            "phase": "Waiting for agent compute to start",
            "phase_started_at": deploy_state.timestamp(start),
        }
        if elapsed >= 20:
            state.update(phase="Deploying the agent source")
            state["phase_started_at"] = deploy_state.timestamp(start + 20)
        if elapsed >= 40:
            state.update(status="succeeded", result={"url": "https://demo.example"})
        deploy_state.write_state(path, state)

    clock = _Clock(start, on_sleep=advance)
    advance(clock)

    result = _service_with(deployment_fixture, clock).wait(None, tmp_path, timeout_s=100)

    assert result.status == "succeeded"
    assert result.url == "https://demo.example"
    assert result.deployment == "agent-bricks-demo"
    assert result.elapsed_s == 40
    assert result.uses_runtime_api is True
    assert set(clock.sleeps) == {2.0}  # the local record is re-read every 2 s
    assert _notes(deployment_fixture) == [
        "• Waiting for agent compute to start…",
        "• Waiting for agent compute to start… 0:16",
        "• Deploying the agent source…",
        "• Deploying the agent source… 0:16",
    ]


def test_wait_returns_running_at_the_timeout(deployment_fixture, tmp_path):
    clock = _Clock()
    deploy_state.write_state(
        deploy_state.state_path(tmp_path, "agent-bricks-demo"),
        {
            "pid": os.getpid(),
            "started_at": deploy_state.timestamp(clock.now - 60),
            "status": "running",
            "phase": "Waiting for agent compute to start",
            "phase_started_at": deploy_state.timestamp(clock.now - 30),
        },
    )

    result = _service_with(deployment_fixture, clock).wait("demo", tmp_path, timeout_s=10)

    assert result.status == "running"
    assert result.phase == "Waiting for agent compute to start"
    assert result.elapsed_s == 70
    assert sum(clock.sleeps) == 10
    # Joined 30 s into the phase, so the next heartbeat (0:45) falls after the 10 s timeout.
    assert _notes(deployment_fixture) == ["• Waiting for agent compute to start…"]


def test_wait_follows_the_app_from_starting_to_active(deployment_fixture, tmp_path):
    payloads = [
        {"compute_status": {"state": "STARTING"}},
        {"compute_status": {"state": "STARTING"}},
        {"compute_status": {"state": "ACTIVE"}, "pending_deployment": {"status": {}}},
        {
            "url": "https://demo.example",
            "compute_status": {"state": "ACTIVE"},
            "active_deployment": {"status": {"state": "SUCCEEDED"}},
        },
    ]
    deployment_fixture.apps.find.side_effect = payloads
    clock = _Clock()

    result = _service_with(deployment_fixture, clock).wait("demo", tmp_path, timeout_s=100)

    assert result.status == "succeeded"
    assert result.url == "https://demo.example"
    assert result.elapsed_s == 15
    assert clock.sleeps == [5.0, 5.0, 5.0]
    deployment_fixture.apps.find.assert_called_with(DeploymentName("agent-bricks-demo"))
    assert _notes(deployment_fixture) == [
        "• Waiting for agent compute to start…",
        "• Deploying the agent source…",
    ]


def test_wait_reports_the_apps_failure_message(deployment_fixture, tmp_path):
    deployment_fixture.apps.find.return_value = {
        "compute_status": {"state": "ACTIVE"},
        "active_deployment": {"status": {"state": "FAILED", "message": "build failed"}},
    }

    result = _service_with(deployment_fixture, _Clock()).wait("demo", tmp_path, timeout_s=100)

    assert result.status == "failed"
    assert result.error == "Could not deploy 'agent-bricks-demo'."
    assert result.log_tail == ["build failed"]
