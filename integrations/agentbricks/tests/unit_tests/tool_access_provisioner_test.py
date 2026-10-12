"""Composition tests for concrete tool-access provisioning during deployment."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from databricks.sdk.service.workspace import (
    ObjectInfo,
    ObjectType,
    WorkspaceObjectAccessControlResponse,
    WorkspaceObjectPermission,
    WorkspaceObjectPermissionLevel,
    WorkspaceObjectPermissions,
)

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.apps_user_auth_client import AppAuthPlan
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject, ToolSpec
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services import deploy_service as deploy_service_mod
from databricks_agentbricks.services.deploy_service import DeployRequest, DeployService
from databricks_agentbricks.services.deployment.provisioners import (
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
from databricks_agentbricks.services.deployment.tool_access import ToolAccessPlan, WorkspaceGrant
from databricks_agentbricks.services.deployment.tool_access_provisioner import ToolAccessProvisioner


def _harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Build a DeployService with the concrete ToolAccessProvisioner and fake external clients."""
    monkeypatch.setattr(deploy_service_mod, "require_managed_tool_support", lambda _: None)

    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.add_tool(ToolSpec.uc_function("search", function="main.tools.search", auth="app"))
    project.write()

    resolver = Mock(spec=ProjectResolver)
    resolver.load.return_value = project
    resolver.resolve_deployment_name.return_value = "demo"
    resolver.resource_bindings.return_value = (None, None, None)

    apps = Mock(spec=AppsClient)
    apps.exists.return_value = False
    apps.get_app_url.return_value = "https://demo.example"
    apps.get_service_principal.return_value = "app-sp"

    workspace_client = Mock()
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = SimpleNamespace(
        host="https://workspace.example",
        current_user="james@example.com",
        workspace_client=workspace_client,
    )
    reporter = Mock(spec=Reporter)
    reporter.progress.return_value = nullcontext()
    reporter.status.return_value = nullcontext()

    app = Mock(spec=AppProvisioner)
    app.prepare_app_auth.return_value = AppAuthPlan(scope_update=None)
    app.resolve_workspace_path.return_value = "/Workspace/demo"
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

    tool_access = ToolAccessProvisioner(apps, "selected", reporter)
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
        reporter=reporter,
    )
    return SimpleNamespace(
        apps=apps,
        app=app,
        service=service,
        project=project,
        tool_access=tool_access,
        workspace_client=workspace_client,
    )


def _request(source: Path) -> DeployRequest:
    return DeployRequest(
        name="demo",
        source=str(source),
        pip_index_url=None,
        workspace_path="/Workspace/demo",
        instance_count=None,
        allow_user_scope_update=False,
    )


def _workspace_permissions(
    principal: str, permission: WorkspaceObjectPermissionLevel
) -> WorkspaceObjectPermissions:
    return WorkspaceObjectPermissions(
        access_control_list=[
            WorkspaceObjectAccessControlResponse(
                service_principal_name=principal,
                all_permissions=[
                    WorkspaceObjectPermission(inherited=False, permission_level=permission)
                ],
            )
        ]
    )


def _plan_with_workspace_grant(harness: SimpleNamespace) -> ToolAccessPlan:
    plan = harness.tool_access.plan(harness.project.tools)
    return ToolAccessPlan(
        app_resources=plan.app_resources,
        workspace_grants=(
            WorkspaceGrant(
                "/Workspace/Shared/tool-inputs", WorkspaceObjectPermissionLevel.CAN_EDIT
            ),
        ),
    )


def test_concrete_provisioner_adds_grants_before_rollout_and_prunes_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    harness = _harness(tmp_path, monkeypatch)
    events: list[str] = []
    harness.apps.add_tool_resources_for_rollout.side_effect = lambda app, resources: events.append(
        "add"
    )
    harness.apps.apply_tool_resources.side_effect = lambda app, resources: events.append("prune")
    harness.app.deploy_source.side_effect = lambda *args: events.append("rollout")

    result = harness.service.deploy(_request(tmp_path))

    assert events == ["add", "rollout", "prune"]
    harness.apps.get_service_principal.assert_called_once_with("agent-bricks-demo")
    harness.apps.add_tool_resources_for_rollout.assert_called_once()
    harness.apps.apply_tool_resources.assert_called_once()
    assert result.tool_access is not None
    assert result.tool_access.app_resources == 1


def test_concrete_provisioner_forwards_uc_function_and_workspace_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    harness = _harness(tmp_path, monkeypatch)
    plan = _plan_with_workspace_grant(harness)
    monkeypatch.setattr(harness.tool_access, "plan", lambda _: plan)

    workspace = harness.workspace_client.workspace
    workspace.get_status.return_value = ObjectInfo(object_id=123, object_type=ObjectType.DIRECTORY)
    workspace.get_permissions.side_effect = [
        WorkspaceObjectPermissions(access_control_list=[]),
        _workspace_permissions("app-sp", WorkspaceObjectPermissionLevel.CAN_EDIT),
    ]
    events: list[str] = []
    workspace.update_permissions.side_effect = lambda *args, **kwargs: events.append("workspace")
    harness.apps.add_tool_resources_for_rollout.side_effect = lambda *args: events.append("apps")
    harness.app.deploy_source.side_effect = lambda *args: events.append("rollout")
    harness.apps.apply_tool_resources.side_effect = lambda *args: events.append("prune")

    result = harness.service.deploy(_request(tmp_path))

    assert events == ["workspace", "apps", "rollout", "prune"]
    assert [resource["uc_securable"] for resource in plan.app_resources] == [
        {
            "securable_full_name": "main.tools.search",
            "securable_type": "FUNCTION",
            "permission": "EXECUTE",
        }
    ]
    harness.apps.get_service_principal.assert_called_once_with("agent-bricks-demo")
    harness.apps.add_tool_resources_for_rollout.assert_called_once_with(
        "agent-bricks-demo", plan.app_resources
    )
    harness.apps.apply_tool_resources.assert_called_once_with(
        "agent-bricks-demo", plan.app_resources
    )
    workspace.get_status.assert_called_once_with("/Workspace/Shared/tool-inputs")
    assert workspace.get_permissions.call_args_list == [
        call("directories", "123"),
        call("directories", "123"),
    ]
    workspace.update_permissions.assert_called_once()
    assert workspace.update_permissions.call_args.args == ("directories", "123")
    request = workspace.update_permissions.call_args.kwargs["access_control_list"][0]
    assert request.as_dict() == {
        "permission_level": "CAN_EDIT",
        "service_principal_name": "app-sp",
    }
    assert result.tool_access is not None
    assert result.tool_access.workspace_grants == 1


def test_concrete_provisioner_workspace_grant_failure_stops_before_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    harness = _harness(tmp_path, monkeypatch)
    plan = _plan_with_workspace_grant(harness)
    monkeypatch.setattr(harness.tool_access, "plan", lambda _: plan)

    workspace = harness.workspace_client.workspace
    workspace.get_status.return_value = ObjectInfo(object_id=123, object_type=ObjectType.DIRECTORY)
    workspace.get_permissions.return_value = WorkspaceObjectPermissions(access_control_list=[])

    with pytest.raises(AgentCliError, match="did not become effective"):
        harness.service.deploy(_request(tmp_path))

    workspace.update_permissions.assert_called_once()
    harness.apps.add_tool_resources_for_rollout.assert_not_called()
    harness.app.deploy_source.assert_not_called()
    harness.apps.apply_tool_resources.assert_not_called()


def test_concrete_provisioner_failure_stops_before_rollout_and_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    harness = _harness(tmp_path, monkeypatch)
    harness.apps.add_tool_resources_for_rollout.return_value = "resource update denied"

    with pytest.raises(AgentCliError, match="explicit tool resources"):
        harness.service.deploy(_request(tmp_path))

    harness.app.deploy_source.assert_not_called()
    harness.apps.apply_tool_resources.assert_not_called()


def test_rollout_failure_keeps_additive_resources_and_skips_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    harness = _harness(tmp_path, monkeypatch)
    harness.apps.add_tool_resources_for_rollout.return_value = None
    harness.app.deploy_source.side_effect = AgentCliError("source rollout failed")

    with pytest.raises(AgentCliError, match="source rollout failed"):
        harness.service.deploy(_request(tmp_path))

    harness.apps.add_tool_resources_for_rollout.assert_called_once()
    harness.apps.apply_tool_resources.assert_not_called()
