"""One service-level example of composing a deployment from injected collaborators."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.app_auth_client import AppAuthClient, AppAuthResult
from databricks_agentbricks.clients.apps_client import AppsClient
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
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.projects.app_manifest import AppManifest
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deploy_service import DeployRequest, DeployService


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
