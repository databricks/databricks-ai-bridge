"""Runtime Store ownership, reuse, and deployment cleanup."""

import json
import pathlib
import types
from contextlib import nullcontext
from unittest import mock

import pytest
from click.testing import CliRunner
from databricks.sdk.errors import PermissionDenied
from delete_test_fixtures import fake_workspace_client
from deploy_service_test import _request, deployment_fixture  # noqa: F401

from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.clients import managed_runtime_store
from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.legacy_runtime_store import LakebaseBackend
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deployment.app_provisioner import (
    AppDeleteWaitResult,
    AppProvisioner,
)
from databricks_agentbricks.services.deployment.names import DeploymentName
from databricks_agentbricks.services.deployment.provisioners import (
    ManifestPatch,
    ProjectContext,
    ResourceContext,
    RuntimeStoreProvisioner,
    RuntimeStoreState,
)

_LEGACY_BACKEND = LakebaseBackend(
    project="p",
    branch="production",
    endpoint_id="primary",
    database="db",
    schema="public",
    tables=(),
    resource_name="postgres",
)


def _runtime_provisioner(*, use_managed: bool, apps: mock.Mock, client=None):
    reporter = mock.Mock(spec=Reporter)
    reporter.status.return_value = nullcontext()
    provider = mock.Mock(spec=ApiClientProvider)
    provider.get.return_value = client or mock.Mock()
    return RuntimeStoreProvisioner(provider, apps, "prof", use_managed, reporter)


def _runtime_ctx(tmp_path: pathlib.Path, *, deployment_exists: bool) -> ResourceContext:
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    return ResourceContext(
        project=ProjectContext(tmp_path, DeploymentName("agent-bricks-myapp"), project),
        memory_store=None,
        session_store=None,
        experiment_name=None,
        deployment_exists=deployment_exists,
    )


def test_managed_after_app_created_waits_for_the_principal_not_the_compute(tmp_path):
    apps = mock.Mock(spec=AppsClient)
    apps.wait_for_service_principal.return_value = "sp-123"
    client = mock.Mock()
    client.create_runtime_store.return_value = _runtime_store_response()
    provisioner = _runtime_provisioner(use_managed=True, apps=apps, client=client)
    ctx = _runtime_ctx(tmp_path, deployment_exists=False)
    state = RuntimeStoreState(True, None, ManifestPatch({}))

    patch = provisioner.after_app_created(ctx, state)
    provisioner.attach_to_app(ctx, state)

    apps.wait_for_service_principal.assert_called_once_with("agent-bricks-myapp")
    apps.get_service_principal.assert_not_called()
    apps.wait_for_active.assert_not_called()
    apps.attach_postgres_backends.assert_not_called()
    assert patch.env["DATABRICKS_AGENTBRICKS_RUNTIME_STORE_USERNAME"] == "sp-123"
    assert patch.env["DATABRICKS_AGENTBRICKS_RUNTIME_STORE_SCHEMA"] == "databricks_agentkit_runtime"


def test_managed_after_app_created_fails_clearly_when_the_principal_never_appears(tmp_path):
    apps = mock.Mock(spec=AppsClient)
    apps.wait_for_service_principal.return_value = None
    provisioner = _runtime_provisioner(use_managed=True, apps=apps)

    with pytest.raises(AgentCliError, match="service principal"):
        provisioner.after_app_created(
            _runtime_ctx(tmp_path, deployment_exists=False),
            RuntimeStoreState(True, None, ManifestPatch({})),
        )


def test_legacy_backend_attaches_only_in_attach_to_app(tmp_path):
    apps = mock.Mock(spec=AppsClient)
    apps.attach_postgres_backends.return_value = None
    provisioner = _runtime_provisioner(use_managed=False, apps=apps)
    ctx = _runtime_ctx(tmp_path, deployment_exists=False)
    state = RuntimeStoreState(True, _LEGACY_BACKEND, ManifestPatch({}))

    assert provisioner.after_app_created(ctx, state) == ManifestPatch({})
    apps.attach_postgres_backends.assert_not_called()
    provisioner.attach_to_app(ctx, state)

    apps.attach_postgres_backends.assert_called_once_with("agent-bricks-myapp", [_LEGACY_BACKEND])
    apps.wait_for_service_principal.assert_not_called()


def test_legacy_attach_failure_raises_with_the_resource_error(tmp_path):
    apps = mock.Mock(spec=AppsClient)
    apps.attach_postgres_backends.return_value = "another operation in progress"
    provisioner = _runtime_provisioner(use_managed=False, apps=apps)

    with pytest.raises(AgentCliError, match="Lakebase resource") as exc:
        provisioner.attach_to_app(
            _runtime_ctx(tmp_path, deployment_exists=False),
            RuntimeStoreState(True, _LEGACY_BACKEND, ManifestPatch({})),
        )
    assert exc.value.hint == "another operation in progress"


def test_resume_of_a_never_deployed_app_still_gets_the_schema_env(tmp_path):
    """An App kept by a failed first deploy has no deployment, so it still counts as new."""
    apps = mock.Mock(spec=AppsClient)
    apps.wait_for_service_principal.return_value = "sp-123"
    client = mock.Mock()
    client.create_runtime_store.return_value = _runtime_store_response()
    provisioner = _runtime_provisioner(use_managed=True, apps=apps, client=client)
    state = RuntimeStoreState(True, None, ManifestPatch({}))

    first = provisioner.after_app_created(_runtime_ctx(tmp_path, deployment_exists=False), state)
    redeploy = provisioner.after_app_created(_runtime_ctx(tmp_path, deployment_exists=True), state)

    assert "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_SCHEMA" in first.env
    assert "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_SCHEMA" not in redeploy.env


def test_deploy_runs_managed_store_before_the_join_and_legacy_attach_after(
    deployment_fixture,  # noqa: F811
    tmp_path,
):
    f = deployment_fixture
    f.runtime.reconcile.return_value = RuntimeStoreState(True, None, ManifestPatch({}))
    phases = mock.Mock()
    phases.attach_mock(f.runtime.after_app_created, "after_app_created")
    phases.attach_mock(f.app.sync_source, "sync_source")
    phases.attach_mock(f.app.wait_for_compute, "wait_for_compute")
    phases.attach_mock(f.runtime.attach_to_app, "attach_to_app")
    phases.attach_mock(f.app.roll_out, "roll_out")

    f.service.deploy(_request(tmp_path))

    assert [call[0] for call in phases.mock_calls] == [
        "after_app_created",
        "sync_source",
        "wait_for_compute",
        "attach_to_app",
        "roll_out",
    ]


@pytest.fixture(autouse=True)
def _managed_runtime_store(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", True)


def _runtime_store_response(app_name="agent-bricks-myapp", sp="sp-123"):
    return {
        "name": f"runtime-stores/{app_name}",
        "owner": {"app": {"name": app_name, "service_principal_id": sp}},
        "storage_backend": {
            "lakebase": {
                "project_id": "databricks-internal-custom-agents",
                "branch": "projects/databricks-internal-custom-agents/branches/production",
                "database_id": "runtime-agent-bricks-myapp-550e8400-e29b-41d4-a716-446655440000",
            }
        },
    }


_DEFAULT_SOURCE = "/Workspace/Users/user@example.com/agentbricks_deployments/agent-bricks-myapp"


def _ctx(client, workspace_client, *, output="text"):
    client.workspace_client = workspace_client
    return types.SimpleNamespace(
        profile="prof",
        output=output,
        api_client_provider=types.SimpleNamespace(get=lambda: client),
        announce_workspace=lambda: None,
    )


def _cli_runner(workspace_client):
    """A `databricks` CLI whose `apps delete` removes the fake SDK's App."""

    def run(args, profile, **kwargs):
        if args[:2] == ["apps", "delete"]:
            workspace_client.remove_app()
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def test_reconcile_runtime_store_creates_with_app_identity() -> None:
    client = mock.Mock()
    client.create_runtime_store.return_value = _runtime_store_response()

    backend = managed_runtime_store.get_or_create_backend(client, "agent-bricks-myapp", "sp-123")

    assert backend.branch == "projects/databricks-internal-custom-agents/branches/production"
    assert backend.database_id == "runtime-agent-bricks-myapp-550e8400-e29b-41d4-a716-446655440000"
    client.create_runtime_store.assert_called_once_with(
        "agent-bricks-myapp", "sp-123", app_name="agent-bricks-myapp", retry_transient=True
    )
    client.get_runtime_store.assert_not_called()


def test_reconcile_runtime_store_reuses_on_already_exists() -> None:
    client = mock.Mock()
    client.create_runtime_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.get_runtime_store.return_value = _runtime_store_response()

    backend = managed_runtime_store.get_or_create_backend(client, "agent-bricks-myapp", "sp-123")

    assert backend.database_id.startswith("runtime-agent-bricks-myapp-")
    client.get_runtime_store.assert_called_once_with("agent-bricks-myapp")


@pytest.mark.parametrize(
    ("app_name", "sp"),
    [
        (None, "sp-123"),
        ("", "sp-123"),
        ("agent-bricks-other", "sp-123"),
        ("agent-bricks-myapp", "other-sp"),
    ],
)
def test_reconcile_runtime_store_rejects_a_different_or_incomplete_owner(app_name, sp):
    client = mock.Mock()
    client.create_runtime_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    resource = _runtime_store_response(sp=sp)
    if app_name is None:
        del resource["owner"]["app"]["name"]
    else:
        resource["owner"]["app"]["name"] = app_name
    client.get_runtime_store.return_value = resource

    with pytest.raises(AgentCliError, match="does not belong to this app identity"):
        managed_runtime_store.get_or_create_backend(client, "agent-bricks-myapp", "sp-123")


@pytest.mark.parametrize("operation", ["create", "get"])
def test_reconcile_runtime_store_propagates_api_failure(operation):
    client = mock.Mock()
    error = AgentCliError("denied", error_code="PERMISSION_DENIED")
    if operation == "get":
        client.create_runtime_store.side_effect = AgentCliError(
            "exists", error_code="ALREADY_EXISTS"
        )
        client.get_runtime_store.side_effect = error
    else:
        client.create_runtime_store.side_effect = error

    with pytest.raises(AgentCliError) as exc:
        managed_runtime_store.get_or_create_backend(client, "agent-bricks-myapp", "sp-123")
    assert exc.value is error


def test_reconcile_runtime_store_requires_app_service_principal() -> None:
    with pytest.raises(AgentCliError, match="app's service principal"):
        managed_runtime_store.get_or_create_backend(mock.Mock(), "agent-bricks-myapp", None)


def test_managed_delete_finishes_before_deleting_app(monkeypatch):
    client = mock.Mock()
    client.get_runtime_store.return_value = _runtime_store_response()
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    cli = mock.Mock(side_effect=_cli_runner(w))
    calls = mock.Mock()
    calls.attach_mock(client, "runtime")
    calls.attach_mock(cli, "cli")
    calls.attach_mock(w, "w")
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = _ctx(client, w, output="json")

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--delete-source", "--yes"], obj=ctx
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "deleted": "agent-bricks-myapp",
        "still_deleting": False,
        "wait_error": None,
        "source_path": _DEFAULT_SOURCE,
        "source_path_status": "deleted",
        "source_path_error": None,
    }
    assert calls.mock_calls == [
        mock.call.w.apps.get("agent-bricks-myapp"),
        mock.call.runtime.get_runtime_store("agent-bricks-myapp"),
        mock.call.runtime.delete_runtime_store("agent-bricks-myapp"),
        mock.call.cli(
            ["apps", "delete", "agent-bricks-myapp"],
            "prof",
            to_stderr=True,
            action="Could not delete deployment 'agent-bricks-myapp'.",
        ),
        mock.call.w.apps.get("agent-bricks-myapp"),
        mock.call.w.workspace.delete(_DEFAULT_SOURCE, recursive=True),
    ]


@pytest.mark.parametrize("operation", ["get_runtime_store", "delete_runtime_store"])
@pytest.mark.parametrize("error_code", ["PERMISSION_DENIED", "UNAVAILABLE", "FEATURE_DISABLED"])
def test_managed_cleanup_errors_retain_the_app(monkeypatch, operation, error_code):
    client = mock.Mock()
    client.get_runtime_store.return_value = _runtime_store_response()
    getattr(client, operation).side_effect = AgentCliError("cleanup failed", error_code=error_code)
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    cli = mock.Mock(side_effect=_cli_runner(w))
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = _ctx(client, w)

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=ctx
    )

    assert result.exit_code != 0
    assert "deployment was retained" in result.output
    assert not any(
        call.args and call.args[0][:2] == ["apps", "delete"] for call in cli.call_args_list
    )


@pytest.mark.parametrize("operation", ["get_runtime_store", "delete_runtime_store"])
def test_managed_store_already_absent_allows_app_deletion(monkeypatch, operation):
    client = mock.Mock()
    client.get_runtime_store.return_value = _runtime_store_response()
    getattr(client, operation).side_effect = AgentCliError("absent", error_code="NOT_FOUND")
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    cli = mock.Mock(side_effect=_cli_runner(w))
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = _ctx(client, w)

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=ctx
    )

    assert result.exit_code == 0, result.output
    assert any(call.args and call.args[0][:2] == ["apps", "delete"] for call in cli.call_args_list)


def test_managed_delete_rejects_a_different_owner(monkeypatch):
    client = mock.Mock()
    client.get_runtime_store.return_value = _runtime_store_response(sp="different-sp")
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    cli = mock.Mock(side_effect=_cli_runner(w))
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = _ctx(client, w)

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=ctx
    )

    assert result.exit_code != 0
    client.delete_runtime_store.assert_not_called()
    assert not any(
        call.args and call.args[0][:2] == ["apps", "delete"] for call in cli.call_args_list
    )


def test_managed_delete_rejects_an_unexpected_resource_name(monkeypatch):
    client = mock.Mock()
    resource = _runtime_store_response()
    resource["name"] = "runtime-stores/other-app"
    client.get_runtime_store.return_value = resource
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    cli = mock.Mock(side_effect=_cli_runner(w))
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = _ctx(client, w)

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=ctx
    )

    assert result.exit_code != 0
    client.delete_runtime_store.assert_not_called()
    assert not any(
        call.args and call.args[0][:2] == ["apps", "delete"] for call in cli.call_args_list
    )


def test_managed_delete_cannot_skip_cleanup_without_the_app_identity(monkeypatch):
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE, service_principal=None)
    cli = mock.Mock(side_effect=_cli_runner(w))
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    client = mock.Mock()
    ctx = _ctx(client, w)

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=ctx
    )

    assert result.exit_code != 0
    assert "deployment was retained" in result.output
    client.assert_not_called()
    cli.assert_not_called()


def test_legacy_delete_preserves_existing_behavior(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    cli = mock.Mock(side_effect=_cli_runner(w))
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    client = mock.Mock()
    ctx = _ctx(client, w)

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=ctx
    )

    assert result.exit_code == 0, result.output
    client.assert_not_called()
    cli.assert_called_once_with(
        ["apps", "delete", "agent-bricks-myapp"],
        "prof",
        to_stderr=True,
        action="Could not delete deployment 'agent-bricks-myapp'.",
    )


def test_delete_of_a_missing_deployment_says_so_and_deletes_nothing(monkeypatch):
    w = fake_workspace_client()
    w.remove_app()
    cli = mock.Mock(side_effect=_cli_runner(w))
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    client = mock.Mock()

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=_ctx(client, w)
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output
    cli.assert_not_called()
    client.delete_runtime_store.assert_not_called()
    w.workspace.delete.assert_not_called()


@pytest.mark.parametrize(
    ("wait", "still_deleting"),
    [(AppDeleteWaitResult(timed_out=True), True), (AppDeleteWaitResult(error="expired"), False)],
)
def test_unconfirmed_delete_reports_the_result_and_fails(monkeypatch, wait, still_deleting):
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    monkeypatch.setattr(deploy_mod, "_databricks", mock.Mock(side_effect=_cli_runner(w)))
    # The wait itself is covered in provisioner tests; here it ends unconfirmed at once.
    monkeypatch.setattr(AppProvisioner, "wait_until_deleted", lambda self, name: wait)

    result = CliRunner().invoke(
        deploy_mod.deployments_delete,
        ["agent-bricks-myapp", "--delete-source", "--yes"],
        obj=_ctx(mock.Mock(), w, output="json"),
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["still_deleting"] is still_deleting
    assert payload["wait_error"] == wait.error
    assert payload["source_path_status"] == "kept_app_not_confirmed"
    w.workspace.delete.assert_not_called()


@pytest.mark.parametrize(
    ("wait", "expected"),
    [
        (AppDeleteWaitResult(timed_out=True), "it may finish on its own"),
        (AppDeleteWaitResult(error="expired"), "agentbricks auth login"),
    ],
)
def test_unconfirmed_delete_text_says_what_to_do(monkeypatch, wait, expected):
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    monkeypatch.setattr(deploy_mod, "_databricks", mock.Mock(side_effect=_cli_runner(w)))
    monkeypatch.setattr(AppProvisioner, "wait_until_deleted", lambda self, name: wait)

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=_ctx(mock.Mock(), w)
    )

    assert result.exit_code == 1
    assert expected in " ".join(result.output.split())


def test_delete_without_delete_source_keeps_the_folder_and_says_how_to_delete_it(monkeypatch):
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    monkeypatch.setattr(deploy_mod, "_databricks", mock.Mock(side_effect=_cli_runner(w)))

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=_ctx(mock.Mock(), w)
    )

    assert result.exit_code == 0, result.output
    w.workspace.delete.assert_not_called()
    assert f"databricks workspace delete {_DEFAULT_SOURCE} --recursive" in " ".join(
        result.output.split()
    )


def test_delete_fails_when_the_requested_source_folder_could_not_be_deleted(monkeypatch):
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    w.workspace.delete.side_effect = PermissionDenied("no permission")
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    monkeypatch.setattr(deploy_mod, "_databricks", mock.Mock(side_effect=_cli_runner(w)))

    result = CliRunner().invoke(
        deploy_mod.deployments_delete,
        ["agent-bricks-myapp", "--delete-source", "--yes"],
        obj=_ctx(mock.Mock(), w),
    )

    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "no permission" in output
    assert f"databricks workspace delete {_DEFAULT_SOURCE} --recursive" in output


def test_delete_command_quotes_a_source_path_for_the_shell(monkeypatch):
    path = "/Workspace/Users/user@example.com/my agent"
    w = fake_workspace_client(source_code_path=path)
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    monkeypatch.setattr(deploy_mod, "_databricks", mock.Mock(side_effect=_cli_runner(w)))

    result = CliRunner().invoke(
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=_ctx(mock.Mock(), w)
    )

    assert result.exit_code == 0, result.output
    assert f"databricks workspace delete '{path}' --recursive" in result.output


@pytest.mark.parametrize(("flags", "named"), [(["--delete-source"], True), ([], False)])
def test_delete_prompt_names_the_folder_only_with_delete_source(monkeypatch, flags, named):
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    monkeypatch.setattr(deploy_mod, "_databricks", mock.Mock(side_effect=_cli_runner(w)))

    result = CliRunner().invoke(
        deploy_mod.deployments_delete,
        ["agent-bricks-myapp", *flags],
        obj=_ctx(mock.Mock(), w),
        input="n\n",
    )

    assert result.exit_code != 0  # declined
    assert (_DEFAULT_SOURCE in " ".join(result.output.split())) is named
    w.workspace.delete.assert_not_called()
