"""Runtime Store ownership, reuse, and deployment cleanup."""

import json
import types
from unittest import mock

import pytest
from click.testing import CliRunner
from delete_test_fixtures import fake_workspace_client

from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.clients import managed_runtime_store
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.deployment.provisioners import AppProvisioner


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
        deploy_mod.deployments_delete, ["agent-bricks-myapp", "--yes"], obj=ctx
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


def test_delete_still_deleting_reports_the_result_and_fails(monkeypatch):
    w = fake_workspace_client(source_code_path=_DEFAULT_SOURCE)
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    monkeypatch.setattr(deploy_mod, "_databricks", mock.Mock(side_effect=_cli_runner(w)))
    # The wait itself is covered in provisioner tests; here it runs out at once.
    monkeypatch.setattr(
        AppProvisioner, "wait_until_deleted", lambda self, name: "still being deleted"
    )

    result = CliRunner().invoke(
        deploy_mod.deployments_delete,
        ["agent-bricks-myapp", "--yes"],
        obj=_ctx(mock.Mock(), w, output="json"),
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["still_deleting"] is True
    assert payload["wait_error"] == "still being deleted"
    assert payload["source_path_status"] == "deleted"
