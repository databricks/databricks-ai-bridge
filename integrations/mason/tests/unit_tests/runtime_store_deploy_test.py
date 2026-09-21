"""Runtime Store ownership, reuse, and deployment cleanup."""

import json
import types
from unittest import mock

import pytest
from click.testing import CliRunner

from databricks_mason.cli import deploy as deploy_mod
from databricks_mason.errors import AgentCliError


@pytest.fixture(autouse=True)
def _managed_runtime_store(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda *args: "sp-123")
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", True)


def _runtime_store_response(app_name="mason-myapp", sp="sp-123"):
    return {
        "name": f"runtime-stores/{app_name}",
        "owner": {"app": {"name": app_name, "service_principal_id": sp}},
        "storage_backend": {
            "lakebase": {
                "project_id": "databricks-internal-custom-agents",
                "branch": "projects/databricks-internal-custom-agents/branches/production",
                "database_id": "runtime-mason-myapp-550e8400-e29b-41d4-a716-446655440000",
            }
        },
    }


def test_reconcile_runtime_store_creates_with_app_identity() -> None:
    client = mock.Mock()
    client.create_runtime_store.return_value = _runtime_store_response()

    backend = deploy_mod.managed_runtime_store.get_or_create_backend(
        client, "mason-myapp", "sp-123"
    )

    assert backend.branch == "projects/databricks-internal-custom-agents/branches/production"
    assert backend.database_id == "runtime-mason-myapp-550e8400-e29b-41d4-a716-446655440000"
    client.create_runtime_store.assert_called_once_with(
        "mason-myapp", "sp-123", app_name="mason-myapp", retry_transient=True
    )
    client.get_runtime_store.assert_not_called()


def test_reconcile_runtime_store_reuses_on_already_exists() -> None:
    client = mock.Mock()
    client.create_runtime_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    client.get_runtime_store.return_value = _runtime_store_response()

    backend = deploy_mod.managed_runtime_store.get_or_create_backend(
        client, "mason-myapp", "sp-123"
    )

    assert backend.database_id.startswith("runtime-mason-myapp-")
    client.get_runtime_store.assert_called_once_with("mason-myapp")


@pytest.mark.parametrize(
    ("app_name", "sp"),
    [(None, "sp-123"), ("", "sp-123"), ("mason-other", "sp-123"), ("mason-myapp", "other-sp")],
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
        deploy_mod.managed_runtime_store.get_or_create_backend(client, "mason-myapp", "sp-123")


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
        deploy_mod.managed_runtime_store.get_or_create_backend(client, "mason-myapp", "sp-123")
    assert exc.value is error


def test_reconcile_runtime_store_requires_app_service_principal() -> None:
    with pytest.raises(AgentCliError, match="app's service principal"):
        deploy_mod.managed_runtime_store.get_or_create_backend(mock.Mock(), "mason-myapp", None)


def test_managed_delete_finishes_before_deleting_app(monkeypatch):
    client = mock.Mock()
    client.get_runtime_store.return_value = _runtime_store_response()
    cli = mock.Mock()
    calls = mock.Mock()
    calls.attach_mock(client, "runtime")
    calls.attach_mock(cli, "cli")
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="json", client=lambda: client)

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"deleted": "mason-myapp"}
    assert calls.mock_calls == [
        mock.call.runtime.get_runtime_store("mason-myapp"),
        mock.call.runtime.delete_runtime_store("mason-myapp"),
        mock.call.cli(
            ["apps", "delete", "mason-myapp"],
            "prof",
            action="Could not delete deployment 'mason-myapp'.",
        ),
    ]


@pytest.mark.parametrize("operation", ["get_runtime_store", "delete_runtime_store"])
@pytest.mark.parametrize("error_code", ["PERMISSION_DENIED", "UNAVAILABLE", "FEATURE_DISABLED"])
def test_managed_cleanup_errors_retain_the_app(monkeypatch, operation, error_code):
    client = mock.Mock()
    client.get_runtime_store.return_value = _runtime_store_response()
    getattr(client, operation).side_effect = AgentCliError("cleanup failed", error_code=error_code)
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code != 0
    assert "deployment was retained" in result.output
    cli.assert_not_called()


@pytest.mark.parametrize("operation", ["get_runtime_store", "delete_runtime_store"])
def test_managed_store_already_absent_allows_app_deletion(monkeypatch, operation):
    client = mock.Mock()
    client.get_runtime_store.return_value = _runtime_store_response()
    getattr(client, operation).side_effect = AgentCliError("absent", error_code="NOT_FOUND")
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code == 0, result.output
    cli.assert_called_once()


def test_managed_delete_rejects_a_different_owner(monkeypatch):
    client = mock.Mock()
    client.get_runtime_store.return_value = _runtime_store_response(sp="different-sp")
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code != 0
    client.delete_runtime_store.assert_not_called()
    cli.assert_not_called()


def test_managed_delete_rejects_an_unexpected_resource_name(monkeypatch):
    client = mock.Mock()
    resource = _runtime_store_response()
    resource["name"] = "runtime-stores/other-app"
    client.get_runtime_store.return_value = resource
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code != 0
    client.delete_runtime_store.assert_not_called()
    cli.assert_not_called()


def test_managed_delete_cannot_skip_cleanup_when_identity_lookup_fails(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda *args: None)
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=mock.Mock())

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code != 0
    assert "deployment was retained" in result.output
    ctx.client.assert_not_called()
    cli.assert_not_called()


def test_legacy_delete_preserves_existing_behavior(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", False)
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=mock.Mock())

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code == 0, result.output
    ctx.client.assert_not_called()
    cli.assert_called_once_with(
        ["apps", "delete", "mason-myapp"],
        "prof",
        action="Could not delete deployment 'mason-myapp'.",
    )
