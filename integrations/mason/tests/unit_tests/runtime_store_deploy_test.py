"""Runtime Store ownership, reuse, and retryable deployment cleanup."""

import json
import types
from unittest import mock

import pytest
from click.testing import CliRunner

from databricks_mason.cli import deploy as deploy_mod
from databricks_mason.errors import AgentCliError


@pytest.fixture(autouse=True)
def _app_identity(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda *args: "sp-123")
    monkeypatch.setenv(deploy_mod._MANAGED_RUNTIME_STORE_ENV, "true")


def _runtime_store_response(store_id, app_name="mason-myapp", sp="sp-123"):
    return {
        "name": f"runtime-stores/{store_id}",
        "owner": {"app": {"name": app_name, "service_principal_id": sp}},
        "storage_backend": {
            "lakebase": {
                "project_id": "databricks-internal-custom-agents",
                "branch": "projects/databricks-internal-custom-agents/branches/production",
                "database_id": "runtime-mason-myapp-550e8400-e29b-41d4-a716-446655440000",
            }
        },
    }


def test_reconcile_runtime_store_creates_with_app_service_principal() -> None:
    client = mock.Mock()
    store_id = deploy_mod.managed_runtime_store.runtime_store_id("mason-myapp", "sp-123")
    client.create_runtime_store.return_value = _runtime_store_response(store_id)

    result = deploy_mod.managed_runtime_store.get_or_create_backend(client, "mason-myapp", "sp-123")

    assert result is not None
    assert result.database == "runtime-mason-myapp-550e8400-e29b-41d4-a716-446655440000"
    client.create_runtime_store.assert_called_once_with(
        store_id, "sp-123", app_name="mason-myapp", retry_transient=True
    )
    client.get_runtime_store.assert_not_called()


def test_reconcile_runtime_store_reuses_on_already_exists() -> None:
    client = mock.Mock()
    client.create_runtime_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    store_id = deploy_mod.managed_runtime_store.runtime_store_id("mason-myapp", "sp-123")
    client.get_runtime_store.return_value = _runtime_store_response(store_id)

    result = deploy_mod.managed_runtime_store.get_or_create_backend(client, "mason-myapp", "sp-123")

    assert result is not None
    assert result.project == "databricks-internal-custom-agents"
    assert result.database == "runtime-mason-myapp-550e8400-e29b-41d4-a716-446655440000"
    assert result.database != store_id
    client.get_runtime_store.assert_called_once_with(store_id)


@pytest.mark.parametrize(
    ("app_name", "sp"), [("mason-other", "sp-123"), ("mason-myapp", "other-sp")]
)
def test_reconcile_runtime_store_rejects_existing_store_owned_by_another_app(app_name, sp):
    client = mock.Mock()
    client.create_runtime_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    store_id = deploy_mod.managed_runtime_store.runtime_store_id("mason-myapp", "sp-123")
    client.get_runtime_store.return_value = _runtime_store_response(store_id, app_name, sp)

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


@pytest.mark.parametrize("legacy_name", [None, ""])
def test_reconcile_and_delete_legacy_store_require_the_same_sp(legacy_name):
    client = mock.Mock()
    client.create_runtime_store.side_effect = AgentCliError("exists", error_code="ALREADY_EXISTS")
    store_id = deploy_mod.managed_runtime_store.runtime_store_id("mason-myapp", "sp-123")
    resource = _runtime_store_response(store_id)
    if legacy_name is None:
        del resource["owner"]["app"]["name"]
    else:
        resource["owner"]["app"]["name"] = legacy_name
    client.get_runtime_store.return_value = resource

    backend = deploy_mod.managed_runtime_store.get_or_create_backend(
        client, "mason-myapp", "sp-123"
    )
    assert backend is not None
    assert backend.database == resource["storage_backend"]["lakebase"]["database_id"]
    deploy_mod.managed_runtime_store.delete(client, "mason-myapp", "sp-123")
    client.delete_runtime_store.assert_called_once_with(store_id)

    client.delete_runtime_store.reset_mock()
    resource["owner"]["app"]["service_principal_id"] = "different-sp"
    with pytest.raises(AgentCliError, match="does not belong to this app identity"):
        deploy_mod.managed_runtime_store.get_or_create_backend(client, "mason-myapp", "sp-123")
    with pytest.raises(AgentCliError, match="does not belong to this app identity"):
        deploy_mod.managed_runtime_store.delete(client, "mason-myapp", "sp-123")
    client.delete_runtime_store.assert_not_called()


def test_delete_runtime_store_finishes_before_deleting_app(monkeypatch):
    client = mock.Mock()
    store_id = deploy_mod.managed_runtime_store.runtime_store_id("mason-myapp", "sp-123")
    client.get_runtime_store.return_value = _runtime_store_response(store_id)
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
        mock.call.runtime.get_runtime_store(store_id),
        mock.call.runtime.delete_runtime_store(store_id),
        mock.call.cli(
            ["apps", "delete", "mason-myapp"],
            "prof",
            action="Could not delete deployment 'mason-myapp'.",
        ),
    ]


@pytest.mark.parametrize("operation", ["get_runtime_store", "delete_runtime_store"])
@pytest.mark.parametrize("error_code", ["PERMISSION_DENIED", "UNAVAILABLE", "FEATURE_DISABLED"])
def test_delete_runtime_store_errors_retain_the_app(monkeypatch, operation, error_code):
    client = mock.Mock()
    store_id = deploy_mod.managed_runtime_store.runtime_store_id("mason-myapp", "sp-123")
    client.get_runtime_store.return_value = _runtime_store_response(store_id)
    getattr(client, operation).side_effect = AgentCliError("cleanup failed", error_code=error_code)
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code != 0
    assert "deployment was retained" in result.output
    assert "Retry" in result.output
    cli.assert_not_called()


@pytest.mark.parametrize("operation", ["get_runtime_store", "delete_runtime_store"])
def test_delete_runtime_store_already_absent_allows_app_deletion(monkeypatch, operation):
    client = mock.Mock()
    store_id = deploy_mod.managed_runtime_store.runtime_store_id("mason-myapp", "sp-123")
    client.get_runtime_store.return_value = _runtime_store_response(store_id)
    getattr(client, operation).side_effect = AgentCliError("absent", error_code="NOT_FOUND")
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code == 0, result.output
    cli.assert_called_once()


@pytest.mark.parametrize(
    ("app_name", "sp"), [("mason-other", "sp-123"), ("mason-myapp", "different-sp")]
)
def test_delete_does_not_remove_store_or_app_for_a_different_owner(monkeypatch, app_name, sp):
    client = mock.Mock()
    store_id = deploy_mod.managed_runtime_store.runtime_store_id("mason-myapp", "sp-123")
    client.get_runtime_store.return_value = _runtime_store_response(store_id, app_name, sp)
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code != 0
    client.delete_runtime_store.assert_not_called()
    cli.assert_not_called()


def test_delete_retries_after_store_deleted_but_app_deletion_failed(monkeypatch):
    client = mock.Mock()
    store_id = deploy_mod.managed_runtime_store.runtime_store_id("mason-myapp", "sp-123")
    client.get_runtime_store.side_effect = [
        _runtime_store_response(store_id),
        AgentCliError("absent", error_code="NOT_FOUND"),
    ]
    cli = mock.Mock(side_effect=[AgentCliError("app deletion failed"), None])
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=lambda: client)

    first = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)
    assert first.exit_code != 0
    retry = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)
    assert retry.exit_code == 0, retry.output
    client.delete_runtime_store.assert_called_once_with(store_id)
    assert cli.call_count == 2


def test_delete_cannot_skip_cleanup_when_app_identity_lookup_fails(monkeypatch):
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda *args: None)
    cli = mock.Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cli)
    ctx = types.SimpleNamespace(profile="prof", output="text", client=mock.Mock())

    result = CliRunner().invoke(deploy_mod.deployments_delete, ["mason-myapp", "--yes"], obj=ctx)

    assert result.exit_code != 0
    assert "deployment was retained" in result.output
    ctx.client.assert_not_called()
    cli.assert_not_called()
