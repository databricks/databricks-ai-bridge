"""Managed deployments provision through the API, never publisher SQL."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound

from databricks_agentbricks.cli import prepared_deploy
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.environment import PUBLISHED_FORMAT


@pytest.fixture
def deployment(monkeypatch):
    workspace, client = MagicMock(), MagicMock()
    app = SimpleNamespace(
        service_principal_client_id="app-sp",
        compute_status=None,
        url="https://app.example",
        resources=[],
    )
    workspace.apps.get.return_value = app
    workspace.apps.create.return_value.result.return_value = app
    workspace.current_user.me.return_value.user_name = "publisher"
    manifest = {
        "format": PUBLISHED_FORMAT,
        "storage_version": 3,
        "status": "published",
        "namespace": "managed-test",
        "runtime_key": "environment:revision",
        "name": "revision",
        "source_hash": "digest",
        "agent_id": "agent",
        "manifest_path": "/Users/publisher/revisions/rev/environment.json",
        "catalog_path": "/Users/publisher/catalog",
        "runtime_store": {
            "name": "runtime-stores/agent-bricks-demo",
            "owner_app": "agent-bricks-demo",
        },
    }
    client.create_runtime_store.return_value = {
        "name": "runtime-stores/agent-bricks-demo",
        "owner": {"app": {"name": "agent-bricks-demo", "service_principal_id": "app-sp"}},
        "storage_backend": {
            "lakebase": {"branch": "projects/p/branches/b", "database_id": "managed-db"}
        },
    }
    monkeypatch.setattr(prepared_deploy, "_workspace_client", lambda profile: workspace)
    monkeypatch.setattr(prepared_deploy, "_AgentBricksApiClient", lambda **kwargs: client)
    monkeypatch.setattr(prepared_deploy, "read_manifest", lambda *args: manifest)
    monkeypatch.setattr(
        prepared_deploy,
        "saved_bundle",
        lambda *args: bundle_files({"uv.lock": b"lock", "pyproject.toml": b"project"}),
    )
    services = MagicMock(side_effect=AssertionError("Publisher must not connect to SQL or Lakebox"))
    monkeypatch.setattr(prepared_deploy, "services", services)
    stage = MagicMock()
    monkeypatch.setattr(prepared_deploy, "stage_runtime", stage)
    monkeypatch.setattr(
        prepared_deploy.requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200)
    )
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("existing", [False, True])
def test_managed_app_store_creation_and_reuse_need_no_publisher_sql(deployment, existing):
    d = deployment
    if not existing:
        d.workspace.apps.get.side_effect = [NotFound("new"), d.app]
    else:
        d.client.get_runtime_store.return_value = d.client.create_runtime_store.return_value
        d.client.create_runtime_store.side_effect = AgentCliError(
            "exists", error_code="ALREADY_EXISTS"
        )
    result = prepared_deploy.deploy_prepared(SimpleNamespace(profile="test"), "demo", "revision")
    assert result["runtime_store"] == "runtime-stores/agent-bricks-demo"
    d.client.create_runtime_store.assert_called_once_with(
        "agent-bricks-demo", "app-sp", app_name="agent-bricks-demo", retry_transient=True
    )
    deployed = d.stage.call_args.args[1]
    assert deployed["runtime_store"]["username"] == "app-sp"
    assert deployed["runtime_store"]["database"] == "managed-db"
    assert "database" not in d.manifest["runtime_store"]  # Immutable publication not rewritten.
    assert not d.workspace.postgres.mock_calls
    d.services.assert_not_called()
    assert d.stage.call_args.args[2] == "app-sp"
    desired = (
        d.workspace.apps.create_update.call_args.kwargs["app"]
        if existing
        else d.workspace.apps.create.call_args.args[0]
    )
    assert desired.resources == []
    d.workspace.apps.deploy.assert_called_once()


def test_wrong_managed_store_owner_stops_before_runtime_upload_or_rollout(deployment):
    d = deployment
    d.client.create_runtime_store.return_value["owner"]["app"]["service_principal_id"] = (
        "different-sp"
    )
    with pytest.raises(AgentCliError, match="does not belong"):
        prepared_deploy.deploy_prepared(SimpleNamespace(profile="test"), "demo", "revision")
    d.stage.assert_not_called()
    d.workspace.workspace.upload.assert_not_called()
    d.workspace.apps.deploy.assert_not_called()


def test_publication_cannot_be_deployed_under_another_app(deployment):
    d = deployment
    with pytest.raises(AgentCliError, match="different App"):
        prepared_deploy.deploy_prepared(SimpleNamespace(profile="test"), "other", "revision")
    assert not d.workspace.apps.mock_calls
    assert not d.client.mock_calls


def test_managed_store_failure_does_not_fall_back_to_legacy_database(deployment):
    d = deployment
    d.client.create_runtime_store.side_effect = AgentCliError(
        "backend unavailable", error_code="UNAVAILABLE"
    )
    with pytest.raises(AgentCliError, match="backend unavailable"):
        prepared_deploy.deploy_prepared(SimpleNamespace(profile="test"), "demo", "revision")
    assert not d.workspace.postgres.mock_calls
    d.stage.assert_not_called()
    d.workspace.apps.deploy.assert_not_called()
