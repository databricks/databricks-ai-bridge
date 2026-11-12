"""Publication, App startup, and first-use installation have separate lifecycles."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound
from fastapi.testclient import TestClient

# Reuse the realistic in-memory mapping and command fixture.
from orchestrator_test import setup  # noqa: F401

from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit.sandbox.environment import (
    ARTIFACT_FORMAT,
    PUBLISHED_FORMAT,
    SHARED_FORMAT,
    bind_environment,
    saved_bundle,
)
from databricks_agentkit.sandbox.store import ExecutionIdentityConflict, Session


def test_legacy_publication_retains_its_shared_sandbox_and_separate_state(setup):
    orchestrator, registry, lakebox, remote = setup
    bind_environment(
        orchestrator,
        {
            "format": SHARED_FORMAT,
            "status": "published",
            "namespace": "test",
            "runtime_key": "environment:published",
        },
    )
    lakebox.workspace.current_user.me.return_value.user_name = "app-sp"
    original = remote.run.side_effect
    installed = []

    def run(command, *a, **kw):
        if command.startswith("if test") and installed:
            return installed[0]
        output = original(command, *a, **kw)
        if command.startswith("python3 -c"):
            installed.append(output)
        return output

    remote.run.side_effect = run
    one = orchestrator.invoke("one", "a", {})
    two = orchestrator.invoke("two", "b", {})
    assert one["sandbox_created"] and not one["environment_reused"]
    assert not two["sandbox_created"] and two["environment_reused"]
    assert one["sandbox_id"] == two["sandbox_id"]
    assert len(installed) == 1 and lakebox.create.call_count == 1
    assert registry.record.owner == "app-sp"
    workers = [c.args[0] for c in remote.run.call_args_list if c.args[0].startswith("cd ")]
    assert len(set(workers)) == 2


def test_wrong_owner_fails_before_even_looking_up_a_sandbox(setup):
    orchestrator, registry, lakebox, _ = setup
    orchestrator.environment_session_id = "environment:published"
    registry.record.owner = "local-user"
    lakebox.workspace.current_user.me.return_value.user_name = "app-sp"
    with pytest.raises(ExecutionIdentityConflict, match="belongs"):
        orchestrator.invoke("s", "i", {})
    lakebox.get.assert_not_called()
    lakebox.find_named.assert_not_called()
    lakebox.create.assert_not_called()


def test_failed_install_retries_on_same_sandbox(setup):
    orchestrator, registry, lakebox, remote = setup
    orchestrator.environment_session_id = "environment:published"
    original = remote.run.side_effect

    def fail(command, *a, **kw):
        if command.startswith("python3 -c"):
            raise RuntimeError("dependency install failed")
        return original(command, *a, **kw)

    remote.run.side_effect = fail
    with pytest.raises(RuntimeError, match="install failed"):
        orchestrator.invoke("s", "i", {})
    assert registry.record.get()["status"] == "failed"
    remote.run.side_effect = original
    result = orchestrator.invoke("s", "retry", {})
    assert result["sandbox_id"] == "sandbox-one"
    assert lakebox.create.call_count == 1


def test_persistent_owner_conflict_does_not_overwrite_the_owner():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = {"execution_principal": "first-owner"}
    with pytest.raises(ExecutionIdentityConflict, match="first-owner"):
        Session(db, "namespace", "key").bind_owner("second-owner")
    assert "COALESCE(sessions.execution_principal" in db.execute.call_args_list[0].args[0]


def test_saved_bundle_rejects_a_mismatched_publication():
    control = MagicMock()
    control.agent.return_value = {"source_hash": "other"}
    with pytest.raises(ValueError, match="saved bundle"):
        saved_bundle(control, {"agent_id": "agent", "source_hash": "published"})


def test_app_startup_does_not_access_or_claim_a_sandbox(tmp_path, monkeypatch):
    import databricks_agentkit.sandbox.app as module
    from databricks_agentkit.sandbox.bundle import bundle_files

    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"lock"})
    manifest = {
        "format": PUBLISHED_FORMAT,
        "status": "published",
        "namespace": "test",
        "runtime_key": "environment:test",
        "agent_id": "agent",
        "source_hash": bundle.digest,
        "name": "test",
        "entrypoint": "agent:invoke",
    }
    binding = tmp_path / "environment.json"
    binding.write_text(json.dumps(manifest))
    monkeypatch.setenv("SANDBOX_NAMESPACE", "test")
    monkeypatch.setenv("AGENTBRICKS_ENVIRONMENT_FILE", str(binding))
    registry, lakebox, control, runs = [MagicMock() for _ in range(4)]
    control.agent.return_value = {
        "bundle_data": bundle.data,
        "source_hash": bundle.digest,
        "file_count": 1,
    }
    for name, mock in [
        ("WorkspaceClient", MagicMock()),
        ("Registry", registry),
        ("Lakebox", lakebox),
        ("ControlStore", control),
        ("RunManager", runs),
        ("EnvironmentStore", MagicMock()),
    ]:
        monkeypatch.setattr(module, name, lambda *a, _mock=mock, **kw: _mock)
    with TestClient(module.app) as client:
        assert client.get("/health").status_code == 200
        config = client.get("/api/config").json()
        assert config["environment_mode"] == "on-demand"
        assert config["workbench"]["sessions"]["mode"] == "custom"
        assert not config["workbench"]["tracing"]["configured"]
    lakebox.create.assert_not_called()
    lakebox.get.assert_not_called()
    lakebox.connect.assert_not_called()
    registry.session.assert_not_called()
    runs.start.assert_called_once()
    # Avoid leaking the real lifespan state into API tests that bypass startup.
    module.app.state.preparations = None
    module.app.state.prepared_environment = None


@pytest.mark.parametrize("manifest_format", [ARTIFACT_FORMAT, PUBLISHED_FORMAT, SHARED_FORMAT])
def test_deployment_rejects_a_locally_bound_environment_before_app_mutation(
    monkeypatch, manifest_format
):
    from databricks_agentbricks.cli import prepared_deploy

    workspace = MagicMock()
    workspace.apps.get.side_effect = NotFound("new app")
    registry, control, orchestrator = MagicMock(), MagicMock(), MagicMock()
    orchestrator.namespace = "test"
    registry.runtime.return_value = {"execution_principal": "local-user", "sandbox": None}
    registry.execution_principals.return_value = ["local-user"]
    manifest = {
        "format": manifest_format,
        "status": "published",
        "namespace": "test",
        "runtime_key": "environment:test",
    }
    monkeypatch.setattr(prepared_deploy, "read_manifest", lambda *a: manifest)
    monkeypatch.setattr(prepared_deploy, "_workspace_client", lambda *a: workspace)
    monkeypatch.setattr(
        prepared_deploy, "services", lambda *a: (registry, control, None, orchestrator)
    )
    monkeypatch.setattr(prepared_deploy, "saved_bundle", lambda *a: object())
    with pytest.raises(AgentCliError, match="belongs to local-user"):
        prepared_deploy.deploy_prepared(SimpleNamespace(profile="DEFAULT"), "new", "env")
    workspace.apps.create.assert_not_called()
    workspace.apps.create_update.assert_not_called()
    orchestrator.invoke.assert_not_called()


@pytest.mark.parametrize("manifest_format", [ARTIFACT_FORMAT, PUBLISHED_FORMAT, SHARED_FORMAT])
@pytest.mark.parametrize(
    "workspace_path", [None, "/Users/publisher/custom-app", "/Workspace/Users/publisher/custom-app"]
)
def test_new_app_deployment_packages_only_orchestrator_and_never_touches_compute(
    monkeypatch, manifest_format, workspace_path
):
    from databricks_agentbricks.cli import prepared_deploy
    from databricks_agentkit.sandbox.bundle import bundle_files

    workspace, registry, control, orchestrator = [MagicMock() for _ in range(4)]
    workspace.current_user.me.return_value.user_name = "publisher"
    app = SimpleNamespace(
        service_principal_client_id="app-sp",
        compute_status=None,
        url="https://app.example",
        resources=[],
    )
    workspace.apps.get.side_effect = [NotFound("new App"), app]
    workspace.apps.create.return_value.result.return_value = app
    orchestrator.namespace = "test"
    registry.runtime.return_value = {"execution_principal": None, "sandbox": None}
    registry.execution_principals.return_value = []
    manifest = {
        "format": manifest_format,
        "status": "published",
        "namespace": "test",
        "runtime_key": "environment:test",
        "name": "test",
        "source_hash": "digest",
        "manifest_path": "/Users/publisher/env/environment.json",
        "catalog_path": "/Users/publisher/deployment/catalog",
        "lakebase": {
            "endpoint": "projects/p/branches/b/endpoints/e",
            "database_resource": "projects/p/branches/b/databases/d",
        },
    }
    monkeypatch.setattr(prepared_deploy, "read_manifest", lambda *a: manifest)
    monkeypatch.setattr(prepared_deploy, "_workspace_client", lambda *a: workspace)
    monkeypatch.setattr(
        prepared_deploy, "services", lambda *a: (registry, control, None, orchestrator)
    )
    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"lock"})
    monkeypatch.setattr(prepared_deploy, "saved_bundle", lambda *a: bundle)
    stage = MagicMock()
    monkeypatch.setattr(prepared_deploy, "stage_runtime", stage)
    monkeypatch.setattr(
        prepared_deploy.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=200)
    )
    result = prepared_deploy.deploy_prepared(
        SimpleNamespace(profile="DEFAULT"), "new", "env", workspace_path=workspace_path
    )
    assert result["sandbox_id"] is None
    assert result["installation"] == "on session preparation or first invocation"
    workspace.apps.create.assert_called_once()
    workspace.apps.deploy.assert_called_once()
    expected_path = (
        "/Workspace/Users/publisher/custom-app"
        if workspace_path
        else "/Workspace/Users/publisher/agentbricks_deployments/agent-bricks-new"
    )
    assert workspace.apps.deploy.call_args.args[1].source_code_path == expected_path
    registry.grant_app_access.assert_called_once_with("app-sp")
    orchestrator.invoke.assert_not_called()
    orchestrator.prepare.assert_not_called()
    stage.assert_called_once()
    folders = {call.args[0] for call in workspace.workspace.get_status.call_args_list}
    assert folders == {"/Users/publisher/env", manifest["catalog_path"]}
    assert workspace.workspace.update_permissions.call_count == 2
