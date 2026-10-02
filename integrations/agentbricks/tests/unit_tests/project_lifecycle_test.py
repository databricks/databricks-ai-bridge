"""Status never provisions; cleanup requires creation evidence and current remote ownership."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from databricks.sdk.errors import NotFound

from databricks_agentbricks.agent_project import AgentProject
from databricks_agentbricks.cli import deploy as deploy_commands
from databricks_agentbricks.cli import project as commands
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_resources import load_receipts, record_resource


@pytest.fixture
def project(tmp_path):
    project = AgentProject.create(
        tmp_path,
        framework="openai",
        server="agentbricks",
        memory_store="memory",
        session_store="sessions",
        experiment_name="/Shared/test",
    )
    project.set_deployment_name("demo")
    project.write()
    return tmp_path


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setattr(commands, "_configured_host", lambda profile: "https://test.example")
    client = Mock(host="https://test.example")
    client.workspace_client.apps.get.return_value = SimpleNamespace(
        service_principal_client_id="principal-1", url="https://demo.example", app_status=None
    )
    return SimpleNamespace(profile="test", output="json", client=Mock(return_value=client))


def receipt(project, **kwargs):
    record_resource(
        project,
        host="https://test.example",
        kind="deployment",
        name="agent-bricks-demo",
        resource_id="principal-1",
        created=True,
        managed_runtime=True,
        **kwargs,
    )


def test_status_and_cleanup_preview_never_initialize_auth_or_mutate(project, context):
    receipt(project)
    before = {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}
    runner = CliRunner()
    status = runner.invoke(commands.status, ["--source", str(project)], obj=context)
    assert status.exit_code == 0, status.output
    document = json.loads(status.output)
    assert document["verification"] == "not_checked"
    assert all(r["verification"] == "not_checked" for r in document["resources"])
    preview = runner.invoke(commands.cleanup, ["--source", str(project)], obj=context)
    assert preview.exit_code == 0, preview.output
    plan = json.loads(preview.output)
    assert [r["kind"] for r in plan["resources"] if r["action"] == "delete"] == ["deployment"]
    assert context.client.call_count == 0
    assert before == {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}


def test_verify_retains_partial_results_and_permission_error(project, context, monkeypatch):
    client = context.client()
    client.get_session_store.side_effect = AgentCliError(
        "No access", error_code="PERMISSION_DENIED"
    )
    monkeypatch.setattr(commands, "_resolve_memory_store", lambda *_: {"name": "memory-stores/123"})
    monkeypatch.setattr(commands, "_set_tracking_uri", lambda *_: None)
    monkeypatch.setattr(
        commands, "_mlflow", lambda: SimpleNamespace(get_experiment_by_name=lambda _: None)
    )
    before = (project / "agent.toml").read_bytes()
    result = CliRunner().invoke(
        commands.status, ["--source", str(project), "--verify"], obj=context
    )
    assert result.exit_code == 0, result.output
    by_kind = {r["kind"]: r for r in json.loads(result.output)["resources"]}
    assert by_kind["deployment"]["verification"] == "accessible"
    assert by_kind["memory_store"]["id"] == "memory-stores/123"
    assert by_kind["session_store"]["verification"] == "error"
    assert by_kind["experiment"]["verification"] == "missing"
    client.create_session_store.assert_not_called()
    client.create_memory_store.assert_not_called()
    assert before == (project / "agent.toml").read_bytes()


def test_cancelled_cleanup_deletes_nothing(project, context):
    receipt(project)
    result = CliRunner().invoke(
        commands.cleanup, ["--source", str(project), "--apply"], obj=context, input="n\n"
    )
    assert result.exit_code == 1
    context.client().workspace_client.apps.delete.assert_not_called()
    context.client().delete_runtime_store.assert_not_called()
    assert load_receipts(project)[0]["cleanup"] == "active"


def test_adopted_and_other_workspace_apps_are_never_deleted(project, context):
    record_resource(
        project,
        host="https://test.example",
        kind="deployment",
        name="agent-bricks-demo",
        resource_id="principal-1",
        created=False,
    )
    record_resource(
        project,
        host="https://other.example",
        kind="deployment",
        name="agent-bricks-other",
        resource_id="principal-other",
        created=True,
    )
    result = CliRunner().invoke(
        commands.cleanup, ["--source", str(project), "--apply", "--yes"], obj=context
    )
    assert result.exit_code == 0, result.output
    assert all(r["action"] == "retain" for r in json.loads(result.stdout)["resources"])
    context.client().workspace_client.apps.delete.assert_not_called()


def test_recreated_app_with_same_name_is_retained(project, context):
    receipt(project)
    context.client().workspace_client.apps.get.return_value.service_principal_client_id = (
        "replacement"
    )
    result = CliRunner().invoke(
        commands.cleanup, ["--source", str(project), "--apply", "--yes"], obj=context
    )
    assert result.exit_code == 1
    assert "identity changed" in result.stdout
    context.client().workspace_client.apps.delete.assert_not_called()
    context.client().delete_runtime_store.assert_not_called()


def test_partial_failure_retry_removes_runtime_before_app_and_is_idempotent(
    project, context, monkeypatch
):
    receipt(project)
    client = context.client()
    events = []

    def delete_runtime(*_):
        events.append("runtime")

    monkeypatch.setattr(commands.lakebase_runtime_store, "delete", delete_runtime)
    client.workspace_client.apps.delete.side_effect = [RuntimeError("try again"), None]
    runner = CliRunner()
    args = ["--source", str(project), "--apply", "--yes"]
    first = runner.invoke(commands.cleanup, args, obj=context)
    assert first.exit_code == 1
    assert load_receipts(project)[0]["cleanup"] == "failed"
    second = runner.invoke(commands.cleanup, args, obj=context)
    assert second.exit_code == 0, second.output
    assert load_receipts(project)[0]["cleanup"] == "deleted"
    third = runner.invoke(commands.cleanup, args, obj=context)
    assert third.exit_code == 0, third.output
    assert client.workspace_client.apps.delete.call_count == 2
    assert events == ["runtime", "runtime"]


def test_runtime_failure_retains_app(project, context, monkeypatch):
    receipt(project)
    monkeypatch.setattr(
        commands.lakebase_runtime_store, "delete", Mock(side_effect=AgentCliError("Wrong owner"))
    )
    result = CliRunner().invoke(
        commands.cleanup, ["--source", str(project), "--apply", "--yes"], obj=context
    )
    assert result.exit_code == 1
    context.client().workspace_client.apps.delete.assert_not_called()


def test_missing_app_never_deletes_unvalidated_runtime(project, context):
    receipt(project)
    context.client().workspace_client.apps.get.side_effect = NotFound("gone")
    result = CliRunner().invoke(
        commands.cleanup, ["--source", str(project), "--apply", "--yes"], obj=context
    )
    assert result.exit_code == 0, result.output
    assert "app_missing" in result.stdout
    context.client().delete_runtime_store.assert_not_called()


def test_receipt_reuse_preserves_created_ownership_but_replacement_does_not(project):
    receipt(project)
    record_resource(
        project,
        host="https://test.example",
        kind="deployment",
        name="agent-bricks-demo",
        resource_id="principal-1",
        created=False,
    )
    assert load_receipts(project)[0]["created"] is True
    record_resource(
        project,
        host="https://test.example",
        kind="deployment",
        name="agent-bricks-demo",
        resource_id="principal-2",
        created=False,
    )
    assert load_receipts(project)[0]["created"] is False


@pytest.mark.parametrize(
    "payload", ["[]", "null", '{"schema_version":2}', '{"schema_version":1,"resources":[{}]}']
)
def test_corrupt_receipts_fail_closed(project, context, payload):
    directory = project / ".agentbricks"
    directory.mkdir()
    (directory / "resources.json").write_text(payload)
    result = CliRunner().invoke(
        commands.cleanup, ["--source", str(project), "--apply", "--yes"], obj=context
    )
    assert result.exit_code != 0
    context.client().workspace_client.apps.delete.assert_not_called()


def test_store_receipt_survives_later_provisioning_failure(project, context, monkeypatch):
    monkeypatch.setattr(
        deploy_commands, "_ensure_memory_store", lambda *_: ({"name": "memory-stores/123"}, True)
    )
    monkeypatch.setattr(
        deploy_commands,
        "_ensure_session_store",
        Mock(side_effect=AgentCliError("Service unavailable")),
    )
    with pytest.raises(AgentCliError, match="Service unavailable"):
        deploy_commands._reconcile_declared_stores(
            "memory", "sessions", context.client(), source=project
        )
    stored = load_receipts(project)
    assert len(stored) == 1
    assert stored[0]["kind"] == "memory_store" and stored[0]["created"]


def test_apply_uses_authenticated_workspace_instead_of_configured_host(project, context):
    receipt(project)
    context.client().host = "https://other.example"
    result = CliRunner().invoke(
        commands.cleanup, ["--source", str(project), "--apply", "--yes"], obj=context
    )
    assert result.exit_code == 0, result.output
    assert all(r["action"] == "retain" for r in json.loads(result.stdout)["resources"])
    context.client().workspace_client.apps.delete.assert_not_called()
