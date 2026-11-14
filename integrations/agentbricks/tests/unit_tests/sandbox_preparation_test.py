"""Preparation and serving must have distinct lifecycle and failure boundaries."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli.app import agentbricks
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.orchestrator import Orchestrator


@pytest.fixture
def runtime(tmp_path):
    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"locked"})
    row = {"source_hash": bundle.digest, "sandbox_id": "sandbox-one", "status": "ready"}
    registry = MagicMock()

    @contextmanager
    def session(key):
        record = MagicMock()
        record.key = "unused"
        record.get.return_value = row
        yield record

    registry.session.side_effect = session
    lakebox = MagicMock()
    lakebox.transport = "https"
    lakebox.get.return_value = {"sandboxId": "sandbox-one", "status": "Running"}
    remote = lakebox.connect.return_value.__enter__.return_value
    ready = {"python": "/source/.venv/bin/python", "timings_ms": {"dependency_install": 42}}

    def run(command, *args, **kwargs):
        if command.startswith('printf "%s"'):
            return "/home/sandbox"
        if command.startswith("if test") or command.startswith("python3 -c"):
            return json.dumps(ready)
        return '{"reply":"hello"}'

    remote.run.side_effect = run
    return (
        Orchestrator(registry, lakebox, tmp_path, "test", worker_mode="oneshot"),
        bundle,
        row,
        remote,
    )


def test_prepare_installs_without_invoking_the_agent(runtime):
    orchestrator, bundle, _, remote = runtime
    original = remote.run.side_effect
    remote.run.side_effect = lambda command, *args, **kwargs: (
        "" if command.startswith("if test") else original(command, *args, **kwargs)
    )
    result = orchestrator.prepare("prepare", bundle=bundle, entrypoint="agent:invoke")
    commands = [call.args[0] for call in remote.run.call_args_list]
    assert any("source.tar.gz" in command and command.startswith("mkdir") for command in commands)
    assert any(command.startswith("python3 -c") for command in commands)
    assert not any(command.startswith("cd ") for command in commands)
    assert "worker_process_and_execution" not in result["timings_ms"]
    assert result["result"] is None


def test_shared_prepared_environment_keeps_session_state_separate(runtime):
    orchestrator, bundle, _, remote = runtime
    orchestrator.prepared_session_id = "environment:shared"
    for session in ["one", "two"]:
        result = orchestrator.invoke(session, "call", {}, bundle=bundle, entrypoint="agent:invoke")
        assert result["environment_reused"]
    worker_commands = [
        call.args[0] for call in remote.run.call_args_list if call.args[0].startswith("cd ")
    ]
    assert len(set(worker_commands)) == 2
    assert all(
        call.args in {("environment:shared",), ("invocation:environment:shared",)}
        for call in orchestrator.registry.session.call_args_list
    )
    orchestrator.lakebox.create.assert_not_called()


@pytest.mark.parametrize("missing", ["sandbox", "installation"])
def test_serving_never_silently_rebuilds_a_missing_environment(runtime, missing):
    orchestrator, bundle, _, remote = runtime
    orchestrator.prepared_session_id = "environment:shared"
    if missing == "sandbox":
        orchestrator.lakebox.get.return_value = None
    else:
        remote.run.side_effect = ["/home/sandbox", ""]
    with pytest.raises(RuntimeError, match="prepare a new environment"):
        orchestrator.invoke("one", "call", {}, bundle=bundle, entrypoint="agent:invoke")
    orchestrator.lakebox.create.assert_not_called()
    assert not any(call.args[0].startswith("python3 -c") for call in remote.run.call_args_list)


def test_deploy_refuses_unready_environment_before_mutating_apps(monkeypatch):
    from databricks_agentbricks.cli import prepared_deploy

    workspace = MagicMock()
    monkeypatch.setattr(prepared_deploy, "_workspace_client", lambda profile: workspace)
    monkeypatch.setattr(prepared_deploy, "read_manifest", lambda *args: {"status": "failed"})
    with pytest.raises(AgentCliError, match="ready environment"):
        prepared_deploy.deploy_prepared(SimpleNamespace(profile="test"), "test", "env")
    workspace.apps.create.assert_not_called()
    workspace.apps.deploy.assert_not_called()


def test_environment_deploy_routes_to_orchestrator_with_instance_count(monkeypatch):
    from databricks_agentbricks.cli import prepared_deploy

    deploy = MagicMock(
        return_value={
            "deployment": "agent-bricks-test",
            "url": "https://app",
            "installation": "prepared",
        }
    )
    monkeypatch.setattr(prepared_deploy, "deploy_prepared", deploy)
    result = CliRunner().invoke(
        agentbricks, ["deploy", "test", "--environment", "env", "--instances", "2"]
    )
    assert result.exit_code == 0, result.output
    assert deploy.call_args.args[1:] == ("test", "env")
    assert deploy.call_args.kwargs["instances"] == 2
    assert "Next steps" in result.stdout
    assert "https://app" in result.stdout
    assert "Choose New Chat" not in result.stdout


@pytest.mark.parametrize(
    "options",
    [["--no-wait"], ["--allowed-origin", "https://example.com"], ["--clear-allowed-origins"]],
)
def test_environment_deploy_rejects_unsupported_options_before_deploy(monkeypatch, options):
    from databricks_agentbricks.cli import prepared_deploy

    deploy = MagicMock()
    monkeypatch.setattr(prepared_deploy, "deploy_prepared", deploy)
    result = CliRunner().invoke(agentbricks, ["deploy", "test", "--environment", "env", *options])
    assert result.exit_code != 0
    assert "--environment does not support" in result.output
    deploy.assert_not_called()


@pytest.mark.parametrize("exists", [True, False])
def test_deploy_rejects_other_sandbox_owner_before_mutating_apps(monkeypatch, exists):
    from databricks.sdk.errors import NotFound

    from databricks_agentbricks.cli import prepared_deploy

    workspace = MagicMock()
    workspace.apps.get.return_value.service_principal_client_id = "app-principal"
    if not exists:
        workspace.apps.get.side_effect = NotFound("not created")
    monkeypatch.setattr(prepared_deploy, "_workspace_client", lambda profile: workspace)
    monkeypatch.setattr(
        prepared_deploy,
        "read_manifest",
        lambda *args: {
            "status": "ready",
            "execution_identity": {"principal": "developer@example.com"},
        },
    )
    with pytest.raises(AgentCliError, match="Prepared sandbox belongs to"):
        prepared_deploy.deploy_prepared(SimpleNamespace(profile="test"), "test", "env")
    workspace.apps.create.assert_not_called()
    workspace.apps.update.assert_not_called()
    workspace.apps.create_update.assert_not_called()
    workspace.apps.deploy.assert_not_called()


@pytest.mark.parametrize("managed", [False, True])
def test_deployment_packages_orchestrator_without_agent_source(tmp_path, monkeypatch, managed):
    from databricks_agentbricks.cli import prepared_deploy

    manifest = {
        "namespace": "ab-env-test",
        "lakebase": {"endpoint": "endpoint", "host": "host", "database": "database"},
    }
    if managed:
        del manifest["lakebase"]
        manifest["runtime_store"] = {
            "name": "runtime-stores/app",
            "owner_app": "app",
            "branch": "projects/p/branches/b",
            "database": "managed-db",
            "username": "app-principal",
        }

    def build(args, **kwargs):
        assert "--wheel" in args
        (tmp_path / "databricks_agentbricks-0.3.0-py3-none-any.whl").write_bytes(b"sdk")

    monkeypatch.setattr(prepared_deploy.subprocess, "run", build)
    prepared_deploy.stage_runtime(
        tmp_path,
        manifest,
        "app-principal",
        index_url="https://pypi.org/simple/",
        sandbox_idle_timeout=1800,
        sandbox_warm_pool_size=2,
    )
    assert {p.name for p in tmp_path.iterdir()} == {
        "databricks_agentbricks-0.3.0-py3-none-any.whl",
        "requirements.txt",
        "environment.json",
        "app.yaml",
    }
    config = json.loads((tmp_path / "app.yaml").read_text())
    assert config["command"] == ["python", "-m", "databricks_agentkit.sandbox"]
    assert {item["name"]: item["value"] for item in config["env"]}[
        "SANDBOX_IDLE_TIMEOUT_SECONDS"
    ] == "1800"
    assert "sandbox" in (tmp_path / "requirements.txt").read_text()
    env = {item["name"]: item["value"] for item in config["env"]}
    assert env["SANDBOX_WARM_POOL_SIZE"] == "2"
    assert int(env["SANDBOX_WARM_POOL_GENERATION"]) > 0
    if managed:
        assert env["DATABRICKS_AGENTBRICKS_RUNTIME_STORE_DATABASE"] == "managed-db"
        assert env["DATABRICKS_AGENTBRICKS_RUNTIME_STORE_USERNAME"] == "app-principal"
        assert not {"PGHOST", "PGDATABASE", "PGUSER", "LAKEBASE_ENDPOINT"}.intersection(env)
    else:
        assert env["PGDATABASE"] == "database"
        assert "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_DATABASE" not in env


def test_runtime_wheel_version_tracks_source_without_modifying_checkout(tmp_path, monkeypatch):
    from pathlib import Path

    import tomli

    from databricks_agentbricks.cli import prepared_deploy

    package = tmp_path / "package"
    source = package / "src/databricks_agentkit/sandbox/static/app.js"
    source.parent.mkdir(parents=True)
    source.write_text("initial UI")
    metadata = package / "pyproject.toml"
    original = '[project]\nname="databricks-agentbricks"\nversion="0.10.0.dev0"\n'
    metadata.write_text(original)
    monkeypatch.setattr(
        prepared_deploy,
        "__file__",
        str(package / "src/databricks_agentbricks/cli/prepared_deploy.py"),
    )
    versions = []

    def build(args, **kwargs):
        snapshot = Path(args[-1])
        assert snapshot != package
        version = tomli.loads((snapshot / "pyproject.toml").read_text())["project"]["version"]
        versions.append(version)
        destination = Path(args[args.index("--out-dir") + 1])
        (destination / f"databricks_agentbricks-{version}-py3-none-any.whl").write_bytes(b"sdk")

    monkeypatch.setattr(prepared_deploy.subprocess, "run", build)
    manifest = {"namespace": "test", "lakebase": {"endpoint": "e", "host": "h", "database": "d"}}
    for name in ["first", "same", "changed"]:
        destination = tmp_path / name
        destination.mkdir()
        if name == "changed":
            source.write_text("updated UI")
        prepared_deploy.stage_runtime(destination, manifest, "principal", index_url="index")
        assert versions[-1] in (destination / "requirements.txt").read_text()
        assert metadata.read_text() == original
    assert versions[0] == versions[1] != versions[2]
    assert all(version.startswith("0.10.0.dev0+sandbox.") for version in versions)


def test_preparation_failure_is_saved_and_cannot_be_published_ready(monkeypatch):
    from databricks_agentbricks import prepared_environment

    bundle = bundle_files({"pyproject.toml": b"project", "uv.lock": b"locked"})
    control, orchestrator = MagicMock(), MagicMock()
    control.agent.return_value = {
        "bundle_data": bundle.data,
        "source_hash": bundle.digest,
        "file_count": 2,
    }
    orchestrator.prepare.side_effect = RuntimeError("dependency installation failed")
    monkeypatch.setattr(
        prepared_environment, "services", lambda *args: (None, control, None, orchestrator)
    )
    states = []
    monkeypatch.setattr(
        prepared_environment, "write_manifest", lambda w, m, **k: states.append(m.copy())
    )
    manifest = {
        "agent_id": "revision",
        "preparation_id": "prep",
        "entrypoint": "agent:invoke",
        "manifest_path": "/env/environment.json",
    }
    with pytest.raises(AgentCliError, match="preparation failed"):
        prepared_environment.prepare(None, manifest)
    assert [state["status"] for state in states] == ["preparing", "failed"]
    assert "dependency installation failed" in states[-1]["error"]
