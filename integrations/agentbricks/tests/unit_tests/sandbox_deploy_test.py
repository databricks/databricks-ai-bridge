"""Deploy publishes immutable source without building or starting sandbox compute."""

import io
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from databricks.sdk.errors import NotFound

from databricks_agentbricks.cli.app import agentbricks
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.sandbox_deploy import (
    SandboxDeployRequest,
    SandboxDeployService,
)
from databricks_agentkit.sandbox.environment import PUBLISHED_FORMAT


@pytest.fixture
def deployment(tmp_path):
    source = tmp_path / "agent"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname="test"\nversion="1"\n')
    (source / "uv.lock").write_text("version = 1\n")
    (source / "agent.toml").write_text('[agent]\ndeployment_name="demo"\n')
    (source / "runtime").mkdir()
    (source / "runtime/sandbox_entry.py").write_text("def invoke(request, state): return {}\n")
    workspace = MagicMock()
    workspace.config.host = "https://workspace"
    workspace.current_user.me.return_value.user_name = "publisher"
    storage = {}

    def download(path, **kwargs):
        if path not in storage:
            raise NotFound(path)
        return io.BytesIO(storage[path])

    def upload(path, contents, **kwargs):
        assert kwargs["overwrite"] is False
        assert path not in storage
        storage[path] = contents

    workspace.workspace.download.side_effect = download
    workspace.workspace.upload.side_effect = upload
    outcome = {
        "deployment": "agent-bricks-demo",
        "url": "https://app",
        "environment": "demo-revision",
        "manifest_path": "/Users/publisher/revision/environment.json",
        "sandbox_id": None,
        "installation": "on session preparation or first invocation",
        "source_hash": "a" * 64,
        "runtime_store": "runtime-stores/agent-bricks-demo",
    }
    deployer = MagicMock(return_value=outcome)
    service = SandboxDeployService(workspace, deployer)
    request = SandboxDeployRequest(name=None, source=source)
    return SimpleNamespace(**locals())


def test_deploy_publishes_source_and_orchestrator_without_compute(deployment):
    d = deployment
    assert d.service.deploy(d.request) == d.outcome
    name, path = d.deployer.call_args.args
    manifest = json.loads(d.storage[path])
    assert name == "agent-bricks-demo"
    assert manifest["format"] == PUBLISHED_FORMAT
    assert manifest["status"] == "published"
    assert manifest["sandbox_scope"] == "session"
    assert manifest["source_path"].startswith("/Users/publisher/agentbricks_deployments/")
    assert manifest["source_path"] + "/uv.lock" in d.storage
    assert "execution_artifact" not in manifest
    assert "execution_identity" not in manifest
    assert not d.workspace.postgres.mock_calls
    assert not d.workspace.apps.mock_calls
    assert not d.workspace.api_client.mock_calls


def test_failed_app_deploy_reuses_identical_publication_on_retry(deployment):
    d = deployment
    d.deployer.side_effect = [RuntimeError("App rollout failed"), d.outcome]
    with pytest.raises(RuntimeError, match="rollout"):
        d.service.deploy(d.request)
    count = d.workspace.workspace.upload.call_count
    first = d.deployer.call_args
    d.service.deploy(d.request)
    assert d.deployer.call_args == first
    assert d.workspace.workspace.upload.call_count == count
    assert len([p for p in d.storage if "/catalog/environments/" in p]) == 1


def test_new_source_preserves_previous_revision_and_namespace(deployment):
    d = deployment
    d.service.deploy(d.request)
    first_path = d.deployer.call_args.args[1]
    first = json.loads(d.storage[first_path])
    original = dict(d.storage)
    (d.source / "runtime/sandbox_entry.py").write_text("def invoke(request, state): return 2\n")
    d.service.deploy(d.request)
    second = json.loads(d.storage[d.deployer.call_args.args[1]])
    assert first["namespace"] == second["namespace"]
    assert first["agent_id"] != second["agent_id"]
    assert first["name"] != second["name"]
    assert all(d.storage[k] == v for k, v in original.items())
    assert len([p for p in d.storage if "/catalog/environments/" in p]) == 2


def test_partial_source_upload_can_resume_without_overwriting(deployment):
    d = deployment
    upload = d.upload
    failed = False

    def interrupt(path, data, **kwargs):
        nonlocal failed
        if path.endswith("uv.lock") and not failed:
            failed = True
            raise TimeoutError("upload interrupted")
        upload(path, data, **kwargs)

    d.workspace.workspace.upload.side_effect = interrupt
    with pytest.raises(TimeoutError):
        d.service.deploy(d.request)
    assert d.storage
    d.deployer.assert_not_called()
    d.service.deploy(d.request)
    d.deployer.assert_called_once()


def test_lost_manifest_upload_reply_is_reconciled(deployment):
    d = deployment

    def lost_reply(path, data, **kwargs):
        d.upload(path, data, **kwargs)
        if path.endswith("environment.json"):
            raise TimeoutError("reply lost")

    d.workspace.workspace.upload.side_effect = lost_reply
    d.service.deploy(d.request)
    assert len([p for p in d.storage if "/catalog/environments/" in p]) == 1
    d.deployer.assert_called_once()


def test_retry_rejects_changed_publication_metadata(deployment):
    d = deployment
    d.service.deploy(d.request)
    path = d.deployer.call_args.args[1]
    manifest = json.loads(d.storage[path])
    manifest["namespace"] = "different"
    d.storage[path] = json.dumps(manifest).encode()
    d.deployer.reset_mock()
    with pytest.raises(AgentCliError, match="metadata does not match"):
        d.service.deploy(d.request)
    d.deployer.assert_not_called()


@pytest.mark.parametrize("problem", ["user-auth", "missing-lock", "missing-entrypoint", "name"])
def test_invalid_input_fails_before_publication(deployment, problem):
    d = deployment
    if problem == "user-auth":
        (d.source / "agent.toml").write_text(
            '[agent]\ndeployment_name="demo"\n[[tools]]\nauth="user"'
        )
    elif problem == "missing-lock":
        (d.source / "uv.lock").unlink()
    elif problem == "missing-entrypoint":
        (d.source / "runtime/sandbox_entry.py").unlink()
    else:
        (d.source / "agent.toml").write_text('[agent]\ndeployment_name="bad/name"')
    with pytest.raises(AgentCliError):
        d.service.deploy(d.request)
    assert not d.storage
    assert not d.workspace.postgres.mock_calls
    d.deployer.assert_not_called()


@pytest.mark.parametrize("output", [None, "json"])
def test_cli_routes_sandbox_deploy_with_git_and_runtime_options(deployment, monkeypatch, output):
    d = deployment
    service = MagicMock()
    service.return_value.deploy.return_value = d.outcome
    monkeypatch.setattr(
        "databricks_agentkit._api_client._workspace_client", lambda profile: d.workspace
    )
    monkeypatch.setattr(
        "databricks_agentbricks.services.sandbox_deploy.SandboxDeployService", service
    )
    result = CliRunner().invoke(
        agentbricks,
        [
            "-p",
            "test",
            *(["--output", output] if output else []),
            "deploy",
            "demo",
            "--sandbox",
            "--source",
            str(d.source),
            "--git-revision",
            "release",
            "--entrypoint",
            "runtime.sandbox_entry:invoke",
            "--instances",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    if output == "json":
        assert json.loads(result.stdout) == d.outcome
    else:
        assert "Next steps" in result.stdout
        assert "https://app" in result.stdout
        assert "Choose New Chat" in result.stdout
        assert "agentbricks -p test invoke agent-bricks-demo" in result.stdout
        assert "agentbricks -p test deployments logs agent-bricks-demo" in result.stdout
        assert result.stdout.rstrip().splitlines()[-1].startswith("Deployed in ")
    request = service.return_value.deploy.call_args.args[0]
    assert (request.name, request.source, request.git_revision, request.instances) == (
        "demo",
        d.source,
        "release",
        2,
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--environment", "old"],
        ["--no-wait"],
        ["--allow-user-scope-update"],
        ["--allowed-origin", "https://example.com"],
        ["--clear-allowed-origins"],
    ],
)
def test_cli_rejects_conflicting_options_before_any_workspace_access(monkeypatch, options):
    client = MagicMock(side_effect=AssertionError("must not access workspace"))
    monkeypatch.setattr("databricks_agentkit._api_client._workspace_client", client)
    result = CliRunner().invoke(agentbricks, ["deploy", "demo", "--sandbox", *options])
    assert result.exit_code != 0
    assert "error:" in result.output.lower()
    client.assert_not_called()


def test_removed_runtime_config_is_rejected_before_workspace_access(monkeypatch):
    client = MagicMock(side_effect=AssertionError("must not access workspace"))
    monkeypatch.setattr("databricks_agentkit._api_client._workspace_client", client)
    monkeypatch.setattr("databricks_agentbricks.cli.environment._workspace_client", client)
    result = CliRunner().invoke(
        agentbricks, ["deploy", "demo", "--sandbox", "--runtime-config", "unused.json"]
    )
    assert result.exit_code == 2
    assert "No such option '--runtime-config'" in result.output
    client.assert_not_called()


def test_managed_publication_keeps_previous_revision_and_namespace(deployment):
    # Captured from this fixture before removing the optional legacy configuration.
    d = deployment
    d.service.deploy(d.request)
    manifest = json.loads(d.storage[d.deployer.call_args.args[1]])
    assert manifest["name"] == "agent-bricks-demo-639ee51996623e1f"
    assert manifest["namespace"] == "ab-deploy-2ac6a0df5689b99436cb"
    assert manifest["agent_id"] == "bd6f1bb13c3ab3dedcaba40d13fb3d43"
    assert manifest["runtime_store"] == {
        "name": "runtime-stores/agent-bricks-demo",
        "owner_app": "agent-bricks-demo",
    }
    assert "lakebase" not in manifest
    assert manifest["storage_version"] == 3


def test_git_deploy_uploads_committed_source_only(deployment):
    d = deployment

    def git(*args):
        return subprocess.run(["git", "-C", str(d.source), *args], check=True, capture_output=True)

    git("init")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "baseline")
    sha = git("rev-parse", "HEAD").stdout.decode().strip()
    (d.source / "runtime/sandbox_entry.py").write_text("uncommitted local change")
    (d.source / "untracked.py").write_text("untracked")
    d.service.deploy(replace(d.request, git_revision="HEAD"))
    manifest = json.loads(d.storage[d.deployer.call_args.args[1]])
    assert sha in json.dumps(manifest["git_source"])
    assert (
        d.storage[manifest["source_path"] + "/runtime/sandbox_entry.py"]
        == b"def invoke(request, state): return {}\n"
    )
    assert not any(path.endswith("untracked.py") for path in d.storage)


@pytest.mark.parametrize("stale_config", [None, "/no-longer-used/runtime.json"])
def test_cli_always_publishes_with_managed_runtime_store(deployment, monkeypatch, stale_config):
    d = deployment
    if stale_config is None:
        monkeypatch.delenv("AGENTBRICKS_SANDBOX_CONFIG", raising=False)
    else:
        monkeypatch.setenv("AGENTBRICKS_SANDBOX_CONFIG", stale_config)
    monkeypatch.setattr("databricks_agentkit._api_client._workspace_client", lambda p: d.workspace)
    monkeypatch.setattr("databricks_agentbricks.cli.prepared_deploy.deploy_prepared", d.deployer)
    result = CliRunner().invoke(
        agentbricks, ["deploy", "demo", "--sandbox", "--source", str(d.source)]
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads(d.storage[d.deployer.call_args.args[2]])
    assert manifest["runtime_store"]["owner_app"] == "agent-bricks-demo"
    assert "lakebase" not in manifest
    assert not d.workspace.postgres.mock_calls


def test_retry_rejects_corrupt_saved_bundle(deployment):
    d = deployment
    d.service.deploy(d.request)
    manifest = json.loads(d.storage[d.deployer.call_args.args[1]])
    archive = f"{manifest['catalog_path']}/sources/{manifest['agent_id']}.tar.gz"
    d.storage[archive] = b"corrupt"
    d.deployer.reset_mock()
    with pytest.raises(ValueError, match="archive digest mismatch"):
        d.service.deploy(d.request)
    d.deployer.assert_not_called()


def test_idle_timeout_is_runtime_configuration_not_source_revision(deployment):
    d = deployment
    d.service.deploy(d.request)
    manifest_path = d.deployer.call_args.args[1]
    d.service.deploy(replace(d.request, sandbox_idle_timeout=1800))
    assert d.deployer.call_args.args[1] == manifest_path
    assert d.deployer.call_args.kwargs["sandbox_idle_timeout"] == 1800


@pytest.mark.parametrize("value", ["0", "59", "86401", "10m"])
def test_deploy_rejects_invalid_sandbox_idle_timeout(value):
    result = CliRunner().invoke(
        agentbricks, ["deploy", "demo", "--sandbox", "--sandbox-idle-timeout", value]
    )
    assert result.exit_code == 2
    assert "sandbox-idle-timeout" in result.output


def test_idle_timeout_requires_sandbox_mode():
    result = CliRunner().invoke(agentbricks, ["deploy", "demo", "--sandbox-idle-timeout", "1800"])
    assert result.exit_code == 2
    assert "require --sandbox" in " ".join(result.output.split())
