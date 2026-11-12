"""Environment commands exercise real request building with mocked platform boundaries."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from databricks.sdk.errors import NotFound

from databricks_agentbricks.cli import environment_client as client_mod
from databricks_agentbricks.cli.app import agentbricks
from databricks_agentbricks.environment_source import project_files
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke.transport import EndpointResponse


@pytest.fixture
def setup_cli(tmp_path, monkeypatch):
    source = tmp_path / "agent"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname="test"\n')
    (source / "uv.lock").write_text("version=1\n")
    (source / "agent.py").write_text("def invoke(request, state_dir): return request\n")
    (source / ".env").write_text("PRIVATE=do-not-upload")
    (source / "private.pem").write_text("do-not-upload")
    (source / ".venv").mkdir()
    (source / ".venv" / "skip").write_text("do-not-upload")
    workspace = MagicMock()
    workspace.config.host = "https://workspace.example"
    workspace.apps.get.return_value.url = "https://launcher.example"
    workspace.current_user.me.return_value.user_name = "me@example.com"
    workspace.workspace.get_status.side_effect = [
        NotFound("new folder"),
        SimpleNamespace(object_id=42),
    ]
    monkeypatch.setattr(client_mod, "_workspace_client", lambda profile: workspace)
    monkeypatch.setattr(
        client_mod.WorkspaceOAuthAuthenticator, "authorization_header", lambda self: "Bearer oauth"
    )
    requests = []
    replies = {}

    class Transport:
        def send(self, request):
            requests.append(request)
            path = request.url.removeprefix("https://launcher.example")
            default = {
                ("GET", "/api/config"): (
                    200,
                    {"workspace": workspace.config.host, "service_principal": "app-sp"},
                ),
                ("GET", "/api/environments"): (200, [{"name": "demo"}]),
                ("POST", "/api/environments/demo/runs"): (
                    202,
                    {"id": "run", "status": "queued", "stage": "Queued"},
                ),
                ("GET", "/api/runs/run"): (
                    200,
                    {"id": "run", "status": "succeeded", "result": {"reply": "ok"}},
                ),
            }
            status, body = replies.get(
                (request.method, path), default.get((request.method, path), (500, {}))
            )
            return EndpointResponse(request.url, status, {}, body, 0.01)

    monkeypatch.setattr(client_mod, "HttpSession", Transport)
    return source, workspace, requests, replies


def run(*args):
    return CliRunner().invoke(
        agentbricks, ["--profile", "test", "-o", "json", "environment", *args]
    )


def test_invoke_polls_background_run_and_sends_retry_identity(setup_cli):
    _, _, requests, _ = setup_cli
    result = run(
        "invoke",
        "demo",
        "--app",
        "launcher",
        "--session-id",
        "s",
        "--invocation-id",
        "i",
        "--retry",
        "--json",
        '{"message":"hi"}',
    )
    assert result.exit_code == 0, result.output
    assert requests[0].body == {
        "session_id": "s",
        "invocation_id": "i",
        "input": {"message": "hi"},
        "retry": True,
    }
    assert requests[0].headers["X-Routing-Key"] == "s"
    assert requests[-1].url.endswith("/api/runs/run")
    assert json.loads(result.output)["result"]["reply"] == "ok"


def test_no_wait_returns_run_id_without_polling(setup_cli):
    _, _, requests, _ = setup_cli
    result = run(
        "invoke",
        "demo",
        "--url",
        "https://launcher.example",
        "--session-id",
        "s",
        "--json",
        "{}",
        "--no-wait",
    )
    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert "Authorization" not in requests[0].headers
    assert json.loads(result.output)["status"] == "queued"


def test_failed_job_is_a_nonzero_cli_error_with_new_invocation_guidance(setup_cli):
    _, _, _, replies = setup_cli
    replies["GET", "/api/runs/run"] = (
        200,
        {"id": "run", "status": "failed", "error": "uv install failed"},
    )
    result = run(
        "invoke",
        "demo",
        "--app",
        "launcher",
        "--session-id",
        "s",
        "--invocation-id",
        "i",
        "--json",
        "{}",
    )
    assert result.exit_code != 0
    assert "uv install failed" in result.output
    assert "new --invocation-id" in result.output


@pytest.mark.parametrize("value", ["[]", "null", "{"])
def test_invalid_input_never_contacts_launcher(setup_cli, value):
    _, workspace, requests, _ = setup_cli
    result = run("invoke", "demo", "--app", "launcher", "--session-id", "s", "--json", value)
    assert result.exit_code != 0
    assert requests == []
    workspace.apps.get.assert_not_called()


def test_retry_requires_original_invocation_id(setup_cli):
    result = run(
        "invoke", "demo", "--app", "launcher", "--session-id", "s", "--json", "{}", "--retry"
    )
    assert result.exit_code != 0
    assert "original --invocation-id" in result.output


def test_project_rejects_missing_lock_and_symlink(setup_cli):
    source, _, _, _ = setup_cli
    (source / "uv.lock").unlink()
    with pytest.raises(AgentCliError, match="uv.lock"):
        project_files(source)
    (source / "uv.lock").symlink_to(source / "pyproject.toml")
    with pytest.raises(AgentCliError, match="symlink"):
        project_files(source)


def test_polling_timeout_preserves_invocation_id_and_does_not_cancel(setup_cli, monkeypatch):
    from databricks_agentbricks.cli import environment as environment_mod

    _, _, requests, _ = setup_cli
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(environment_mod, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    result = run(
        "invoke",
        "demo",
        "--app",
        "launcher",
        "--session-id",
        "s",
        "--invocation-id",
        "i",
        "--json",
        "{}",
        "--timeout",
        "0.1",
    )
    assert result.exit_code != 0
    assert "Timed out waiting for run run" in result.output
    assert "--invocation-id i" in result.output
    assert len(requests) == 1


def test_project_source_limit_is_checked_before_upload(setup_cli, monkeypatch):
    from databricks_agentbricks import environment_source

    source, workspace, _, _ = setup_cli
    monkeypatch.setattr(environment_source, "MAX_BYTES", 1)
    with pytest.raises(AgentCliError, match="20 MiB"):
        project_files(source)
    workspace.workspace.mkdirs.assert_not_called()


def test_lost_submission_response_reports_generated_id_for_safe_recovery(setup_cli, monkeypatch):
    from databricks_agentbricks.cli import environment as environment_mod

    _, _, requests, _ = setup_cli
    monkeypatch.setattr(environment_mod, "uuid4", lambda: "generated-call-id")

    class BrokenTransport:
        def send(self, request):
            requests.append(request)
            raise AgentCliError("Connection lost after submission")

    monkeypatch.setattr(client_mod, "HttpSession", BrokenTransport)
    result = run("invoke", "demo", "--app", "launcher", "--session-id", "s", "--json", "{}")
    assert result.exit_code != 0
    assert "--invocation-id generated-call-id" in result.output
    assert requests[0].body["invocation_id"] == "generated-call-id"


def test_environment_create_is_removed_without_contacting_workspace(setup_cli):
    _, workspace, requests, _ = setup_cli
    result = run("create", "demo")
    assert result.exit_code != 0
    assert "unknown command `create`" in result.output
    assert not workspace.mock_calls
    assert not requests
    help_result = run("--help")
    assert "deploy --sandbox" in help_result.output
    assert "create" not in agentbricks.commands["environment"].commands
