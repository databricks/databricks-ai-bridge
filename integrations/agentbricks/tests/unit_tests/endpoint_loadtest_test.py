from __future__ import annotations

import json
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli import endpoint as endpoint_mod
from databricks_agentbricks.cli.endpoint import endpoint
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke import loadtest as loadtest_mod
from databricks_agentbricks.services.invoke.transport import EndpointRequest


class _Ctx:
    profile = "test-profile"
    output = "json"


@pytest.mark.parametrize("routing_key", [None, "chosen-session"])
def test_loadtest_reuses_app_auth_and_request_with_per_user_routing(monkeypatch, routing_key):
    captured = {}
    apps = Mock()
    apps.get_app_url.return_value = "https://app.example"
    authenticator = Mock()
    authenticator.authorization_header.return_value = "Bearer test-token"
    transport = Mock()
    transport.send.side_effect = AssertionError("Load tests must delegate HTTP to Locust")
    monkeypatch.setattr(endpoint_mod, "AppsClient", lambda profile: apps)
    monkeypatch.setattr(endpoint_mod, "WorkspaceOAuthAuthenticator", lambda profile: authenticator)
    monkeypatch.setattr(endpoint_mod, "HttpSession", lambda: transport)
    monkeypatch.setattr(
        endpoint_mod,
        "run_loadtest",
        lambda request, **kwargs: captured.update(request=request, **kwargs),
    )
    args = (
        "invoke my-agent --path /events --method PATCH --query mode=fast "
        "--json null --timeout 12.5 --sse --loadtest"
    ).split()
    if routing_key:
        args.extend(["--routing-key", routing_key])

    result = CliRunner().invoke(endpoint, args, obj=_Ctx())

    assert result.exit_code == 0, result.output
    apps.get_app_url.assert_called_once_with("agent-bricks-my-agent")
    request = captured.pop("request")
    assert request.url == "https://app.example/events?mode=fast"
    assert request.method == "PATCH"
    assert request.headers["Authorization"] == "Bearer test-token"
    assert request.headers["Content-Type"] == "application/json"
    assert request.body is None
    assert request.body_set is True
    assert request.sse is True
    assert request.timeout == 12.5
    if routing_key:
        assert request.headers["X-Routing-Key"] == routing_key
    else:
        assert uuid.UUID(request.headers["X-Routing-Key"])
    assert captured == {
        "concurrency": 1,
        "duration": 60,
        "output": "json",
        "per_user_routing": not routing_key,
    }


def test_loadtest_url_uses_requested_concurrency_and_duration_without_app_routing(monkeypatch):
    captured = {}

    def run(request, **kwargs):
        captured.update(request=request, **kwargs)

    monkeypatch.setattr(endpoint_mod, "run_loadtest", run)
    context = _Ctx()
    context.output = "text"
    result = CliRunner().invoke(
        endpoint,
        (
            "invoke --url http://localhost:8000 --path /health --method GET "
            "--loadtest --concurrency 4 --duration 3"
        ).split(),
        obj=context,
    )

    assert result.exit_code == 0, result.output
    request = captured.pop("request")
    assert request.url == "http://localhost:8000/health"
    assert request.headers == {}
    assert request.body_set is False
    assert captured == {
        "concurrency": 4,
        "duration": 3,
        "output": "text",
        "per_user_routing": False,
    }


@pytest.mark.parametrize("option", ["--concurrency", "--duration"])
def test_load_settings_require_loadtest_before_resolving_an_endpoint(monkeypatch, option):
    monkeypatch.setattr(
        endpoint_mod,
        "build_invoke_service",
        lambda *args: pytest.fail("Must reject before resolving"),
    )

    result = CliRunner().invoke(endpoint, ["invoke", "--path", "/run", option, "2"], obj=_Ctx())

    assert result.exit_code != 0
    assert "--loadtest" in result.output


@pytest.mark.parametrize("option", ["--concurrency", "--duration"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_load_settings_must_be_positive(option, value):
    result = CliRunner().invoke(
        endpoint,
        ["invoke", "--path", "/run", "--loadtest", option, value],
        obj=_Ctx(),
    )

    assert result.exit_code == 2
    assert option in result.output
    assert "range" in result.output


def _request() -> EndpointRequest:
    return EndpointRequest(
        url="https://app.example/events?mode=fast",
        method="POST",
        headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        body=None,
        timeout=12.5,
        sse=True,
        body_set=True,
    )


def test_missing_locust_gives_install_hint_without_starting_a_process(monkeypatch):
    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: None)
    monkeypatch.setattr(
        loadtest_mod.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not spawn")
    )

    with pytest.raises(AgentCliError) as exc:
        loadtest_mod.run_loadtest(
            _request(), concurrency=1, duration=60, output="text", per_user_routing=False
        )

    assert "locust" in str(exc.value).lower()
    assert "databricks-agentbricks[loadtest]" in (exc.value.hint or "")


@pytest.mark.parametrize("output", ["text", "json"])
def test_locust_receives_private_request_on_stdin_and_native_options(monkeypatch, output):
    captured = {}

    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: object())
    monkeypatch.setattr(loadtest_mod.subprocess, "run", run)
    request = _request()

    loadtest_mod.run_loadtest(
        request,
        concurrency=4,
        duration=3,
        output=output,
        per_user_routing=True,
    )

    command = captured.pop("command")
    assert command[:3] == [sys.executable, "-m", "locust"]
    assert Path(command[command.index("-f") + 1]).is_file()
    for option, expected in (
        ("--users", "4"),
        ("--spawn-rate", "4"),
        ("--run-time", "3s"),
        ("--stop-timeout", "13"),
    ):
        assert command[command.index(option) + 1] == expected
    assert "--headless" in command
    assert "--only-summary" in command
    assert ("--json" in command) is (output == "json")
    assert not any("test-token" in argument for argument in command)
    assert json.loads(captured["input"]) == {
        "request": asdict(request),
        "per_user_routing": True,
    }
    assert captured["text"] is True
    assert not captured.get("capture_output")


@pytest.mark.parametrize("returncode", [1, 2])
def test_locust_failure_status_is_preserved(monkeypatch, returncode):
    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: object())
    monkeypatch.setattr(
        loadtest_mod.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, returncode=returncode),
    )

    result = CliRunner().invoke(
        endpoint,
        "invoke --url http://localhost:8000 --path /run --loadtest".split(),
        obj=_Ctx(),
    )
    assert result.exit_code == returncode
