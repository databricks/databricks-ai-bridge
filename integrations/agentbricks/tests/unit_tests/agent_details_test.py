"""Exercise the UI gate and real deploy output without contacting a workspace."""

from __future__ import annotations

import json
import os
import threading
import types

import pytest
from click.testing import CliRunner
from deploy_test import _DeployApi, _DeployRunner, _real_deploy_context
from rich.console import Console

from databricks_agentbricks import errors
from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.cli import telemetry
from databricks_agentbricks.clients import agent_details_client
from databricks_agentbricks.clients.agent_details_client import AgentDetailsClient
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.group import AgentBricksCommand
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.services.deployment import deploy_state

_UI_FLAG = "databricks.fe.customagents.agentCentricView.enable"
_PREVIEW = "agent_memory_sessions"
_APP = "agent-bricks-demo"
_DETAILS_URL = f"https://workspace.example/ml/agents/details/{_APP}?o=123"


@pytest.mark.parametrize("background", [False, True])
def test_printed_deploy_link_matches_the_emitted_command(tmp_path, monkeypatch, background):
    from urllib.parse import parse_qs, urlsplit

    monkeypatch.setenv("AGENTBRICKS_DISABLE_TELEMETRY", "0")
    monkeypatch.setattr(errors, "_OUTPUT_MODE", "json")
    AgentProject.create(tmp_path, framework="langgraph", server="custom").write()
    events = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))
    monkeypatch.setattr(deploy_mod, "spawn_background_deploy", lambda *a, **kw: 2**22 + 1)
    api = _DeployApi(events)
    provider, _ = _provider({"flags": {_UI_FLAG: "true", _PREVIEW: "true"}})
    api.workspace_client = provider.get().workspace_client
    logs = []

    def capture(ctx, **kwargs):
        logs.append(telemetry._build_log(ctx, **kwargs).to_dict())

    monkeypatch.setattr(telemetry, "emit_command", capture)
    command = AgentBricksCommand(
        name="deploy", callback=deploy_mod.deploy.callback, params=deploy_mod.deploy.params
    )
    args = ["demo", "--source", str(tmp_path), "--pip-index-url", ""]
    if background:
        args.append("--no-wait")
    result = CliRunner().invoke(command, args, obj=_real_deploy_context(api, output="json"))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    (log,) = logs
    params = parse_qs(urlsplit(payload["details_url"]).query)
    assert params["source"] == ["agentbricks-cli"]
    assert params["cli_invocation_id"] == [log["invocation_id"]]
    assert log["details_link_emitted"] is True
    assert log["background_child"] is False
    assert log["deployment_outcome"] == (
        "DEPLOYMENT_OUTCOME_RUNNING" if background else "DEPLOYMENT_OUTCOME_SUCCEEDED"
    )


class _UiApi:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def do(self, method, path, *, query):
        self.requests.append((method, path, query))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _provider(response, *, host="https://workspace.example/", workspace_id="123"):
    ui_api = _UiApi(response)
    client = types.SimpleNamespace(
        host=host,
        workspace_client=types.SimpleNamespace(
            config=types.SimpleNamespace(workspace_id=workspace_id), api_client=ui_api
        ),
    )
    return types.SimpleNamespace(get=lambda: client), ui_api


@pytest.mark.parametrize(
    "snapshot",
    [
        {"flags": {_UI_FLAG: "true", _PREVIEW: "true"}},
        {"flags": {_UI_FLAG: True, _PREVIEW: True}},
        {"flags": {_PREVIEW: "true"}, "flag_items": {_UI_FLAG: {"valueJsonStr": "true"}}},
    ],
)
def test_both_gates_enable_workspace_scoped_link(snapshot):
    provider, api = _provider(snapshot)
    assert AgentDetailsClient(provider).get_url(_APP) == _DETAILS_URL
    assert api.requests == [
        ("GET", "/api/2.0/ui-flags", {"shouldHashAndEncodeFlagItems": "false", "o": "123"})
    ]


@pytest.mark.parametrize(
    "snapshot",
    [
        {},
        None,
        {"flags": {}},
        {"flags": {_UI_FLAG: "true", _PREVIEW: "false"}},
        {"flags": {_UI_FLAG: "false", _PREVIEW: "true"}},
        {"flags": {_UI_FLAG: "true", _PREVIEW: 1}},
        {"flags": {_PREVIEW: "true"}, "flag_items": {_UI_FLAG: None}},
        {
            "flags": {_UI_FLAG: "true", _PREVIEW: "true"},
            "flag_items": {_UI_FLAG: {"valueJsonStr": "false"}},
        },
        RuntimeError("workspace unavailable"),
    ],
)
def test_disabled_missing_or_unavailable_gates_omit_link(snapshot):
    provider, _ = _provider(snapshot)
    assert AgentDetailsClient(provider).get_url(_APP) is None


def test_workspace_host_without_explicit_workspace_id():
    provider, _ = _provider({"flags": {_UI_FLAG: "true", _PREVIEW: "true"}}, workspace_id=None)
    assert AgentDetailsClient(provider).get_url(_APP) == _DETAILS_URL.split("?")[0]


def test_slow_lookup_does_not_hold_up_output(monkeypatch):
    release = threading.Event()
    provider, api = _provider({})

    def blocked_request(*args, **kwargs):
        release.wait(5)
        return {}

    api.do = blocked_request
    monkeypatch.setattr(agent_details_client, "_DETAILS_LOOKUP_TIMEOUT_S", 0.01)
    try:
        assert AgentDetailsClient(provider).get_url(_APP) is None
    finally:
        release.set()


@pytest.mark.parametrize("exists", [False, True], ids=["first-deploy", "redeploy"])
@pytest.mark.parametrize("output", ["text", "json"])
@pytest.mark.parametrize("enabled", [False, True])
def test_deploy_and_wait_output_follow_gate(tmp_path, monkeypatch, exists, output, enabled):
    monkeypatch.setattr(errors, "_OUTPUT_MODE", output)
    monkeypatch.setattr(render, "_stdout", Console(width=160))
    AgentProject.create(tmp_path, framework="langgraph", server="custom").write()
    events = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events, exists=exists))
    api = _DeployApi(events)
    provider, _ = _provider({"flags": {_UI_FLAG: str(enabled).lower(), _PREVIEW: "true"}})
    api.workspace_client = provider.get().workspace_client
    ctx = _real_deploy_context(api, output=output)
    deployed = CliRunner().invoke(
        deploy_mod.deploy, ["demo", "--source", str(tmp_path), "--pip-index-url", ""], obj=ctx
    )
    waited = CliRunner().invoke(
        deploy_mod.deployments_wait, [_APP, "--source", str(tmp_path)], obj=ctx
    )
    for result in (deployed, waited):
        assert result.exit_code == 0, result.output
        if output == "json":
            payload = json.loads(result.stdout)
            assert payload["url"] == "https://agent.example"
            assert payload.get("details_url") == (_DETAILS_URL if enabled else None)
        else:
            assert "https://agent.example" in result.stdout
            assert (_DETAILS_URL in result.stdout) is enabled


@pytest.mark.parametrize("output", ["text", "json"])
def test_background_handoff_includes_link(tmp_path, monkeypatch, output):
    AgentProject.create(tmp_path, framework="langgraph", server="custom").write()
    events = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))
    monkeypatch.setattr(deploy_mod, "spawn_background_deploy", lambda *a, **kw: 2**22 + 1)
    api = _DeployApi(events)
    provider, _ = _provider({"flags": {_UI_FLAG: "true", _PREVIEW: "true"}})
    api.workspace_client = provider.get().workspace_client
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(tmp_path), "--no-wait"],
        obj=_real_deploy_context(api, output=output),
    )
    assert result.exit_code == 0, result.output
    if output == "json":
        payload = json.loads(result.stdout)
        assert payload["status"] == "started"
        assert payload["details_url"] == _DETAILS_URL
    else:
        assert "started in the background" in result.stdout
        assert _DETAILS_URL in result.stdout


@pytest.mark.parametrize("status", [deploy_state.RUNNING, deploy_state.SUCCEEDED])
@pytest.mark.parametrize("output", ["text", "json"])
@pytest.mark.parametrize(
    "host,workspace_id,ui_enabled,preview_enabled,has_link",
    [
        ("https://workspace.example/", "123", True, True, True),
        ("https://other.example/", "123", True, True, False),
        ("https://workspace.example/", "456", True, True, False),
        ("https://workspace.example/", None, True, True, False),
        ("https://workspace.example/", "123", False, True, False),
        ("https://workspace.example/", "123", True, False, False),
    ],
    ids=["same-workspace", "changed-host", "changed-id", "missing-id", "ui-off", "beta-off"],
)
def test_local_wait_verifies_workspace_and_live_gates(
    tmp_path, monkeypatch, status, output, host, workspace_id, ui_enabled, preview_enabled, has_link
):
    monkeypatch.setattr(errors, "_OUTPUT_MODE", output)
    monkeypatch.setattr(render, "_stdout", Console(width=160))
    AgentProject.create(tmp_path, framework="langgraph", server="custom").write()
    events = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))
    # Keep the handoff record running without spawning a deployment process.
    monkeypatch.setattr(deploy_mod, "spawn_background_deploy", lambda *a, **kw: os.getpid())
    provider, _ = _provider({"flags": {_UI_FLAG: "true", _PREVIEW: "true"}})
    api = _DeployApi(events)
    api.workspace_client = provider.get().workspace_client
    args = ["demo", "--source", str(tmp_path), "--pip-index-url", ""]
    if status == deploy_state.RUNNING:
        args.append("--no-wait")
    deployed = CliRunner().invoke(
        deploy_mod.deploy, args, obj=_real_deploy_context(api, output=output)
    )
    assert deployed.exit_code == 0, deployed.output
    state = deploy_state.read_state(deploy_state.state_path(tmp_path, _APP))
    assert state is not None
    assert state["workspace"] == {"host": "https://workspace.example", "workspace_id": "123"}
    assert state["status"] == status

    wait_provider, ui_api = _provider(
        {"flags": {_UI_FLAG: ui_enabled, _PREVIEW: preview_enabled}},
        host=host,
        workspace_id=workspace_id,
    )
    ctx = types.SimpleNamespace(
        profile="different-profile", output=output, api_client_provider=wait_provider
    )
    result = CliRunner().invoke(
        deploy_mod.deployments_wait, [_APP, "--source", str(tmp_path), "--timeout", "0"], obj=ctx
    )
    assert result.exit_code == (75 if status == deploy_state.RUNNING else 0), result.output
    if output == "json":
        payload = json.loads(result.stdout)
        assert payload["status"] == status
        assert payload["url"] == (
            None if status == deploy_state.RUNNING else "https://agent.example"
        )
        assert payload.get("details_url") == (_DETAILS_URL if has_link else None)
    else:
        assert (_DETAILS_URL in result.stdout) is has_link
        if status == deploy_state.SUCCEEDED:
            assert "https://agent.example" in result.stdout
    matches = host.rstrip("/") == "https://workspace.example" and workspace_id == "123"
    assert len(ui_api.requests) == (1 if matches else 0)


@pytest.mark.parametrize("status", [deploy_state.RUNNING, deploy_state.SUCCEEDED])
@pytest.mark.parametrize(
    "workspace",
    [
        None,
        {},
        {"host": "https://workspace.example"},
        {"host": "https://workspace.example", "workspace_id": None},
    ],
    ids=["old-record", "empty-identity", "host-only", "missing-id"],
)
def test_local_wait_omits_unprovable_links_without_authenticating(tmp_path, status, workspace):
    project = AgentProject.create(tmp_path, framework="langgraph", server="custom")
    project.write()
    state = {
        "status": status,
        "pid": os.getpid(),
        "result": {"url": "https://agent.example", "ready": True},
    }
    if workspace is not None:
        state["workspace"] = workspace
    deploy_state.write_state(deploy_state.state_path(tmp_path, _APP), state)
    auth_attempts = []

    def unexpected_auth():
        auth_attempts.append(True)
        raise AssertionError("local wait authenticated")

    ctx = types.SimpleNamespace(
        profile="other-profile",
        output="json",
        api_client_provider=types.SimpleNamespace(get=unexpected_auth),
    )
    result = CliRunner().invoke(
        deploy_mod.deployments_wait, [_APP, "--source", str(tmp_path), "--timeout", "0"], obj=ctx
    )
    assert result.exit_code == (75 if status == deploy_state.RUNNING else 0), result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == status
    assert payload["url"] == "https://agent.example"
    assert "details_url" not in payload
    assert auth_attempts == []
