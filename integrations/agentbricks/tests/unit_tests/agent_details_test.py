"""Exercise the UI gate and real deploy output without contacting a workspace."""

from __future__ import annotations

import json
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

_UI_FLAG = "databricks.fe.customagents.agentCentricView.enable"
_PREVIEW = "agent_memory_sessions"
_APP = "agent-bricks-demo"
_DETAILS_URL = f"https://workspace.example/ml/agents/apps/{_APP}?o=123"


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


def _provider(response, *, workspace_id="123"):
    ui_api = _UiApi(response)
    client = types.SimpleNamespace(
        host="https://workspace.example/",
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
