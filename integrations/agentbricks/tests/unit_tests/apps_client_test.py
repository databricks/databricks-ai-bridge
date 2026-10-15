"""Unit tests for AppsClient: the Apps CLI accessor and its injectable runner."""

from __future__ import annotations

import shlex
import types

import pytest

from databricks_agentbricks.clients import apps_client as apps_client_mod
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError


def _runner(calls, *, returncode=0, stdout="", stderr=""):
    def run(args, profile, **kwargs):
        calls.append((args, profile, kwargs))
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


def test_constructor_stores_profile_and_injected_runner():
    calls = []
    client = AppsClient("prof", runner=_runner(calls))
    client.exists("agent-bricks-myapp")
    assert calls == [
        (["apps", "get", "agent-bricks-myapp"], "prof", {"capture": True, "check": False})
    ]


def test_exists_true_on_zero_returncode():
    calls = []
    client = AppsClient("prof", runner=_runner(calls, returncode=0))
    assert client.exists("agent-bricks-myapp") is True


@pytest.mark.parametrize(
    "stderr",
    [
        "Error: App with name agent-bricks-myapp does not exist or is deleted.",
        "Error: RESOURCE_DOES_NOT_EXIST: app missing",
        "Error: resource_does_not_exist: app missing",
    ],
)
def test_exists_false_on_definitive_not_found(stderr):
    client = AppsClient("prof", runner=_runner([], returncode=1, stderr=stderr))
    assert client.exists("agent-bricks-myapp") is False


@pytest.mark.parametrize(
    "stderr", ["Error: 401 Unauthorized", "connection refused", "", "Error: Not Found"]
)
def test_exists_raises_when_the_probe_fails_for_another_reason(stderr):
    client = AppsClient("prof", runner=_runner([], returncode=1, stderr=stderr))
    with pytest.raises(AgentCliError, match="Could not check whether deployment") as raised:
        client.exists("agent-bricks-myapp")
    hint = raised.value.hint or ""
    assert hint.startswith(stderr or "The `databricks apps get` command failed without a message.")
    assert "`agentbricks auth login prof`" in hint


def test_service_principal_parses_from_json():
    calls = []
    client = AppsClient(
        "prof",
        runner=_runner(calls, stdout='{"service_principal_client_id": "sp-123"}'),
    )
    assert client.get_service_principal("agent-bricks-myapp") == "sp-123"
    assert calls == [
        (
            ["apps", "get", "agent-bricks-myapp", "-o", "json"],
            "prof",
            {"capture": True, "check": False},
        )
    ]


def test_service_principal_none_on_nonzero_returncode():
    client = AppsClient("prof", runner=_runner([], returncode=1, stdout="{}"))
    assert client.get_service_principal("agent-bricks-myapp") is None


def test_service_principal_none_on_invalid_json():
    client = AppsClient("prof", runner=_runner([], stdout="not json"))
    assert client.get_service_principal("agent-bricks-myapp") is None


def test_url_parses_from_json():
    client = AppsClient("prof", runner=_runner([], stdout='{"url": "https://myapp.example"}'))
    assert client.get_app_url("agent-bricks-myapp") == "https://myapp.example"


def test_url_none_when_missing():
    client = AppsClient("prof", runner=_runner([], stdout="{}"))
    assert client.get_app_url("agent-bricks-myapp") is None


def test_url_none_when_empty_string():
    # An empty url in the payload is treated the same as missing.
    client = AppsClient("prof", runner=_runner([], stdout='{"url": ""}'))
    assert client.get_app_url("agent-bricks-myapp") is None


def test_compute_state_parses_nested_field():
    client = AppsClient(
        "prof", runner=_runner([], stdout='{"compute_status": {"state": "ACTIVE"}}')
    )
    assert client.get_compute_state("agent-bricks-myapp") == "ACTIVE"


def test_compute_state_none_on_nonzero_returncode():
    client = AppsClient("prof", runner=_runner([], returncode=1, stdout="{}"))
    assert client.get_compute_state("agent-bricks-myapp") is None


def test_wait_for_running_returns_when_compute_active():
    client = AppsClient(
        "prof", runner=_runner([], stdout='{"compute_status": {"state": "ACTIVE"}}')
    )
    client.wait_for_active("app", timeout_s=1)  # returns without raising


def test_wait_for_running_times_out(monkeypatch):
    client = AppsClient(
        "prof", runner=_runner([], stdout='{"compute_status": {"state": "STARTING"}}')
    )
    monkeypatch.setattr(apps_client_mod.time, "sleep", lambda s: None)  # don't actually wait
    with pytest.raises(AgentCliError):
        client.wait_for_active("app", timeout_s=0)


@pytest.mark.parametrize("profile", [None, "DEFAULT", "profile with spaces"])
def test_deploy_failure_suggests_status_and_logs_with_the_selected_profile(monkeypatch, profile):
    # Exercise the real runner's uncaptured failure path: Apps emits its own diagnostic to
    # the terminal, then Agent Bricks must explain how to investigate it in this workspace.
    from databricks_agentbricks.clients.databricks_cli import _databricks

    calls = []

    def fail_deploy(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return types.SimpleNamespace(returncode=1, stdout=None, stderr=None)

    monkeypatch.setattr(apps_client_mod.subprocess, "run", fail_deploy)
    with pytest.raises(AgentCliError, match="Could not deploy 'agent-bricks-demo'") as raised:
        AppsClient(profile, runner=_databricks).deploy("agent-bricks-demo", "/Workspace/source")

    hint = raised.value.hint or ""
    prefix = ["agentbricks"] + (["--profile", profile] if profile else [])
    assert shlex.join([*prefix, "deployments", "get", "agent-bricks-demo"]) in hint
    assert shlex.join([*prefix, "deployments", "logs", "agent-bricks-demo"]) in hint
    assert "before retrying" in hint
    assert "same project" in hint
    # Preserve the blocking, streaming invocation; do not retry or run extra control-plane calls.
    assert len(calls) == 1
    assert calls[0][0][1:6] == [
        "apps",
        "deploy",
        "agent-bricks-demo",
        "--source-code-path",
        "/Workspace/source",
    ]
    assert calls[0][1]["capture_output"] is False


def test_deploy_failure_preserves_original_message_code_and_detail():
    original = AgentCliError(
        "The deployment request was rejected.",
        error_code="PERMISSION_DENIED",
        hint="Backend detail.",
    )

    def fail_deploy(*args, **kwargs):
        raise original

    with pytest.raises(AgentCliError) as raised:
        AppsClient("DEFAULT", runner=fail_deploy).deploy("agent-bricks-demo", "/Workspace/source")

    assert raised.value.message == original.message
    assert raised.value.error_code == original.error_code
    assert (raised.value.hint or "").startswith("Backend detail.")
    assert "deployments logs agent-bricks-demo" in (raised.value.hint or "")
    assert raised.value.__cause__ is original


def test_successful_deploy_preserves_the_streaming_invocation():
    calls = []
    AppsClient("DEFAULT", runner=_runner(calls)).deploy("agent-bricks-demo", "/Workspace/source")
    assert calls == [
        (
            ["apps", "deploy", "agent-bricks-demo", "--source-code-path", "/Workspace/source"],
            "DEFAULT",
            {"action": "Could not deploy 'agent-bricks-demo'."},
        )
    ]
