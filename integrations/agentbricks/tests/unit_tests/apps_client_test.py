"""Unit tests for AppsClient: the Apps CLI accessor and its injectable runner."""

from __future__ import annotations

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
    assert "`agentbricks login --profile prof`" in hint


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


@pytest.mark.parametrize(
    "call",
    [
        lambda client: client.sync_source("app", "src", "/Workspace/dst"),
        lambda client: client.deploy("app", "/Workspace/dst"),
        lambda client: client.start("app"),
        lambda client: client.stop("app"),
        lambda client: client.delete("app"),
    ],
)
def test_lifecycle_commands_print_to_stderr(call):
    # Their output is progress; stdout is reserved for the command's `--output json` payload.
    calls = []
    call(AppsClient("prof", runner=_runner(calls)))
    assert [kwargs.get("to_stderr") for _, _, kwargs in calls] == [True]


def test_wait_until_deleted_returns_once_the_app_is_gone(monkeypatch):
    answers = iter([0, 0, 1])

    def run(args, profile, **kwargs):
        code = next(answers)
        return types.SimpleNamespace(
            returncode=code,
            stdout="",
            stderr="" if code == 0 else "App does not exist or is deleted.",
        )

    monkeypatch.setattr(apps_client_mod.time, "sleep", lambda s: None)
    AppsClient("prof", runner=run).wait_until_deleted("app")


def test_wait_until_deleted_times_out(monkeypatch):
    monkeypatch.setattr(apps_client_mod.time, "sleep", lambda s: None)
    with pytest.raises(AgentCliError, match="still being deleted"):
        AppsClient("prof", runner=_runner([])).wait_until_deleted("app", timeout_s=0)


@pytest.mark.parametrize("key", ["active_deployment", "pending_deployment"])
def test_recorded_source_code_path_reads_the_latest_deployment(key):
    app = {key: {"source_code_path": "/Workspace/Users/u/agentbricks_deployments/app"}}
    path = apps_client_mod.recorded_source_code_path(app)
    assert path == "/Workspace/Users/u/agentbricks_deployments/app"


def test_recorded_source_code_path_is_none_without_a_deployment():
    assert apps_client_mod.recorded_source_code_path({}) is None


def test_find_returns_the_app():
    client = AppsClient("prof", runner=_runner([], stdout='{"name": "app"}'))
    assert client.find("app") == {"name": "app"}


def test_find_returns_none_only_when_the_app_does_not_exist():
    stderr = "Error: App with name app does not exist or is deleted."
    assert AppsClient("prof", runner=_runner([], returncode=1, stderr=stderr)).find("app") is None


@pytest.mark.parametrize("stderr", ["Error: 401 Unauthorized", "HTTP 503", ""])
def test_find_raises_when_it_cannot_tell(stderr):
    client = AppsClient("prof", runner=_runner([], returncode=1, stderr=stderr))
    with pytest.raises(AgentCliError, match="Could not check whether deployment"):
        client.find("app")


def test_wait_until_deleted_retries_a_check_that_fails(monkeypatch):
    answers = iter([(1, "HTTP 503"), (0, ""), (1, "App does not exist or is deleted.")])

    def run(args, profile, **kwargs):
        code, stderr = next(answers)
        return types.SimpleNamespace(returncode=code, stdout="", stderr=stderr)

    monkeypatch.setattr(apps_client_mod.time, "sleep", lambda s: None)
    AppsClient("prof", runner=run).wait_until_deleted("app")


def test_delete_source_deletes_the_folder_recursively():
    calls = []
    assert AppsClient("prof", runner=_runner(calls)).delete_source("/Workspace/x") is True
    assert calls[0][0] == ["workspace", "delete", "/Workspace/x", "--recursive"]


@pytest.mark.parametrize(
    "stderr", ["Error: Path (/Workspace/x) doesn't exist.", "error: resource_does_not_exist"]
)
def test_delete_source_of_a_missing_folder_is_not_an_error(stderr):
    client = AppsClient("prof", runner=_runner([], returncode=1, stderr=stderr))
    assert client.delete_source("/Workspace/x") is False


def test_delete_source_does_not_read_other_missing_things_as_the_folder():
    stderr = "Error: principal does not exist"
    client = AppsClient("prof", runner=_runner([], returncode=1, stderr=stderr))
    with pytest.raises(AgentCliError, match="Could not delete the uploaded source folder"):
        client.delete_source("/Workspace/x")


def test_delete_source_raises_on_other_failures():
    client = AppsClient("prof", runner=_runner([], returncode=1, stderr="PERMISSION_DENIED"))
    with pytest.raises(AgentCliError, match="Could not delete the uploaded source folder"):
        client.delete_source("/Workspace/x")
