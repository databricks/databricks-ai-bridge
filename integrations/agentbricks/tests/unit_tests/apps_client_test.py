"""Unit tests for AppsClient: the Apps CLI accessor and its injectable runner."""

from __future__ import annotations

import types

import pytest

from databricks_agentbricks.clients import apps_client as apps_client_mod
from databricks_agentbricks.clients.apps_client import AppsClient, AppState
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


def _sequence_runner(calls, payloads):
    """Serve one `apps get -o json` payload per call, repeating the last one."""

    def run(args, profile, **kwargs):
        calls.append(args)
        stdout = payloads[min(len(calls), len(payloads)) - 1]
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return run


def test_service_principal_miss_is_not_cached():
    calls = []
    client = AppsClient(
        "prof",
        runner=_sequence_runner(calls, ["{}", '{"service_principal_client_id": "sp-123"}']),
    )

    assert client.get_service_principal("agent-bricks-myapp") is None
    assert client.get_service_principal("agent-bricks-myapp") == "sp-123"
    assert client.get_service_principal("agent-bricks-myapp") == "sp-123"
    assert len(calls) == 2  # the hit is cached; the miss was re-read


def test_wait_for_service_principal_polls_until_it_appears(monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr(apps_client_mod.time, "sleep", sleeps.append)
    client = AppsClient(
        "prof",
        runner=_sequence_runner(calls, ["{}", "{}", '{"service_principal_client_id": "sp-123"}']),
    )

    assert client.wait_for_service_principal("agent-bricks-myapp") == "sp-123"
    assert len(calls) == 3
    assert sleeps == [2, 2]


def test_wait_for_service_principal_gives_up_after_its_timeout(monkeypatch):
    clock = iter(range(0, 1000, 30))
    monkeypatch.setattr(apps_client_mod.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(apps_client_mod.time, "sleep", lambda s: None)
    calls = []
    client = AppsClient("prof", runner=_sequence_runner(calls, ["{}"]))

    assert client.wait_for_service_principal("agent-bricks-myapp", timeout_s=60) is None
    assert len(calls) == 2  # the clock reads 30 s after the first miss and 60 s after the second


def test_app_state_reads_compute_deployment_and_seeds_the_principal():
    calls = []
    payload = (
        '{"compute_status": {"state": "STOPPED"}, "active_deployment": {"deployment_id": "d"},'
        ' "service_principal_client_id": "sp-123"}'
    )
    client = AppsClient("prof", runner=_runner(calls, stdout=payload))

    assert client.app_state("agent-bricks-myapp") == AppState(
        compute_state="STOPPED", has_active_deployment=True, service_principal_client_id="sp-123"
    )
    assert client.get_service_principal("agent-bricks-myapp") == "sp-123"
    assert calls == [
        (
            ["apps", "get", "agent-bricks-myapp", "-o", "json"],
            "prof",
            {"capture": True, "check": False},
        )
    ]


def test_app_state_of_a_never_deployed_app():
    client = AppsClient("prof", runner=_runner([], stdout='{"computeStatus": {"state": "ACTIVE"}}'))

    assert client.app_state("agent-bricks-myapp") == AppState("ACTIVE", False, None)


def test_app_state_none_only_on_definitive_not_found():
    missing = AppsClient(
        "prof", runner=_runner([], returncode=1, stderr="Error: app does not exist")
    )
    assert missing.app_state("agent-bricks-myapp") is None

    failing = AppsClient("prof", runner=_runner([], returncode=1, stderr="Error: 401"))
    with pytest.raises(AgentCliError, match="Could not check whether deployment"):
        failing.app_state("agent-bricks-myapp")

    garbled = AppsClient("prof", runner=_runner([], stdout="not json"))
    with pytest.raises(AgentCliError, match="Could not check whether deployment"):
        garbled.app_state("agent-bricks-myapp")


def test_no_wait_calls_pass_the_flag():
    calls = []
    client = AppsClient("prof", runner=_runner(calls))

    client.create("agent-bricks-myapp", None, no_wait=True)
    client.start_no_wait("agent-bricks-myapp")

    assert [call[0] for call in calls] == [
        ["apps", "create", "agent-bricks-myapp", "--no-wait"],
        ["apps", "start", "agent-bricks-myapp", "--no-wait"],
    ]


def test_create_blocks_by_default_for_other_callers():
    calls = []
    AppsClient("prof", runner=_runner(calls)).create("agent-bricks-myapp", None)

    assert calls[0][0] == ["apps", "create", "agent-bricks-myapp"]
