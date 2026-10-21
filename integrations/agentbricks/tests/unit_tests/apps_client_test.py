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


# --- busy-App retry ----------------------------------------------------------

_BUSY = (
    "Error: Cannot complete deployment operation for app agent-bricks-myapp. "
    "Another operation is in progress. Please retry in a few moments."
)


class _BusyRunner:
    """Fails the first ``busy`` App-changing calls with the Apps busy error, then succeeds."""

    def __init__(self, *, busy: int, error: str = _BUSY, stream: str = "stderr"):
        self.busy = busy
        self.error = error
        self.stream = stream
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, args, profile, **kwargs):
        args = list(args)
        self.calls.append((args, kwargs))
        if args[:2] == ["apps", "get"]:
            return types.SimpleNamespace(returncode=0, stdout='{"resources": []}', stderr="")
        if self.busy:
            self.busy -= 1
            out = {"stdout": "", "stderr": ""}
            out[self.stream] = self.error
            return types.SimpleNamespace(returncode=1, **out)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    def changes(self, verb: list[str]) -> list[list[str]]:
        return [args for args, _ in self.calls if args[: len(verb)] == verb]


def _busy_client(runner: _BusyRunner):
    notices: list[str] = []
    sleeps: list[float] = []
    client = AppsClient("prof", runner=runner, on_busy=notices.append, sleep=sleeps.append)
    return client, notices, sleeps


def test_deploy_retries_while_the_app_is_busy_then_succeeds():
    # The `apps deploy` output is relayed, so the busy text arrives on the tee'd stdout tail.
    runner = _BusyRunner(busy=2, stream="stdout")
    client, notices, sleeps = _busy_client(runner)

    client.deploy("agent-bricks-myapp", "/Workspace/x")

    assert len(runner.changes(["apps", "deploy"])) == 3  # two busy refusals, one deploy
    assert notices == [
        "App is busy (another operation in progress); retrying in 5s…",
        "App is busy (another operation in progress); retrying in 10s…",
    ]
    assert sleeps == [5, 10]
    assert all(kwargs.get("tee") is True for _, kwargs in runner.calls)


def test_busy_match_is_case_insensitive():
    runner = _BusyRunner(busy=1, error="ANOTHER OPERATION IS IN PROGRESS")
    client, notices, _ = _busy_client(runner)
    client.deploy("agent-bricks-myapp", "/Workspace/x")
    assert len(notices) == 1


def test_persistent_busy_fails_with_the_original_error_and_wait_hint():
    runner = _BusyRunner(busy=100, stream="stdout")
    client, notices, sleeps = _busy_client(runner)

    with pytest.raises(AgentCliError) as raised:
        client.deploy("agent-bricks-myapp", "/Workspace/x")

    assert raised.value.message == "Could not deploy 'agent-bricks-myapp'."
    hint = raised.value.hint or ""
    assert _BUSY in hint
    assert "agentbricks deployments wait agent-bricks-myapp" in hint
    assert "re-run `agentbricks deploy`" in hint
    assert sleeps == [5, 10, 15, 20, 30, 30]
    assert len(notices) == 6
    assert len(runner.changes(["apps", "deploy"])) == 7


def test_other_deploy_errors_are_not_retried():
    runner = _BusyRunner(busy=1, error="Error: app crashed on startup")
    client, notices, sleeps = _busy_client(runner)

    with pytest.raises(AgentCliError) as raised:
        client.deploy("agent-bricks-myapp", "/Workspace/x")

    assert raised.value.hint is None  # the relayed output was already shown live
    assert notices == [] and sleeps == []
    assert len(runner.changes(["apps", "deploy"])) == 1


def test_scale_update_retries_while_busy():
    runner = _BusyRunner(busy=2)
    client, notices, sleeps = _busy_client(runner)

    client.create_update_instances("agent-bricks-myapp", 2)

    assert len(runner.changes(["apps", "create-update"])) == 3
    assert sleeps == [5, 10]
    assert len(notices) == 2


def test_scale_update_other_error_is_not_retried():
    runner = _BusyRunner(busy=1, error="Error: PERMISSION_DENIED")
    client, notices, _ = _busy_client(runner)

    with pytest.raises(AgentCliError, match="Could not update deployment") as raised:
        client.create_update_instances("agent-bricks-myapp", 2)

    assert raised.value.hint == "Error: PERMISSION_DENIED"
    assert notices == []


def test_resource_update_retries_while_busy():
    runner = _BusyRunner(busy=2)
    client, notices, sleeps = _busy_client(runner)

    error = client.reconcile_resources(
        "agent-bricks-myapp", [{"name": "x"}], owned_names=["x"], owned_prefixes=()
    )

    assert error is None
    assert len(runner.changes(["apps", "create-update"])) == 3
    assert sleeps == [5, 10]
    assert len(notices) == 2


def test_resource_update_busy_budget_returns_error_with_hint():
    runner = _BusyRunner(busy=100)
    client, _, _ = _busy_client(runner)

    error = client.reconcile_resources(
        "agent-bricks-myapp", [{"name": "x"}], owned_names=["x"], owned_prefixes=()
    )

    assert error is not None
    assert _BUSY in error
    assert "agentbricks deployments wait agent-bricks-myapp" in error


def test_tool_resource_update_retries_while_busy_but_reads_do_not():
    runner = _BusyRunner(busy=1)
    client, notices, _ = _busy_client(runner)
    resource = {
        "name": "agentbricks-tool-fn",
        "uc_securable": {"securable_type": "FUNCTION", "permission": "EXECUTE"},
    }

    # The fake's verify read returns no resources, so the call ends in a mismatch error; what
    # matters here is that the masked update was retried once and the reads were not.
    client.add_tool_resources_for_rollout("agent-bricks-myapp", [resource])

    assert len(runner.changes(["apps", "create-update"])) == 2
    assert len(notices) == 1
    assert len(runner.changes(["apps", "get"])) == 2


def test_find_returns_payload_or_none_when_missing():
    present = AppsClient("prof", runner=_runner([], stdout='{"name": "agent-bricks-myapp"}'))
    assert present.find("agent-bricks-myapp") == {"name": "agent-bricks-myapp"}

    missing = AppsClient(
        "prof", runner=_runner([], returncode=1, stderr="Error: app does not exist")
    )
    assert missing.find("agent-bricks-myapp") is None

    broken = AppsClient("prof", runner=_runner([], returncode=1, stderr="Error: 401"))
    with pytest.raises(AgentCliError, match="Could not check whether deployment"):
        broken.find("agent-bricks-myapp")
