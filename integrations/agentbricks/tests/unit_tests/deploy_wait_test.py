"""CLI tests for waiting on a deploy: clean JSON, the state file, `--no-wait`, and `deployments wait`."""

from __future__ import annotations

import functools
import json
import os
import pathlib
import subprocess
import sys
import types

import pytest
from click.testing import CliRunner
from deploy_test import _DeployApi, _DeployRunner, _real_deploy_context

from databricks_agentbricks import errors
from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.services.deployment import deploy_state

_APP = "agent-bricks-demo"
# Larger than Linux's maximum pid, so no process can have it.
_DEAD_PID = 2**22 + 1


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    AgentProject.create(tmp_path, framework="langgraph", server="custom").write()
    return tmp_path


@pytest.fixture
def json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # The root group sets this from `-o json`; these tests invoke the subcommands directly.
    monkeypatch.setattr(errors, "_OUTPUT_MODE", "json")


def _state(source: pathlib.Path) -> dict:
    state = deploy_state.read_state(deploy_state.state_path(source, _APP))
    assert state is not None
    return state


def _write_state(source: pathlib.Path, **fields) -> None:
    now = deploy_state.timestamp(deploy_state.time.time())
    state = {
        "deployment": _APP,
        "pid": os.getpid(),
        "started_at": now,
        "phase": "Waiting for agent compute to start",
        "phase_started_at": now,
        "status": "running",
        "error": None,
        "log": None,
    }
    state.update(fields)
    deploy_state.write_state(deploy_state.state_path(source, _APP), state)


# --- FR-2 / FR-3: a foreground deploy -------------------------------------------------------


def test_json_deploy_writes_one_document_and_records_the_result(
    project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, json_mode
):
    events: list[tuple] = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(project), "--pip-index-url", ""],
        obj=_real_deploy_context(_DeployApi(events), output="json"),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)  # the whole of stdout is one JSON document
    assert payload["deployment"] == _APP
    # Progress and relayed `apps create` output went to stderr instead.
    assert "• Creating the agent and starting its compute…" in result.stderr
    assert "Agent compute" in result.stderr
    state = _state(project)
    assert state["status"] == "succeeded"
    assert state["result"] == payload
    assert state["pid"] == os.getpid()
    assert state["log"] is None
    assert {"started_at", "phase_started_at", "error"} <= state.keys()


def test_failed_deploy_records_the_error(project: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    events: list[tuple] = []
    runner = _DeployRunner(events)

    def failing(args, profile, **kwargs):
        if list(args[:2]) == ["apps", "deploy"]:
            return types.SimpleNamespace(returncode=1, stdout="Error: app crashed", stderr="")
        return runner(args, profile, **kwargs)

    monkeypatch.setattr(deploy_mod, "_databricks", failing)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(project), "--pip-index-url", ""],
        obj=_real_deploy_context(_DeployApi(events)),
    )

    assert result.exit_code == 1
    state = _state(project)
    assert state["status"] == "failed"
    assert state["error"] == f"Could not deploy '{_APP}'."
    assert state["phase"] == "Deploying the agent source"


# --- FR-4: a second deploy while one is running ---------------------------------------------


@pytest.mark.parametrize("extra", [[], ["--no-wait"]])
def test_deploy_refuses_while_another_deploy_is_running(
    project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
):
    _write_state(project, pid=os.getppid())  # a live process that isn't this one
    events: list[tuple] = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))
    spawned = []
    monkeypatch.setattr(deploy_mod, "spawn_background_deploy", lambda *a, **k: spawned.append(a))

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(project), *extra],
        obj=_real_deploy_context(_DeployApi(events)),
    )

    assert result.exit_code == 1
    assert f"A deploy of {_APP} is already running (pid {os.getppid()})." in result.output
    assert f"agentbricks deployments wait {_APP}" in result.output
    assert events == [] and spawned == []
    assert _state(project)["pid"] == os.getppid()  # the record is untouched


def test_deploy_proceeds_when_the_recorded_deploy_died(
    project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    _write_state(project, pid=_DEAD_PID)
    events: list[tuple] = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(project), "--pip-index-url", ""],
        obj=_real_deploy_context(_DeployApi(events)),
    )

    assert result.exit_code == 0, result.output
    assert _state(project)["status"] == "succeeded"


# --- FR-5: deploy --no-wait --------------------------------------------------------------------


class _FakePopen:
    calls: list[tuple[list[str], dict]] = []

    def __init__(self, argv, **kwargs):
        type(self).calls.append((argv, kwargs))
        self.pid = 4321


@pytest.fixture
def popen(monkeypatch: pytest.MonkeyPatch) -> type[_FakePopen]:
    _FakePopen.calls = []
    monkeypatch.setattr(
        deploy_mod,
        "spawn_background_deploy",
        functools.partial(deploy_mod.spawn_background_deploy, popen=_FakePopen),
    )
    return _FakePopen


def test_no_wait_spawns_a_detached_child_and_records_it(
    project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, popen
):
    events: list[tuple] = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(project), "--instances", "2", "--no-wait"],
        obj=_real_deploy_context(_DeployApi(events)),
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        f"Deploy of {_APP} started in the background. Next: agentbricks deployments wait {_APP}\n"
    )
    [(argv, kwargs)] = popen.calls
    assert argv == [
        sys.executable,
        "-m",
        "databricks_agentbricks.cli",
        "--profile",
        "prof",
        "deploy",
        "demo",
        "--source",
        str(project.resolve()),
        "--instances",
        "2",
        "--background-child",
    ]
    log = deploy_state.log_path(project, _APP)
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["stdout"].name == str(log)
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    state = _state(project)
    assert state["status"] == "running"
    assert state["pid"] == 4321
    assert state["log"] == str(log)
    # Nothing past the quick checks ran in the foreground.
    app_calls = [list(event[1]) for event in events if event[0] == "apps"]
    assert not any(call[:2] in (["apps", "create"], ["apps", "deploy"]) for call in app_calls)
    assert not any(call[0] == "sync" for call in app_calls)


def test_no_wait_json_reports_the_handoff(
    project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, popen, json_mode
):
    events: list[tuple] = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(project), "--no-wait"],
        obj=_real_deploy_context(_DeployApi(events), output="json"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "deployment": _APP,
        "status": "started",
        "log": str(deploy_state.log_path(project, _APP)),
        "state_file": str(deploy_state.state_path(project, _APP)),
    }


def test_no_wait_passes_through_non_default_options(
    project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, popen
):
    events: list[tuple] = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))
    ctx = _real_deploy_context(_DeployApi(events))
    ctx.profile = None

    result = CliRunner().invoke(
        deploy_mod.deploy,
        [
            "demo",
            "--source",
            str(project),
            "--pip-index-url",
            "",
            "--workspace-path",
            "/Workspace/agents/demo",
            "--no-wait",
        ],
        obj=ctx,
    )

    assert result.exit_code == 0, result.output
    [(argv, _)] = popen.calls
    assert argv[3:] == [
        "deploy",
        "demo",
        "--source",
        str(project.resolve()),
        "--pip-index-url=",
        "--workspace-path",
        "/Workspace/agents/demo",
        "--background-child",
    ]


def test_no_wait_bad_name_fails_synchronously_and_spawns_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, popen
):
    AgentProject.create(tmp_path, framework="langgraph", server="custom").write()
    events: list[tuple] = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["--source", str(tmp_path), "--no-wait"],
        obj=_real_deploy_context(_DeployApi(events)),
    )

    assert result.exit_code == 1
    assert "No deployment name given" in result.output
    assert popen.calls == [] and events == []
    assert not (tmp_path / ".agentbricks").exists()


def test_no_wait_auth_failure_fails_synchronously_and_spawns_nothing(
    project: pathlib.Path, monkeypatch: pytest.MonkeyPatch, popen
):
    class _SignedOut(_DeployApi):
        @property
        def current_user(self):
            raise ValueError("default auth: cannot configure default credentials")

    events: list[tuple] = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(project), "--no-wait"],
        obj=_real_deploy_context(_SignedOut(events)),
    )

    assert result.exit_code == 1
    assert "Could not authenticate with the Databricks workspace." in result.output
    assert popen.calls == []


def test_background_child_runs_the_deploy_into_the_parents_record(
    project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    # The parent recorded the child (this process) as running; the child must not refuse itself.
    deploy_state.record_background_start(project, _APP, pid=os.getpid(), now=1_800_000_000.0)
    events: list[tuple] = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))
    monkeypatch.setattr(deploy_mod, "_ignore_hangup", lambda: None)

    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["demo", "--source", str(project), "--pip-index-url", "", "--background-child"],
        obj=_real_deploy_context(_DeployApi(events)),
    )

    assert result.exit_code == 0, result.output
    state = _state(project)
    assert state["status"] == "succeeded"
    assert state["started_at"] == deploy_state.timestamp(1_800_000_000.0)
    assert state["log"] == str(deploy_state.log_path(project, _APP))


@pytest.mark.skipif(os.name == "nt", reason="POSIX sessions")
def test_spawned_child_runs_in_its_own_session_with_output_in_the_log(tmp_path: pathlib.Path):
    log = tmp_path / ".agentbricks" / "deploy-x.log"
    script = (
        "import os, sys; "
        "print('own session:', os.getsid(0) == os.getpid()); "
        "print('stdin:', repr(sys.stdin.read())); "
        "print('to stderr', file=sys.stderr)"
    )

    pid = deploy_mod.spawn_background_deploy([sys.executable, "-c", script], log=log)

    deadline = deploy_state.time.monotonic() + 10
    while deploy_state.pid_alive(pid) and deploy_state.time.monotonic() < deadline:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            break
        deploy_state.time.sleep(0.01)
    assert log.read_text().splitlines() == ["own session: True", "stdin: ''", "to stderr"]


def test_background_child_flag_is_hidden_from_help():
    result = CliRunner().invoke(deploy_mod.deploy, ["--help"])
    assert "--no-wait" in result.output
    assert "--background-child" not in result.output


# --- FR-6 / FR-8: deployments wait, following the local record ------------------------------


def _wait(source: pathlib.Path, *args: str, output: str = "text", runner=None):
    events: list[tuple] = []
    return CliRunner().invoke(
        deploy_mod.deployments,
        ["wait", *args, "--source", str(source)],
        obj=_real_deploy_context(_DeployApi(events), output=output),
    )


def test_wait_reports_a_finished_deploy_with_its_url(project: pathlib.Path):
    _write_state(project, status="succeeded", result={"url": "https://agent.example"})

    result = _wait(project, _APP)

    assert result.exit_code == 0, result.output
    assert "Deployed agent 'agent-bricks-demo'" in result.stdout
    assert "https://agent.example" in result.stdout
    assert f"agentbricks endpoint invoke {_APP}" in result.stdout


def test_wait_reports_a_failed_background_deploy_with_its_log_tail(project: pathlib.Path):
    log = deploy_state.log_path(project, _APP)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(f"log line {i}\n" for i in range(30)))
    _write_state(
        project,
        status="failed",
        error=f"Could not deploy '{_APP}'.",
        phase="Deploying the agent source",
        log=str(log),
    )

    result = _wait(project, _APP)

    assert result.exit_code == 1
    assert f"error: Could not deploy '{_APP}'." in result.stderr
    assert "log line 29" in result.stderr
    assert "log line 10" in result.stderr
    assert "log line 9\n" not in result.stderr


def test_wait_times_out_with_exit_75_while_still_deploying(project: pathlib.Path):
    _write_state(project, pid=os.getppid())

    result = _wait(project, _APP, "--timeout", "0")

    assert result.exit_code == 75
    assert "• Waiting for agent compute to start…" in result.stderr
    assert result.stdout.startswith("still deploying: Waiting for agent compute to start, 0:0")
    assert result.stdout.rstrip().endswith("elapsed — run the same command again")


def test_wait_reports_a_dead_deploy_process_as_failed(project: pathlib.Path):
    _write_state(project, pid=_DEAD_PID)

    result = _wait(project, _APP, "--timeout", "0")

    assert result.exit_code == 1
    assert "deploy process exited unexpectedly" in result.stderr


@pytest.mark.parametrize(
    ("fields", "exit_code", "status"),
    [
        ({"status": "succeeded", "result": {"url": "https://agent.example"}}, 0, "succeeded"),
        ({"status": "failed", "error": "boom"}, 1, "failed"),
        ({"pid": os.getppid()}, 75, "running"),
    ],
)
def test_wait_json_prints_exactly_one_object_with_the_same_exit_code(
    project: pathlib.Path, json_mode, fields, exit_code, status
):
    _write_state(project, **fields)

    result = _wait(project, _APP, "--timeout", "0", output="json")

    assert result.exit_code == exit_code
    payload = json.loads(result.stdout)
    assert set(payload) == {
        "deployment",
        "status",
        "phase",
        "elapsed_s",
        "url",
        "error",
        "log_tail",
    }
    assert payload["deployment"] == _APP
    assert payload["status"] == status


# --- FR-7: deployments wait from the App's state -----------------------------------------------


def _app_runner(payloads: list[dict | None], seen: list[list[str]]):
    def run(args, profile, **kwargs):
        seen.append(list(args))
        payload = payloads.pop(0) if len(payloads) > 1 else payloads[0]
        if payload is None:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="Error: does not exist")
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    return run


def test_wait_defaults_name_from_agent_toml_and_reads_the_app(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.set_deployment_name("demo")
    project.write()
    seen: list[list[str]] = []
    payload = {
        "url": "https://agent.example",
        "compute_status": {"state": "ACTIVE"},
        "active_deployment": {"status": {"state": "SUCCEEDED"}},
    }
    monkeypatch.setattr(deploy_mod, "_databricks", _app_runner([payload], seen))

    result = _wait(tmp_path)

    assert result.exit_code == 0, result.output
    assert seen == [["apps", "get", _APP, "-o", "json"]]
    assert "https://agent.example" in result.stdout
    assert "--path /api/invocations" in result.stdout  # an Agent Bricks Runtime project


def test_wait_reports_a_failed_latest_deployment(
    project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    payload = {
        "compute_status": {"state": "ACTIVE"},
        "pending_deployment": {
            "status": {"state": "FAILED", "message": "app crashed: ModuleNotFoundError"}
        },
    }
    monkeypatch.setattr(deploy_mod, "_databricks", _app_runner([payload], []))

    result = _wait(project, _APP)

    assert result.exit_code == 1
    assert f"Could not deploy '{_APP}'." in result.stderr
    assert "app crashed: ModuleNotFoundError" in result.stderr


def test_wait_reports_a_missing_app(project: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deploy_mod, "_databricks", _app_runner([None], []))

    result = _wait(project, _APP)

    assert result.exit_code == 1
    assert f"Deployment '{_APP}' does not exist." in result.stderr


_PACKAGE = pathlib.Path(__file__).resolve().parents[2]


def test_readme_and_cli_reference_list_no_wait_and_wait():
    readme = (_PACKAGE / "README.md").read_text()
    reference = (_PACKAGE / "cli.md").read_text()

    note = readme.split("### Deploying from a coding agent", 1)[1].split("\n## ", 1)[0]
    for phrase in ("~3 min", "timeout: 600000", "--no-wait", "exit 75", "Don't poll"):
        assert phrase in note
    assert "[--no-wait]" in readme
    assert "wait       [<name>] [--timeout SECONDS] [--source PATH]" in readme
    assert "| `--no-wait` | flag |" in reference
    assert "#### `agentbricks deployments wait`" in reference
    assert "| `--timeout <TIMEOUT>` | integer range | `100` |" in reference
    assert "background-child" not in readme + reference


def test_wait_help_lists_exit_codes_and_examples():
    result = CliRunner().invoke(deploy_mod.deployments, ["wait", "--help"])
    assert result.exit_code == 0
    assert "75 still deploying" in result.output
    assert "--timeout" in result.output and "100" in result.output
