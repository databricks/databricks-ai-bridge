"""Contracts for endpoint load-test plans, orchestration, and private response export."""

from __future__ import annotations

import json
import signal
import sys
import uuid
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli import endpoint as endpoint_mod
from databricks_agentbricks.cli.endpoint import endpoint
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services import endpoint_service as service_mod
from databricks_agentbricks.services.endpoint_service import EndpointService, EndpointTarget
from databricks_agentbricks.services.invoke.transport import EndpointRequest, EndpointResponse
from databricks_agentbricks.services.loadtest import runner as loadtest_mod
from databricks_agentbricks.services.loadtest.plan import LoadTestPlan


class _Ctx:
    profile = "test-profile"
    output = "json"


def _service():
    apps, auth, transport = Mock(), Mock(), Mock()
    apps.app_url.return_value = "https://app.example"
    auth.authorization_header.return_value = "Bearer test-token"
    return EndpointService(apps_client=apps, authenticator=auth, transport=transport), apps, auth


def _plan(**changes: Any) -> LoadTestPlan:
    values: dict[str, Any] = {"target": EndpointTarget(url="http://localhost:8000", path="/run")}
    values.update(changes)
    return LoadTestPlan(**values)


def _request():
    return EndpointRequest(
        url="https://app.example/run",
        method="POST",
        headers={"Authorization": "Bearer test-token"},
        body=None,
        timeout=12.5,
    )


def _process(monkeypatch, *, returncode=0, summary=None, observations=()):
    captured = {}
    process = MagicMock(returncode=returncode, pid=123)
    process.__enter__.return_value = process

    def communicate(input=None):
        if input is not None:
            config = captured["config"] = json.loads(input)
            if summary is not None:
                Path(config["summary_file"]).write_text(json.dumps(summary))
            if config["observations_file"] is not None:
                destination = Path(config["observations_file"])
                captured["capture_mode"] = destination.stat().st_mode & 0o777
                with destination.open("a") as stream:
                    for observation in observations:
                        stream.write(json.dumps(observation) + "\n")
        return None, None

    process.communicate.side_effect = communicate

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return process

    monkeypatch.setattr(loadtest_mod.subprocess, "Popen", popen)
    return process, captured


def _capture_cli(monkeypatch, *, summary=None, returncode=0):
    service = Mock()
    service.loadtest.return_value = loadtest_mod.LoadtestResult(returncode, summary or {})
    monkeypatch.setattr(endpoint_mod, "build_endpoint_service", lambda obj: service)
    monkeypatch.setattr(endpoint_mod, "render_loadtest_summary", Mock())
    return service


@pytest.mark.parametrize("routing_key", [None, "chosen-session"])
def test_loadtest_resolves_shared_target_auth_and_one_body_source(monkeypatch, routing_key):
    service, apps, auth = _service()
    monkeypatch.setattr(service_mod, "find_spec", lambda name: object())
    run = Mock(return_value=loadtest_mod.LoadtestResult(0, {}))
    monkeypatch.setattr(service_mod, "run_loadtest", run)
    target = EndpointTarget(
        app="my-agent",
        path="/events",
        method="PATCH",
        query=("mode=fast",),
        timeout=12.5,
        sse=True,
        routing_key=routing_key,
    )
    plan = _plan(target=target, bodies=("null", '{"prompt":"second"}'))
    service.loadtest(plan)
    apps.app_url.assert_called_once_with("agent-bricks-my-agent")
    auth.authorization_header.assert_called_once_with()
    request = run.call_args.args[0]
    assert request.url == "https://app.example/events?mode=fast"
    assert request.method == "PATCH" and request.timeout == 12.5 and request.sse is True
    assert request.body is None and request.body_set is True
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == "Bearer test-token"
    if routing_key:
        assert request.headers["X-Routing-Key"] == routing_key
    else:
        assert uuid.UUID(request.headers["X-Routing-Key"])
    assert run.call_args.kwargs == {"plan": plan, "per_user_routing": routing_key is None}


def test_cli_passes_immutable_plan_and_emits_only_summary(monkeypatch, tmp_path):
    summary = {"schema_version": 1, "counts": {"succeeded": 5}}
    service = _capture_cli(monkeypatch, summary=summary)
    render = Mock()
    monkeypatch.setattr(endpoint_mod, "render_loadtest_summary", render)
    requests = tmp_path / "requests.json"
    requests.write_text('[null,{"input":"hello"}]')
    result = CliRunner().invoke(
        endpoint,
        [
            "loadtest",
            "--url",
            "http://localhost:8000",
            "--path",
            "/run",
            "--requests",
            str(requests),
            "--concurrency",
            "4",
            "--max-duration",
            "3",
            "--max-requests",
            "5",
            "--no-warmup",
            "--save-responses",
            str(tmp_path / "responses.json"),
        ],
        obj=_Ctx(),
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == summary
    render.assert_called_once_with(summary, report_dir=None, output="json")
    plan = service.loadtest.call_args.args[0]
    assert plan.bodies == ("null", '{"input": "hello"}')
    assert plan.concurrency == 4 and plan.duration == 3 and plan.max_requests == 5
    assert plan.warmup is False and plan.responses_file == tmp_path / "responses.json"
    with pytest.raises(FrozenInstanceError):
        plan.duration = 100


def test_omitted_requests_is_bodyless_and_warmup_defaults_on(monkeypatch):
    service = _capture_cli(monkeypatch)
    result = CliRunner().invoke(
        endpoint,
        ["loadtest", "--url", "http://localhost:8000", "--path", "/health", "--method", "GET"],
        obj=_Ctx(),
    )
    assert result.exit_code == 0, result.output
    plan = service.loadtest.call_args.args[0]
    assert plan.bodies is None and plan.warmup is True


def test_invoke_keeps_single_json_body_without_load(monkeypatch):
    service = _capture_cli(monkeypatch)
    service.invoke.return_value = EndpointResponse(
        url="http://localhost:8000/run",
        status_code=200,
        headers={},
        body={"output": "answer"},
        elapsed_seconds=0.1,
    )
    result = CliRunner().invoke(
        endpoint,
        ["invoke", "--url", "http://localhost:8000", "--path", "/run", "--json", "null"],
        obj=_Ctx(),
    )
    assert result.exit_code == 0, result.output
    assert service.invoke.call_args.args[0].json_value == "null"
    service.loadtest.assert_not_called()


@pytest.mark.parametrize(
    "option", ["--json", "--follow-events", "--sample-requests", "--cases", "--scorer", "--traces"]
)
def test_loadtest_rejects_removed_or_invoke_only_options(option):
    result = CliRunner().invoke(endpoint, ["loadtest", "--path", "/run", option], obj=_Ctx())
    assert result.exit_code == 2 and "No such option" in result.output


@pytest.mark.parametrize(
    "option,value",
    [
        ("--concurrency", "2"),
        ("--requests", "requests.json"),
        ("--save-responses", "responses.json"),
        ("--no-warmup", None),
    ],
)
def test_load_options_are_not_accepted_by_invoke(option, value):
    args = ["invoke", "--path", "/run", option] + ([value] if value else [])
    result = CliRunner().invoke(endpoint, args, obj=_Ctx())
    assert result.exit_code == 2 and "No such option" in result.output


@pytest.mark.parametrize("option", ["--concurrency", "--max-duration", "--max-requests"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_load_settings_must_be_positive(option, value):
    result = CliRunner().invoke(endpoint, ["loadtest", "--path", "/run", option, value], obj=_Ctx())
    assert result.exit_code == 2 and "range" in result.output


@pytest.mark.parametrize("field", ["concurrency", "duration", "max_requests"])
@pytest.mark.parametrize("value", [0, -1])
def test_service_rejects_nonpositive_plan_limits_before_workspace_access(field, value):
    service, apps, auth = _service()
    with pytest.raises(AgentCliError, match="must be positive"):
        service.loadtest(_plan(target=EndpointTarget(app="demo", path="/run"), **{field: value}))
    apps.app_url.assert_not_called()
    auth.authorization_header.assert_not_called()


@pytest.mark.parametrize("contents", ["not json", "{}", "[]"])
def test_invalid_requests_rejected_before_workspace_collaborators(monkeypatch, tmp_path, contents):
    requests = tmp_path / "requests.json"
    requests.write_text(contents)
    monkeypatch.setattr(
        endpoint_mod, "build_endpoint_service", lambda *args: pytest.fail("Must not resolve")
    )
    result = CliRunner().invoke(
        endpoint, ["loadtest", "--path", "/run", "--requests", str(requests)], obj=_Ctx()
    )
    assert result.exit_code != 0 and "--requests" in result.output


def test_missing_locust_fails_before_app_lookup_or_oauth(monkeypatch):
    service, apps, auth = _service()
    monkeypatch.setattr(service_mod, "find_spec", lambda name: None)
    with pytest.raises(AgentCliError) as error:
        service.loadtest(_plan(target=EndpointTarget(app="demo", path="/run")))
    assert "databricks-agentbricks[loadtest]" in (error.value.hint or "")
    apps.app_url.assert_not_called()
    auth.authorization_header.assert_not_called()


@pytest.mark.parametrize(
    "target",
    [
        EndpointTarget(app="demo", path="https://other.example/run"),
        EndpointTarget(app="demo", path="/run", query=("invalid",)),
        EndpointTarget(app="demo", path="/run", timeout=float("inf")),
        EndpointTarget(app="demo", path="/run", timeout=float("nan")),
    ],
)
def test_bad_target_fails_before_app_lookup_or_oauth(target):
    service, apps, auth = _service()
    with pytest.raises(AgentCliError):
        service.loadtest(_plan(target=target))
    apps.app_url.assert_not_called()
    auth.authorization_header.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["existing-response", "missing-parent", "inside-report", "nonempty-report"]
)
def test_bad_export_or_report_fails_before_app_lookup_or_oauth(tmp_path, failure):
    service, apps, auth = _service()
    report, responses = tmp_path / "report", tmp_path / "responses.json"
    if failure == "existing-response":
        responses.write_text("preserved")
    elif failure == "missing-parent":
        responses = tmp_path / "missing" / "responses.json"
    elif failure == "inside-report":
        responses = report / "responses.json"
    else:
        report.mkdir()
        (report / "summary.json").write_text("preserved")
    with pytest.raises(AgentCliError):
        service.loadtest(
            _plan(
                target=EndpointTarget(app="demo", path="/run"),
                report_dir=report,
                responses_file=responses,
            )
        )
    apps.app_url.assert_not_called()
    auth.authorization_header.assert_not_called()


@pytest.mark.parametrize("warmup", [False, True])
def test_locust_private_stdin_single_body_source_and_stderr_output(monkeypatch, warmup):
    _, captured = _process(monkeypatch, summary={"counts": {"succeeded": 7}})
    plan = _plan(concurrency=4, duration=3, bodies=("null",), warmup=warmup)
    result = loadtest_mod.run_loadtest(_request(), plan=plan, per_user_routing=True)
    assert result.returncode == 0
    command = captured["command"]
    assert command[:3] == [sys.executable, "-m", "locust"]
    assert Path(command[command.index("-f") + 1]).is_file()
    for option, expected in (
        ("--users", "4"),
        ("--spawn-rate", "4"),
        ("--run-time", "3s"),
        ("--stop-timeout", "13"),
    ):
        assert command[command.index(option) + 1] == expected
    assert "--json" not in command and "--headless" in command
    assert "test-token" not in " ".join(command)
    assert captured["stdout"] is sys.stderr and captured["stdin"] is loadtest_mod.subprocess.PIPE
    config = captured["config"]
    assert "body" not in config["request"] and config["bodies"] == ["null"]
    assert config["warmup_requested"] is warmup and config["observations_file"] is None
    assert "cases" not in config and "sample_requests" not in config


@pytest.mark.parametrize("returncode,outcome", [(0, "completed"), (143, "cancelled")])
def test_exports_every_success_privately_after_drain(monkeypatch, tmp_path, returncode, outcome):
    observations = [
        {
            "case_index": i % 2,
            "inputs": {"private-input": i},
            "outputs": "private-answer",
            "trace_id": "tr-private",
        }
        for i in range(45)
    ]
    summary = {"counts": {"succeeded": 45}, "outcome": outcome}
    _, captured = _process(
        monkeypatch, returncode=returncode, summary=summary, observations=observations
    )
    report, responses = tmp_path / "report", tmp_path / "responses.json"
    plan = _plan(bodies=("null", "{}"), report_dir=report, responses_file=responses)
    plan.validate()
    result = loadtest_mod.run_loadtest(_request(), plan=plan, per_user_routing=False)
    assert result.returncode == returncode
    assert json.loads(responses.read_text()) == {
        "schema_version": 1,
        "case_count": 2,
        "observations": observations,
    }
    assert responses.stat().st_mode & 0o777 == 0o600 and captured["capture_mode"] == 0o600
    assert not Path(captured["config"]["observations_file"]).exists()
    public = (report / "summary.json").read_text()
    assert json.loads(public) == summary
    assert all(
        private not in public
        for private in ("private-input", "private-answer", "tr-private", "test-token")
    )


def test_report_options_delegate_native_artifacts(monkeypatch, tmp_path):
    _, captured = _process(monkeypatch, summary={"outcome": "completed"})
    report = tmp_path / "report"
    plan = _plan(report_dir=report)
    plan.validate()
    loadtest_mod.run_loadtest(_request(), plan=plan, per_user_routing=False)
    command = captured["command"]
    assert command[command.index("--html") + 1] == str(report / "locust.html")
    assert command[command.index("--csv") + 1] == str(report / "locust")
    assert "--csv-full-history" in command


def test_warmup_error_is_actionable(monkeypatch):
    _process(monkeypatch, returncode=1, summary={"warmup_error": "HTTP 401"})
    with pytest.raises(AgentCliError, match="warm-up failed: HTTP 401.*No measured requests"):
        loadtest_mod.run_loadtest(_request(), plan=_plan(), per_user_routing=False)


def test_capture_failure_preserves_reports_and_captured_successes(monkeypatch, tmp_path):
    _process(
        monkeypatch,
        returncode=1,
        summary={"capture_error": "Response capture failed (OSError)"},
        observations=[{"outputs": "answer"}],
    )
    plan = _plan(report_dir=tmp_path / "report", responses_file=tmp_path / "responses.json")
    plan.validate()
    with pytest.raises(AgentCliError, match="Response capture failed"):
        loadtest_mod.run_loadtest(_request(), plan=plan, per_user_routing=False)
    assert plan.report_dir is not None and plan.responses_file is not None
    assert (plan.report_dir / "summary.json").is_file()
    assert json.loads(plan.responses_file.read_text())["observations"] == [{"outputs": "answer"}]


def test_export_preserves_file_created_after_validation(monkeypatch, tmp_path):
    _process(monkeypatch, summary={"outcome": "completed"})
    responses = tmp_path / "responses.json"
    plan = _plan(responses_file=responses)
    plan.validate()
    responses.write_text("preserved")
    with pytest.raises(AgentCliError, match="Could not save responses"):
        loadtest_mod.run_loadtest(_request(), plan=plan, per_user_routing=False)
    assert responses.read_text() == "preserved"


def test_interrupt_preserves_summary_and_export(monkeypatch, tmp_path):
    process, _ = _process(
        monkeypatch, summary={"counts": {"succeeded": 1}}, observations=[{"outputs": "answer"}]
    )
    communicate = process.communicate.side_effect
    interrupted = False

    def interrupt_once(input=None):
        nonlocal interrupted
        result = communicate(input)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    process.communicate.side_effect = interrupt_once
    responses = tmp_path / "responses.json"
    result = loadtest_mod.run_loadtest(
        _request(), plan=_plan(responses_file=responses), per_user_routing=False
    )
    assert result.returncode == 130 and result.summary["outcome"] == "cancelled"
    process.send_signal.assert_called_once_with(signal.SIGINT)
    assert json.loads(responses.read_text())["observations"] == [{"outputs": "answer"}]


def test_cli_preserves_runner_failure_status(monkeypatch):
    _capture_cli(monkeypatch, returncode=2, summary={"outcome": "incomplete"})
    result = CliRunner().invoke(
        endpoint, ["loadtest", "--url", "http://localhost:8000", "--path", "/run"], obj=_Ctx()
    )
    assert result.exit_code == 2 and json.loads(result.stdout) == {"outcome": "incomplete"}


def test_help_documents_loadtest_contract():
    result = CliRunner().invoke(endpoint, ["loadtest", "--help"])
    assert result.exit_code == 0
    text = " ".join(result.output.split()).replace("- ", "-")
    for contract in (
        "whichever comes first",
        "drain in-flight requests",
        "all HTML, CSV, and JSON reports together",
        "all successful inputs, outputs",
        "excludes cold-start latency",
        "--warmup / --no-warmup",
    ):
        assert contract in text


def test_capture_disk_failure_preserves_complete_records_before_partial_line(tmp_path):
    source = tmp_path / "observations.jsonl"
    source.write_text('{"outputs":"first answer"}\n{"outputs":"truncated')
    destination = tmp_path / "responses.json"
    loadtest_mod._save_responses(source, destination, 1, capture_failed=True)
    assert json.loads(destination.read_text())["observations"] == [{"outputs": "first answer"}]


def test_unexpected_corrupt_capture_is_reported_and_partial_export_removed(tmp_path):
    source = tmp_path / "observations.jsonl"
    source.write_text('{"outputs":"first answer"}\n{"outputs":"truncated')
    destination = tmp_path / "responses.json"
    with pytest.raises(AgentCliError, match="Could not save responses"):
        loadtest_mod._save_responses(source, destination, 1)
    assert not destination.exists()


def test_sigterm_is_forwarded_and_returns_shell_cancellation_status(monkeypatch):
    process, _ = _process(monkeypatch, returncode=-15, summary={"counts": {"succeeded": 1}})
    handlers = {}

    def signal_handler(signum, callback):
        previous = handlers.get(signum)
        handlers[signum] = callback
        return previous

    monkeypatch.setattr(loadtest_mod.signal, "signal", signal_handler)
    communicate = process.communicate.side_effect

    def terminate_during_load(input=None):
        result = communicate(input)
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return result

    process.communicate.side_effect = terminate_during_load
    result = loadtest_mod.run_loadtest(_request(), plan=_plan(), per_user_routing=False)
    assert result.returncode == 143
    assert result.summary["outcome"] == "cancelled"
    process.send_signal.assert_called_once_with(signal.SIGTERM)
