from __future__ import annotations

import json
import signal
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli import endpoint as endpoint_mod
from databricks_agentbricks.cli.endpoint import endpoint
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke import loadtest as loadtest_mod
from databricks_agentbricks.services.invoke.loadtest_streaming import (
    StreamError,
    StreamMetrics,
    output_tokens,
)
from databricks_agentbricks.services.invoke.transport import EndpointRequest, EndpointResponse


class _Ctx:
    profile = "test-profile"
    output = "json"


def _capture_loadtest(monkeypatch):
    captured = {}

    def run(request, **kwargs):
        captured.update(request=request, **kwargs)
        return loadtest_mod.LoadtestResult(0, {})

    monkeypatch.setattr(endpoint_mod, "run_loadtest", run)
    return captured


@pytest.mark.parametrize("routing_key", [None, "chosen-session"])
def test_loadtest_reuses_app_auth_and_request_with_per_user_routing(monkeypatch, routing_key):
    captured = _capture_loadtest(monkeypatch)
    apps = Mock()
    apps.get_app_url.return_value = "https://app.example"
    authenticator = Mock()
    authenticator.authorization_header.return_value = "Bearer test-token"
    transport = Mock()
    transport.send.side_effect = AssertionError("CLI must delegate preflight and load to runner")
    monkeypatch.setattr(endpoint_mod, "AppsClient", lambda profile: apps)
    monkeypatch.setattr(endpoint_mod, "WorkspaceOAuthAuthenticator", lambda profile: authenticator)
    monkeypatch.setattr(endpoint_mod, "HttpSession", lambda: transport)
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
        "max_requests": None,
        "cases": None,
        "report_dir": None,
        "follow_events": False,
        "mlflow_service": None,
    }


def test_loadtest_url_uses_requested_settings_without_app_routing(monkeypatch):
    captured = _capture_loadtest(monkeypatch)
    context = _Ctx()
    context.output = "text"
    result = CliRunner().invoke(
        endpoint,
        (
            "invoke --url http://localhost:8000 --path /health --method GET "
            "--loadtest --concurrency 4 --duration 3 --max-requests 9"
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
        "max_requests": 9,
        "cases": None,
        "report_dir": None,
        "follow_events": False,
        "mlflow_service": None,
    }


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--concurrency", "2"),
        ("--duration", "2"),
        ("--max-requests", "2"),
        ("--cases", "cases.json"),
        ("--report-dir", "reports"),
        ("--experiment-id", "123"),
        ("--scorer", "quality"),
        ("--mlflow-max-requests", "5"),
        ("--warehouse", "warehouse"),
    ],
)
def test_load_settings_require_loadtest_before_resolving_an_endpoint(
    monkeypatch, tmp_path, option, value
):
    if option == "--cases":
        cases = tmp_path / value
        cases.write_text("[{}]")
        value = str(cases)
    monkeypatch.setattr(
        endpoint_mod,
        "build_invoke_service",
        lambda *args: pytest.fail("Must reject before resolving"),
    )

    result = CliRunner().invoke(endpoint, ["invoke", "--path", "/run", option, value], obj=_Ctx())

    assert result.exit_code != 0
    assert "--loadtest" in result.output


@pytest.mark.parametrize("option", ["--concurrency", "--duration", "--max-requests"])
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


@pytest.mark.parametrize("contents", ["not json", "{}", "[]"])
def test_invalid_cases_rejected_before_endpoint_resolution(monkeypatch, tmp_path, contents):
    cases = tmp_path / "cases.json"
    cases.write_text(contents)
    monkeypatch.setattr(
        endpoint_mod,
        "build_invoke_service",
        lambda *args: pytest.fail("Must reject before resolving"),
    )

    result = CliRunner().invoke(
        endpoint,
        ["invoke", "--path", "/run", "--loadtest", "--cases", str(cases)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "cases" in result.output.lower()


def test_cases_and_json_are_mutually_exclusive(monkeypatch, tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text('[{"prompt": "from file"}]')
    monkeypatch.setattr(
        endpoint_mod,
        "build_invoke_service",
        lambda *args: pytest.fail("Must reject before resolving"),
    )

    result = CliRunner().invoke(
        endpoint,
        ["invoke", "--path", "/run", "--loadtest", "--cases", str(cases), "--json", "null"],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "--cases" in result.output and "--json" in result.output


def test_cases_and_report_location_are_forwarded(monkeypatch, tmp_path):
    captured = _capture_loadtest(monkeypatch)
    cases = tmp_path / "cases.json"
    payloads = [{"prompt": "first", "id": "{{uuid}}"}, None]
    cases.write_text(json.dumps(payloads))
    report_dir = tmp_path / "report"

    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "http://localhost:8000",
            "--path",
            "/run",
            "--loadtest",
            "--cases",
            str(cases),
            "--report-dir",
            str(report_dir),
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert captured["cases"] == payloads
    assert Path(captured["report_dir"]) == report_dir
    # Explicit JSON null is a body; cases may contain arbitrary JSON payloads.
    assert captured["request"].body_set


def _request(**changes) -> EndpointRequest:
    values = {
        "url": "https://app.example/events?mode=fast",
        "method": "POST",
        "headers": {"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        "body": None,
        "timeout": 12.5,
        "sse": True,
        "body_set": True,
    }
    return EndpointRequest(**(values | changes))


def test_mlflow_options_forward_registered_scorers_and_expectations(monkeypatch, tmp_path):
    captured = _capture_loadtest(monkeypatch)
    mlflow = Mock()
    constructor = Mock(return_value=mlflow)
    monkeypatch.setattr(endpoint_mod, "LoadTestMLflowService", constructor)
    expectations = tmp_path / "expected.json"
    expectations.write_text('[{"expected_response":"two"}]')
    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "http://localhost:8000",
            "--path",
            "/run",
            "--loadtest",
            "--experiment-id",
            "123",
            "--scorer",
            "correctness",
            "--scorer",
            "relevance",
            "--traces",
            "--mlflow-max-requests",
            "3",
            "--warehouse",
            "abc",
            "--expectations",
            str(expectations),
        ],
        obj=_Ctx(),
    )
    assert result.exit_code == 0, result.output
    constructor.assert_called_once_with(
        "test-profile",
        "123",
        scorer_names=("correctness", "relevance"),
        traces=True,
        max_requests=3,
        expectations=[{"expected_response": "two"}],
        warehouse_id="abc",
    )
    assert captured["mlflow_service"] is mlflow


@pytest.mark.parametrize(
    "options, expected",
    [
        (["--traces"], "--experiment-id"),
        (["--scorer", "quality"], "--experiment-id"),
        (["--experiment-id", "123"], "--scorer or --traces"),
        (["--mlflow-max-requests", "2"], "--scorer or --traces"),
    ],
)
def test_mlflow_options_reject_incomplete_configuration_before_endpoint_resolution(
    monkeypatch, options, expected
):
    monkeypatch.setattr(
        endpoint_mod,
        "build_invoke_service",
        lambda *a: pytest.fail("Must reject before resolution"),
    )
    result = CliRunner().invoke(
        endpoint, ["invoke", "--path", "/run", "--loadtest", *options], obj=_Ctx()
    )
    assert result.exit_code != 0
    assert expected in result.output


@pytest.mark.parametrize("contents", ["{}", "[]", "[null]", "[{},{}]", "not-json"])
def test_expectations_require_one_object_per_case(monkeypatch, tmp_path, contents):
    expectations = tmp_path / "expected.json"
    expectations.write_text(contents)
    monkeypatch.setattr(
        endpoint_mod,
        "build_invoke_service",
        lambda *a: pytest.fail("Must reject before resolution"),
    )
    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--path",
            "/run",
            "--loadtest",
            "--scorer",
            "quality",
            "--experiment-id",
            "123",
            "--expectations",
            str(expectations),
        ],
        obj=_Ctx(),
    )
    assert result.exit_code != 0
    assert "--expectations" in result.output


def _preflight(monkeypatch, *, status=200, error=None, events=()):
    transport = Mock()
    transport.send.return_value = EndpointResponse(
        url="https://app.example/events?mode=fast",
        status_code=status,
        headers={},
        body={},
        elapsed_seconds=0.1,
        events=events,
    )
    if error is not None:
        transport.send.side_effect = error
    monkeypatch.setattr(loadtest_mod, "HttpSession", lambda **kwargs: transport)
    return transport


def _process(monkeypatch, *, returncode=0, summary=None):
    captured = {}
    process = MagicMock(returncode=returncode, pid=123)
    process.__enter__.return_value = process

    def communicate(input=None):
        if input is not None:
            captured["config"] = json.loads(input)
            if summary is not None:
                Path(captured["config"]["summary_file"]).write_text(json.dumps(summary))
        return None, None

    process.communicate.side_effect = communicate

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return process

    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: object())
    monkeypatch.setattr(loadtest_mod.subprocess, "Popen", popen)
    return process, captured


def test_missing_locust_gives_install_hint_without_starting_a_process(monkeypatch):
    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: None)
    monkeypatch.setattr(
        loadtest_mod.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not spawn")
    )

    with pytest.raises(AgentCliError) as exc:
        loadtest_mod.run_loadtest(
            _request(), concurrency=1, duration=60, output="text", per_user_routing=False
        )

    assert "locust" in str(exc.value).lower()
    assert "databricks-agentbricks[loadtest]" in (exc.value.hint or "")


@pytest.mark.parametrize("output", ["text", "json"])
def test_locust_receives_private_request_on_stdin_and_native_options(monkeypatch, output):
    preflight = _preflight(monkeypatch)
    _, captured = _process(monkeypatch, summary={"requests": 7})
    request = _request()

    result = loadtest_mod.run_loadtest(
        request,
        concurrency=4,
        duration=3,
        output=output,
        per_user_routing=True,
    )

    assert result.returncode == 0
    assert result.summary["requests"] == 7
    preflight.send.assert_called_once()
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
    assert "--headless" in command
    assert ("--json" in command) is (output == "json")
    assert not any("test-token" in argument for argument in command)
    assert captured["config"]["request"] == asdict(request)
    assert captured["config"]["per_user_routing"] is True
    assert captured["text"] is True
    assert captured["stdin"] is loadtest_mod.subprocess.PIPE
    assert captured["start_new_session"] is True
    assert "stdout" not in captured and "stderr" not in captured


@pytest.mark.parametrize(
    "body,method", [({}, "POST"), ({"background": True, "stream": True}, "GET"), (None, "POST")]
)
def test_follow_events_rejects_invalid_invocation_bodies_before_traffic(monkeypatch, body, method):
    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: object())
    transport = _preflight(monkeypatch)
    with pytest.raises(AgentCliError, match="background=true"):
        loadtest_mod.run_loadtest(
            _request(body=body, method=method),
            concurrency=1,
            duration=1,
            output="text",
            per_user_routing=False,
            follow_events=True,
        )
    transport.send.assert_not_called()


def test_mlflow_enrichment_runs_after_load_and_updates_saved_summary(monkeypatch, tmp_path):
    _preflight(monkeypatch)
    _, captured = _process(
        monkeypatch, summary={"counts": {"succeeded": 2}, "outcome": "completed"}
    )
    mlflow = Mock(max_requests=3)
    mlflow.finish.return_value = {"sampled_successes": 0, "evaluation": {"status": "no_samples"}}
    report = tmp_path / "report"
    result = loadtest_mod.run_loadtest(
        _request(),
        concurrency=1,
        duration=1,
        output="text",
        per_user_routing=False,
        mlflow_service=mlflow,
        report_dir=report,
    )
    mlflow.prepare.assert_called_once_with()
    mlflow.finish.assert_called_once_with([])
    assert captured["config"]["mlflow_max_requests"] == 3
    assert result.summary["mlflow"]["eligible_successes"] == 2
    assert json.loads((report / "summary.json").read_text()) == result.summary


def test_cancelled_load_does_not_start_judging(monkeypatch):
    _preflight(monkeypatch)
    _process(
        monkeypatch, returncode=143, summary={"counts": {"succeeded": 2}, "outcome": "cancelled"}
    )
    mlflow = Mock(max_requests=3)
    result = loadtest_mod.run_loadtest(
        _request(),
        concurrency=1,
        duration=1,
        output="text",
        per_user_routing=False,
        mlflow_service=mlflow,
    )
    mlflow.finish.assert_not_called()
    assert result.returncode == 143
    assert result.summary["mlflow"] == {"status": "not_run", "reason": "cancelled"}


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_during_mlflow_keeps_completed_performance_report(
    monkeypatch, tmp_path, stop_signal
):
    _preflight(monkeypatch)
    _process(monkeypatch, summary={"counts": {"succeeded": 2}, "outcome": "completed"})
    mlflow = Mock(max_requests=3)
    previous_terminate = signal.getsignal(signal.SIGTERM)

    def interrupted(observations):
        if stop_signal == signal.SIGTERM:
            handler = cast(Callable[[int, object], None], signal.getsignal(signal.SIGTERM))
            assert callable(handler)
            handler(signal.SIGTERM, None)
        raise KeyboardInterrupt

    mlflow.finish.side_effect = interrupted
    report = tmp_path / "report"
    result = loadtest_mod.run_loadtest(
        _request(),
        concurrency=1,
        duration=1,
        output="text",
        per_user_routing=False,
        mlflow_service=mlflow,
        report_dir=report,
    )
    assert result.returncode == 128 + stop_signal
    assert result.summary["outcome"] == "completed"
    assert result.summary["mlflow"] == {"status": "not_run", "reason": "analysis_cancelled"}
    assert json.loads((report / "summary.json").read_text()) == result.summary
    assert signal.getsignal(signal.SIGTERM) == previous_terminate


@pytest.mark.parametrize(
    "enrichment,code",
    [
        ({"evaluation": {"status": "failed"}}, 1),
        ({"evaluation": {"status": "partial"}}, 1),
        ({"evaluation": {"status": "completed", "metrics": {"quality/mean": 0}}}, 0),
        ({"traces": {"status": "partial", "missing": 3, "errors": 0}}, 0),
        ({"traces": {"status": "partial", "errors": 1}}, 1),
    ],
)
def test_mlflow_execution_errors_preserve_performance_but_exit_nonzero(
    monkeypatch, enrichment, code
):
    _preflight(monkeypatch)
    _process(monkeypatch, summary={"counts": {"succeeded": 2}, "outcome": "completed"})
    mlflow = Mock(max_requests=3)
    mlflow.finish.return_value = enrichment
    result = loadtest_mod.run_loadtest(
        _request(),
        concurrency=1,
        duration=1,
        output="text",
        per_user_routing=False,
        mlflow_service=mlflow,
    )
    assert result.returncode == code
    assert result.summary["outcome"] == "completed"


def test_report_options_save_summary_without_persisting_the_private_request(monkeypatch, tmp_path):
    _preflight(monkeypatch)
    summary = {"requests": 7, "outcome": "completed"}
    _, captured = _process(monkeypatch, summary=summary)
    report_dir = tmp_path / "report"

    result = loadtest_mod.run_loadtest(
        _request(),
        concurrency=2,
        duration=30,
        output="text",
        per_user_routing=False,
        max_requests=7,
        cases=[{"prompt": "private payload"}],
        report_dir=report_dir,
    )

    command = captured["command"]
    assert command[command.index("--html") + 1] == str(report_dir / "locust.html")
    assert command[command.index("--csv") + 1] == str(report_dir / "locust")
    assert "--csv-full-history" in command
    assert captured["config"]["max_requests"] == 7
    assert captured["config"]["cases"] == [{"prompt": "private payload"}]
    assert result.summary == summary
    assert json.loads((report_dir / "summary.json").read_text()) == summary
    assert all("test-token" not in path.read_text() for path in report_dir.iterdir())
    assert all("private payload" not in path.read_text() for path in report_dir.iterdir())


@pytest.mark.parametrize("status", [302, 401, 404, 503])
def test_failed_preflight_never_starts_load(monkeypatch, status):
    _preflight(monkeypatch, status=status)
    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: object())
    monkeypatch.setattr(
        loadtest_mod.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not spawn")
    )

    with pytest.raises(AgentCliError, match=str(status)):
        loadtest_mod.run_loadtest(
            _request(), concurrency=10, duration=60, output="text", per_user_routing=False
        )


@pytest.mark.parametrize(
    "data",
    [
        {"type": "run.failed", "error": {"message": "application failure"}},
        {"type": "response.output_text.delta", "delta": "incomplete response"},
        {"choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}]},
    ],
)
def test_failed_or_truncated_sse_preflight_never_starts_load(monkeypatch, data):
    _preflight(monkeypatch, events=({"data": data},))
    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: object())
    monkeypatch.setattr(
        loadtest_mod.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not spawn")
    )

    with pytest.raises(AgentCliError, match="preflight failed"):
        loadtest_mod.run_loadtest(
            _request(), concurrency=10, duration=60, output="text", per_user_routing=False
        )


def test_unreachable_preflight_never_starts_load(monkeypatch):
    _preflight(monkeypatch, error=AgentCliError("Could not reach endpoint"))
    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: object())
    monkeypatch.setattr(
        loadtest_mod.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not spawn")
    )

    with pytest.raises(AgentCliError, match="Could not reach"):
        loadtest_mod.run_loadtest(
            _request(), concurrency=10, duration=60, output="text", per_user_routing=False
        )


def test_preflight_materializes_fresh_uuid_without_mutating_the_load_payload(monkeypatch):
    preflight = _preflight(monkeypatch)
    _, captured = _process(monkeypatch)
    payload = {"id": "{{uuid}}", "nested": {"session_id": "{{uuid}}"}}
    request = _request(body=payload)

    loadtest_mod.run_loadtest(
        request,
        concurrency=1,
        duration=1,
        output="text",
        per_user_routing=False,
        cases=[payload, {"prompt": "second"}],
    )

    sent = preflight.send.call_args.args[0]
    assert uuid.UUID(sent.body["id"])
    assert sent.body["id"] == sent.body["nested"]["session_id"]
    assert request.body == payload
    assert captured["config"]["request"]["body"] == payload
    assert captured["config"]["cases"] == [payload, {"prompt": "second"}]


def test_report_directory_is_not_overwritten(monkeypatch, tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    original = directory / "summary.json"
    original.write_text("keep this")
    _preflight(monkeypatch)
    monkeypatch.setattr(loadtest_mod, "find_spec", lambda name: object())
    monkeypatch.setattr(
        loadtest_mod.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not spawn")
    )

    with pytest.raises(AgentCliError):
        loadtest_mod.run_loadtest(
            _request(),
            concurrency=1,
            duration=1,
            output="text",
            per_user_routing=False,
            report_dir=directory,
        )

    assert original.read_text() == "keep this"


@pytest.mark.parametrize("returncode", [1, 2])
def test_locust_failure_status_is_preserved(monkeypatch, returncode):
    _preflight(monkeypatch)
    _process(monkeypatch, returncode=returncode)

    result = CliRunner().invoke(
        endpoint,
        "invoke --url http://localhost:8000 --path /run --loadtest".split(),
        obj=_Ctx(),
    )
    assert result.exit_code == returncode


def test_interrupt_forwards_signal_and_preserves_the_summary(monkeypatch):
    _preflight(monkeypatch)
    process, captured = _process(monkeypatch, summary={"requests": 2})
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

    result = loadtest_mod.run_loadtest(
        _request(), concurrency=1, duration=60, output="text", per_user_routing=False
    )

    process.send_signal.assert_called_once_with(signal.SIGINT)
    assert process.communicate.call_count == 2
    assert result.returncode == 130
    assert result.summary["requests"] == 2
    assert "test-token" not in " ".join(captured["command"])


@pytest.mark.parametrize(
    "event",
    [
        {"response": "hello"},
        {"message": "hello"},
        {"usage": {"output_tokens": True}},
        {"usage": {"output_tokens": -1}},
        {"usage": {"output_tokens": "3"}},
    ],
)
def test_invalid_or_missing_token_usage_is_unknown(event):
    assert output_tokens(event) is None


def test_durable_sse_measures_first_answer_and_retains_explicit_usage():
    metrics = StreamMetrics()
    metrics.observe({"type": "run.started"}, elapsed_ms=5)
    metrics.observe({"type": "delta", "content": ""}, elapsed_ms=20)
    metrics.observe({"type": "delta", "content": "answer"}, elapsed_ms=75)
    metrics.observe({"type": "delta", "content": "tail"}, elapsed_ms=90)
    metrics.observe({"type": "message", "message": {"usage_metadata": {"output_tokens": 7}}})
    metrics.observe({"type": "run.completed"})
    metrics.finish()

    assert metrics.ttft_ms == 75
    assert metrics.output_tokens == 7


def test_chat_sse_skips_role_metadata_and_preserves_zero_usage():
    metrics = StreamMetrics()
    metrics.observe({"choices": [{"delta": {"role": "assistant"}}]}, elapsed_ms=10)
    metrics.observe({"choices": [{"delta": {"content": "answer"}}]}, elapsed_ms=80)
    metrics.observe({"choices": [{"delta": {}, "finish_reason": "stop"}]}, elapsed_ms=200)
    metrics.observe({"choices": [], "usage": {"completion_tokens": 0}})
    metrics.finish()

    assert metrics.ttft_ms == 80
    assert metrics.output_tokens == 0


def test_supported_sse_requires_completion_while_generic_sse_remains_unknown():
    generic = StreamMetrics()
    generic.observe({"text": "generic event"}, elapsed_ms=10)
    generic.finish()
    assert generic.ttft_ms is None
    assert generic.output_tokens is None

    supported = StreamMetrics()
    supported.observe({"type": "response.output_text.delta", "delta": "partial"}, elapsed_ms=50)
    with pytest.raises(StreamError, match="Incomplete"):
        supported.finish()
