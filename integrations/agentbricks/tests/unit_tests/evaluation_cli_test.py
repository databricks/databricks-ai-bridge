"""Saved-response admission, standalone evaluation, privacy, and failure reports."""

from __future__ import annotations

import json
import signal
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.evaluation_service import (
    EvaluationService,
    load_evaluation_data,
)

mod = import_module("databricks_agentbricks.cli.eval")


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "responses.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_count": 2,
                "observations": [
                    {
                        "case_index": 1,
                        "inputs": {"query": "PRIVATE"},
                        "outputs": "PRIVATE",
                        "trace_id": None,
                    },
                    {
                        "case_index": 0,
                        "inputs": {"query": "PRIVATE"},
                        "outputs": "PRIVATE",
                        "trace_id": "explicit",
                    },
                ],
            }
        )
    )
    return path


def invoke(artifact, *options):
    return CliRunner().invoke(
        mod.evaluation,
        ["run", "--data", str(artifact), "--experiment-id", "123", *options],
        obj=SimpleNamespace(profile="selected", output="json"),
        env={"AGENTBRICKS_DISABLE_TELEMETRY": "1"},
    )


def test_eval_cli_passes_validated_saved_data_and_outputs_content_free_report(
    artifact, tmp_path, monkeypatch
):
    service = Mock()
    service.finish.return_value = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": "123",
        "sample_limit": 1,
        "sampled_successes": 1,
        "evaluation": {"status": "disabled", "scorers": []},
        "traces": {"status": "partial", "selected": 1, "fetched": 0},
    }
    constructor = Mock(return_value=service)
    monkeypatch.setattr(mod, "build_evaluation_service", constructor)
    report = tmp_path / "evaluation"
    result = invoke(artifact, "--traces", "--max-samples", "1", "--report-dir", str(report))
    assert result.exit_code == 0, result.output
    assert constructor.call_args.args[0].profile == "selected"
    assert constructor.call_args.kwargs["scorer_names"] == ()
    assert constructor.call_args.kwargs["max_samples"] == 1
    assert len(service.finish.call_args.args[0]) == 1
    assert service.finish.call_args.args[0][0]["case_index"] == 1
    assert json.loads(result.stdout) == service.finish.return_value
    assert json.loads((report / "summary.json").read_text()) == service.finish.return_value
    assert "Evaluation report" in result.stderr
    assert "PRIVATE" not in result.output + (report / "index.html").read_text()


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"case_count": 0},
        {"case_count": True},
        {"observations": {}},
        {"observations": [{"case_index": -1, "inputs": {}, "outputs": "PRIVATE"}]},
        {"observations": [{"case_index": 2, "inputs": {}, "outputs": "PRIVATE"}]},
        {"observations": [{"case_index": True, "inputs": {}, "outputs": "PRIVATE"}]},
        {"observations": [{"case_index": 0, "outputs": "PRIVATE"}]},
        {"observations": [{"case_index": 0, "inputs": {}, "outputs": "PRIVATE", "trace_id": {}}]},
    ],
)
def test_invalid_input_is_rejected_before_workspace_calls(artifact, monkeypatch, update):
    data = json.loads(artifact.read_text())
    data.update(update)
    artifact.write_text(json.dumps(data))
    factory = Mock()
    monkeypatch.setattr(mod, "build_evaluation_service", factory)
    result = invoke(artifact, "--traces")
    assert result.exit_code != 0
    assert "PRIVATE" not in result.output
    factory.assert_not_called()


@pytest.mark.parametrize(
    "content",
    [
        "PRIVATE malformed",
        '{"PRIVATE":NaN}',
        '{"PRIVATE":"\ud800"}'.encode("utf-8", errors="surrogatepass"),
    ],
)
def test_read_errors_are_sanitized(artifact, content):
    artifact.write_bytes(content if isinstance(content, bytes) else content.encode())
    result = invoke(artifact, "--traces")
    assert result.exit_code != 0
    assert "PRIVATE" not in result.output
    assert "JSON" in result.output


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--warehouse", "id", "--scorer", "quality"],
        ["--expectations", "missing", "--traces"],
        ["--max-samples", "0", "--traces"],
    ],
)
def test_invalid_option_combinations_fail_before_workspace_calls(artifact, monkeypatch, options):
    factory = Mock()
    monkeypatch.setattr(mod, "build_evaluation_service", factory)
    result = invoke(artifact, *options)
    assert result.exit_code != 0
    factory.assert_not_called()


@pytest.mark.parametrize(
    "content", ['[{"expected_response":"PRIVATE"}]', "null", "{}", '["PRIVATE", "PRIVATE"]']
)
def test_case_expectations_require_original_case_count_even_with_sample_limit(
    artifact, tmp_path, monkeypatch, content
):
    expectations = tmp_path / "expectations.json"
    expectations.write_text(content)
    factory = Mock()
    monkeypatch.setattr(mod, "build_evaluation_service", factory)
    result = invoke(
        artifact, "--scorer", "quality", "--max-samples", "1", "--expectations", str(expectations)
    )
    assert result.exit_code != 0
    assert "one per original request case" in " ".join(result.output.split())
    assert "PRIVATE" not in result.output
    factory.assert_not_called()


def test_nonempty_report_directory_is_preserved(artifact, tmp_path, monkeypatch):
    report = tmp_path / "existing"
    report.mkdir()
    (report / "index.html").write_text("old")
    factory = Mock()
    monkeypatch.setattr(mod, "build_evaluation_service", factory)
    result = invoke(artifact, "--traces", "--report-dir", str(report))
    assert result.exit_code != 0
    assert (report / "index.html").read_text() == "old"
    factory.assert_not_called()


def test_preflight_failure_preserves_report_and_actionable_dependency_hint(
    artifact, tmp_path, monkeypatch
):
    service = Mock()
    service.finish.side_effect = AgentCliError(
        "Evaluation dependencies are required.", hint="Install databricks-agentbricks[evaluation]."
    )
    monkeypatch.setattr(mod, "build_evaluation_service", lambda *args, **kwargs: service)
    report = tmp_path / "failed"
    result = invoke(artifact, "--scorer", "quality", "--report-dir", str(report))
    assert result.exit_code == 1
    assert "databricks-agentbricks[evaluation]" in result.stderr
    assert json.loads(result.stdout)["status"] == "failed"
    assert (report / "index.html").is_file()
    assert json.loads((report / "summary.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cancelled_analysis_saves_report_and_signal_exit_code(
    artifact, tmp_path, monkeypatch, signum
):
    service = Mock()

    def interrupted(observations):
        handler = signal.getsignal(signum)
        assert handler is not None and not isinstance(handler, int)
        handler(signum, None)

    service.finish.side_effect = interrupted
    monkeypatch.setattr(mod, "build_evaluation_service", lambda *args, **kwargs: service)
    report = tmp_path / "cancelled"
    original = signal.getsignal(signum)
    result = invoke(artifact, "--traces", "--report-dir", str(report))
    assert result.exit_code == 128 + signum
    assert signal.getsignal(signum) == original
    assert json.loads(result.stdout)["status"] == "cancelled"
    assert (report / "index.html").is_file()


def test_data_loader_accepts_zero_successes_and_validates_unsampled_rows(artifact):
    artifact.write_text('{"schema_version":1,"case_count":1,"observations":[]}')
    assert load_evaluation_data(artifact) == (1, [])
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_count": 1,
                "observations": [
                    {"case_index": 0, "inputs": {}, "outputs": "answer"},
                    {"case_index": 1, "inputs": {}, "outputs": "answer"},
                ],
            }
        )
    )
    with pytest.raises(AgentCliError):
        load_evaluation_data(artifact, max_samples=1)


def test_service_cancellation_retains_completed_trace_reads_without_content():
    client = Mock()
    client.experiment_by_id.return_value = SimpleNamespace(tags={})
    client.browser_url.return_value = None
    client.get_trace.side_effect = [
        SimpleNamespace(data=SimpleNamespace(spans=[])),
        KeyboardInterrupt(),
    ]
    result = EvaluationService("selected", "123", traces=True, client=client).finish(
        [
            {"case_index": 0, "inputs": "PRIVATE", "outputs": "PRIVATE", "trace_id": "a"},
            {"case_index": 0, "inputs": "PRIVATE", "outputs": "PRIVATE", "trace_id": "b"},
        ]
    )
    assert result["status"] == result["traces"]["status"] == "cancelled"
    assert result["traces"]["fetched"] == 1
    assert result["traces"]["coverage"] == 0.5
    assert "PRIVATE" not in json.dumps(result)
    client.get_scorer.assert_not_called()
    client.evaluate.assert_not_called()
