from __future__ import annotations

import pytest

from databricks_agentbricks.presentation.loadtest import render_loadtest_summary


@pytest.fixture
def summary():
    latency = {"count": 4, "min": 10, "mean": 15, "max": 20, "p50": 15, "p95": 20, "p99": 20}
    return {
        "schema_version": 1,
        "workload": {"concurrency": 2, "duration_seconds": 5, "max_requests": 8, "cases": 2},
        "outcome": "failed",
        "counts": {"started": 6, "completed": 5, "succeeded": 4, "failed": 1, "incomplete": 1},
        "elapsed_seconds_including_drain": 6,
        "completed_rps": 5 / 6,
        "successful_rps": 4 / 6,
        "latency_ms": {"all_completed": {**latency, "count": 5}, "successful": latency},
        "ttft_ms": {**latency, "count": 3},
        "ttft_coverage": {"measured": 3, "unknown": 1, "eligible_successes": 4},
        "usage": {"observed_successes": 2, "eligible_successes": 4, "output_tokens": 30},
        "errors": {"HTTP 503": 1},
    }


def test_report_keeps_stdout_empty_and_distinguishes_failed_and_incomplete(summary, capsys):
    render_loadtest_summary(summary)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "1 failed (20.00% of completed), 1 incomplete" in captured.err
    assert "0.83 completed/s, 0.67 successful/s" in captured.err
    assert "Successful latency (ms)" in captured.err
    assert "All completed latency (ms; Locust)" in captured.err
    assert "3/4 successful responses measured, 1 unknown" in captured.err
    assert "30 (usage reported by 2/4 successful responses)" in captured.err
    assert "1: HTTP 503" in captured.err


def test_html_escapes_dynamic_fields_and_links_to_native_artifacts(summary, tmp_path):
    summary["outcome"] = '<script>alert("outcome")</script>'
    summary["errors"] = {'<img src=x onerror="alert(1)">': 1}

    render_loadtest_summary(summary, report_dir=tmp_path)

    html = (tmp_path / "index.html").read_text()
    assert "<script>" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    for artifact in ("locust.html", "locust_stats.csv", "summary.json"):
        assert f"href='{artifact}'" in html
    assert "unknown first-content times are not treated as zero" in html


def test_zero_completed_reports_unknown_percent_and_latency(summary, capsys):
    summary["counts"].update(completed=0, succeeded=0, failed=0, incomplete=6)
    summary["latency_ms"]["successful"] = {"count": 0}
    summary["latency_ms"]["all_completed"] = {"count": 0}
    summary["ttft_ms"] = {"count": 0}

    render_loadtest_summary(summary)

    captured = capsys.readouterr()
    assert "0 failed (n/a of completed), 6 incomplete" in captured.err
    assert "Successful latency (ms): n/a (0 samples)" in captured.err
    assert "nan" not in captured.err


def test_missing_child_summary_does_not_replace_child_failure(tmp_path, capsys):
    render_loadtest_summary({}, report_dir=tmp_path)

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert not (tmp_path / "index.html").exists()


@pytest.fixture
def mlflow_summary(summary):
    summary["mlflow"] = {
        "experiment_id": "123",
        "sample_limit": 3,
        "sampled_successes": 2,
        "eligible_successes": 4,
        "evaluation": {
            "status": "partial",
            "run_url": "https://workspace.example/ml/experiments/123/runs/abc",
            "scorers": [
                {
                    "name": "correctness",
                    "status": "partial",
                    "selected": 2,
                    "scored": 1,
                    "errors": 1,
                    "metrics": {"correctness/mean": 0.75},
                    "error_types": ["JudgeError"],
                }
            ],
        },
        "traces": {
            "status": "partial",
            "selected": 2,
            "correlated": 1,
            "fetched": 1,
            "missing": 1,
            "errors": 0,
            "spans": [
                {"span_type": "LLM", "count": 2, "total_duration_ms": 250, "mean_duration_ms": 125}
            ],
            "urls": ["https://workspace.example/ml/experiments/123/traces/tr-abc"],
        },
    }
    return summary


def test_mlflow_report_distinguishes_sampled_evaluation_and_trace_coverage(
    mlflow_summary, tmp_path, capsys
):
    render_loadtest_summary(mlflow_summary, report_dir=tmp_path)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Outcome: failed" in captured.err
    assert (
        "2/4 successful requests captured; first 3 dispatched load requests eligible"
        in captured.err
    )
    assert "Evaluation status: partial" in captured.err
    assert "Scorer correctness: partial: 1/2 scored, 1 errors" in captured.err
    assert "correctness/mean: 0.75" in captured.err
    assert (
        "Trace coverage: 1/2 fetched, 1 explicitly correlated, 1 missing, 0 errors" in captured.err
    )
    assert "Span type LLM: 2 spans, 250.00ms total, 125.00ms average" in captured.err
    html = (tmp_path / "index.html").read_text()
    assert "Evaluation errors are separate from request failures" in html
    assert "Span durations can overlap" in html
    assert 'href="https://workspace.example/ml/experiments/123/runs/abc"' in html
    assert 'href="https://workspace.example/ml/experiments/123/traces/tr-abc"' in html


def test_mlflow_report_escapes_labels_and_omits_raw_contents_and_unsafe_links(
    mlflow_summary, tmp_path, capsys
):
    evaluation = mlflow_summary["mlflow"]["evaluation"]
    evaluation["scorers"][0]["name"] = "<script>scorer</script>"
    evaluation["run_url"] = "javascript:alert(1)"
    evaluation["raw_results"] = [{"answer": "PRIVATE ANSWER", "rationale": "PRIVATE RATIONALE"}]
    mlflow_summary["mlflow"]["traces"]["urls"] = [
        "https://username:secret@workspace.example/trace",
        "https://[invalid",
    ]
    mlflow_summary["mlflow"]["traces"]["spans"][0]["name"] = "PRIVATE PROMPT"

    render_loadtest_summary(mlflow_summary, report_dir=tmp_path)

    captured = capsys.readouterr()
    html = (tmp_path / "index.html").read_text()
    assert "&lt;script&gt;scorer&lt;/script&gt;" in html
    assert "<script>" not in html
    for secret in (
        "PRIVATE ANSWER",
        "PRIVATE RATIONALE",
        "PRIVATE PROMPT",
        "javascript:",
        "username:secret",
    ):
        assert secret not in html
        assert secret not in captured.err


def test_cancelled_load_reports_mlflow_not_run_without_replacing_performance(summary, capsys):
    summary["outcome"] = "cancelled"
    summary["mlflow"] = {"status": "not_run", "reason": "cancelled"}

    render_loadtest_summary(summary)

    captured = capsys.readouterr()
    assert "Outcome: cancelled" in captured.err
    assert "MLflow status: not run (cancelled)" in captured.err
    assert "Successful latency (ms)" in captured.err


def test_follow_events_report_counts_invocations_and_explains_full_journey(
    summary, tmp_path, capsys
):
    summary["workload"]["follow_events"] = True

    render_loadtest_summary(summary, report_dir=tmp_path)

    captured = capsys.readouterr()
    assert "Invocations: 6 started, 5 completed" in captured.err
    assert "one counted invocation" in captured.err
    assert "answer-text TTFT starts at POST submission" in captured.err
    html = (tmp_path / "index.html").read_text()
    assert "POST submission → GET event stream → GET final status" in html
    assert "Latency covers all three requests" in html


def test_unknown_mlflow_scoring_coverage_is_not_reported_as_zero(mlflow_summary, capsys):
    evaluation = mlflow_summary["mlflow"]["evaluation"]
    evaluation["scorers"][0]["scored"] = None
    evaluation["error_types"] = ["TrackingError"]

    render_loadtest_summary(mlflow_summary)

    captured = capsys.readouterr()
    assert "unknown/2 scored" in captured.err
    assert "Evaluation error types: TrackingError" in captured.err
