from __future__ import annotations

import pytest

from databricks_agentbricks.presentation.evaluation import render_evaluation_summary


@pytest.fixture
def mlflow_summary():
    summary = {
        "experiment_id": "123",
        "sample_limit": 3,
        "sampled_successes": 2,
        "status": "failed",
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
    render_evaluation_summary(mlflow_summary, report_dir=tmp_path)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Outcome: failed" in captured.err
    assert "2 captured responses selected (limit 3)" in captured.err
    assert "Evaluation status: partial" in captured.err
    assert "Scorer correctness: partial: 1/2 scored, 1 errors" in captured.err
    assert "correctness/mean: 0.75" in captured.err
    assert (
        "Trace coverage: 1/2 fetched, 1 explicitly correlated, 1 missing, 0 errors" in captured.err
    )
    assert "Span type LLM: 2 spans, 250.00ms total, 125.00ms average" in captured.err
    html = (tmp_path / "index.html").read_text()
    assert "never invokes the target agent" in html
    assert "Span durations can overlap" in html
    assert 'href="https://workspace.example/ml/experiments/123/runs/abc"' in html
    assert 'href="https://workspace.example/ml/experiments/123/traces/tr-abc"' in html


def test_mlflow_report_escapes_labels_and_omits_raw_contents_and_unsafe_links(
    mlflow_summary, tmp_path, capsys
):
    evaluation = mlflow_summary["evaluation"]
    evaluation["scorers"][0]["name"] = "<script>scorer</script>"
    evaluation["run_url"] = "javascript:alert(1)"
    evaluation["raw_results"] = [{"answer": "PRIVATE ANSWER", "rationale": "PRIVATE RATIONALE"}]
    mlflow_summary["traces"]["urls"] = [
        "https://username:secret@workspace.example/trace",
        "https://[invalid",
    ]
    mlflow_summary["traces"]["spans"][0]["name"] = "PRIVATE PROMPT"

    render_evaluation_summary(mlflow_summary, report_dir=tmp_path)

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


def test_unknown_mlflow_scoring_coverage_is_not_reported_as_zero(mlflow_summary, capsys):
    evaluation = mlflow_summary["evaluation"]
    evaluation["scorers"][0]["scored"] = None
    evaluation["error_types"] = ["TrackingError"]

    render_evaluation_summary(mlflow_summary)

    captured = capsys.readouterr()
    assert "unknown/2 scored" in captured.err
    assert "Evaluation error types: TrackingError" in captured.err
