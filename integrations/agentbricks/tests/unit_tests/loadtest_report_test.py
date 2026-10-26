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


def test_json_output_still_writes_html_without_printing_text(summary, tmp_path, capsys):
    render_loadtest_summary(summary, report_dir=tmp_path, output="json")

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert "Load test report" in (tmp_path / "index.html").read_text()


@pytest.mark.parametrize("warmup", [False, True])
def test_report_explains_warmup_and_cold_start_accounting(summary, tmp_path, capsys, warmup):
    summary["workload"]["warmup_requests"] = int(warmup)

    render_loadtest_summary(summary, report_dir=tmp_path)

    captured = capsys.readouterr()
    assert "Requests: 6 started, 5 completed" in captured.err
    html = (tmp_path / "index.html").read_text()
    if warmup:
        assert "1 real request excluded" in captured.err
        assert "warm-up request, including its cold-start latency, is excluded" in html
    else:
        assert "Disabled; all requests measured" in captured.err
        assert "Warm-up is disabled; all requests are measured" in html
