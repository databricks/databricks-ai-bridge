"""Human-readable load-test results without changing Locust's JSON stdout."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import click


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


def _latency(stats: dict) -> str:
    if not stats.get("count"):
        return "n/a (0 samples)"
    return (
        ", ".join(
            f"{label} {_number(stats.get(key))}"
            for key, label in (
                ("p50", "p50"),
                ("p95", "p95"),
                ("p99", "p99"),
                ("mean", "avg"),
                ("min", "min"),
                ("max", "max"),
            )
        )
        + f" ({stats['count']} samples)"
    )


def _rows(summary: dict) -> list[tuple[str, str]]:
    workload = summary["workload"]
    counts = summary["counts"]
    coverage = summary["ttft_coverage"]
    usage = summary["usage"]
    completed = counts["completed"]
    failure_rate = f"{counts['failed'] / completed:.2%}" if completed else "n/a"
    cases = workload["cases"]
    if isinstance(cases, list):
        cases = len(cases)
    rows = [
        ("Outcome", str(summary["outcome"])),
        (
            "Workload",
            f"{workload['concurrency']} concurrent users, "
            f"{workload['duration_seconds']}s requested, "
            f"{workload.get('max_requests') or 'unlimited'} requests, {cases} cases",
        ),
        ("Observed peak in flight", str(summary.get("observed_peak_in_flight", "unknown"))),
        (
            "Invocations" if workload.get("follow_events") else "Requests",
            f"{counts['started']} started, {completed} completed, "
            f"{counts['succeeded']} succeeded, {counts['failed']} failed "
            f"({failure_rate} of completed), {counts['incomplete']} incomplete",
        ),
        ("Elapsed including drain", f"{_number(summary['elapsed_seconds_including_drain'])}s"),
        (
            "Throughput including drain",
            f"{_number(summary['completed_rps'])} completed/s, "
            f"{_number(summary['successful_rps'])} successful/s",
        ),
        ("Successful latency (ms)", _latency(summary["latency_ms"]["successful"])),
        ("All completed latency (ms; Locust)", _latency(summary["latency_ms"]["all_completed"])),
        ("Answer-text TTFT (ms)", _latency(summary["ttft_ms"])),
        (
            "First-content coverage",
            f"{coverage['measured']}/{coverage['eligible_successes']} successful responses measured, "
            f"{coverage['unknown']} unknown",
        ),
        (
            "Observed output tokens",
            f"{usage['output_tokens'] if usage['output_tokens'] is not None else 'unknown'} "
            f"(usage reported by {usage['observed_successes']}/{usage['eligible_successes']} "
            "successful responses)",
        ),
    ]
    if workload.get("follow_events"):
        rows.insert(
            2,
            (
                "Invocation journey",
                "POST submission → GET event stream → GET final status; one counted invocation. "
                "Latency covers all three requests; answer-text TTFT starts at POST submission.",
            ),
        )
    return rows


def _table(rows: list[tuple[str, str]]) -> str:
    return (
        "<table>"
        + "".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows
        )
        + "</table>"
    )


def render_loadtest_summary(summary: dict, *, report_dir: Path | None = None) -> None:
    """Print final metrics to stderr and optionally write an HTML report overview."""
    if not summary:
        return
    rows = _rows(summary)
    errors = summary.get("errors", {})
    click.echo("\nLoad test report", err=True)
    for label, value in rows:
        click.echo(f"  {label}: {value}", err=True)
    if summary.get("generator_cpu_warning"):
        click.echo(
            "  Generator CPU exceeded Locust's limit; results may be generator-limited.", err=True
        )
    if errors:
        click.echo("  Errors:", err=True)
        for error, count in errors.items():
            click.echo(f"    {count}: {error}", err=True)
    if report_dir is None:
        return

    error_rows = "".join(
        f"<tr><td>{escape(str(error))}</td><td>{escape(str(count))}</td></tr>"
        for error, count in errors.items()
    )
    errors_html = (
        "<table><tr><th>Error</th><th>Count</th></tr>" + error_rows + "</table>"
        if errors
        else "<p>No errors recorded.</p>"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Agent Bricks load test</title><style>"
        "body{font:16px system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;"
        "color:#202124}table{border-collapse:collapse;width:100%;margin:20px 0}"
        "th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #ddd;"
        "overflow-wrap:anywhere}th{background:#f6f7f8}a{color:#0759ac}"
        "</style><h1>Load test report</h1>"
        + _table(rows)
        + "<p>Latency measures the full response, including streaming completion. "
        "Throughput uses elapsed time including in-flight request drain. "
        "One preflight request is excluded from counts and timings. "
        "Locust histogram percentiles are rounded; all-completed timings include failures. "
        "Performance completion does not establish sustainable capacity or answer quality. "
        "Token totals include only usage reported by successful responses; "
        "unknown first-content times are not treated as zero.</p>"
        + "<h2>Request errors</h2>"
        + errors_html
        + "<h2>Details</h2><ul><li><a href='locust.html'>Locust charts and statistics</a></li>"
        "<li><a href='locust_stats.csv'>Locust statistics CSV</a></li>"
        "<li><a href='summary.json'>Summary JSON</a></li></ul></html>",
        encoding="utf-8",
    )
    click.echo(f"  Report: {report_dir / 'index.html'}", err=True)
