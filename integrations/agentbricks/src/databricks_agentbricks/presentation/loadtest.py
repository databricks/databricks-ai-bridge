"""Human-readable overview of the Agent Bricks load-test summary."""

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
    warmup = summary["workload"].get("warmup_requests", 0)
    rows = [
        ("Outcome", str(summary["outcome"])),
        (
            "Workload",
            f"{workload['concurrency']} concurrent users, "
            f"{workload['duration_seconds']}s requested, "
            f"{workload.get('max_requests') or 'unlimited'} requests, {cases} cases",
        ),
        (
            "Warm-up",
            "1 real request excluded from counts and timings; cold-start latency excluded"
            if warmup
            else "Disabled; all requests measured, including any cold-start latency",
        ),
        ("Observed peak in flight", str(summary.get("observed_peak_in_flight", "unknown"))),
        (
            "Requests",
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
    return rows


def _table(rows: list[tuple[str, str]]) -> str:
    return (
        "<table>"
        + "".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows
        )
        + "</table>"
    )


def render_loadtest_summary(
    summary: dict, *, report_dir: Path | None = None, output: str = "text"
) -> None:
    """Write the HTML overview and print metrics when text output is selected."""
    if not summary:
        return
    rows = _rows(summary)
    errors = summary.get("errors", {})
    if output == "text":
        click.echo("\nLoad test report", err=True)
        for label, value in rows:
            click.echo(f"  {label}: {value}", err=True)
        if summary.get("generator_cpu_warning"):
            click.echo(
                "  Generator CPU exceeded Locust's limit; results may be generator-limited.",
                err=True,
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
        + (
            "The real warm-up request, including its cold-start latency, is excluded from counts and timings. "
            if summary["workload"].get("warmup_requests", 0)
            else "Warm-up is disabled; all requests are measured, including any cold-start latency. "
        )
        + "Locust histogram percentiles are rounded; all-completed timings include failures. "
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
    if output == "text":
        click.echo(f"  Report: {report_dir / 'index.html'}", err=True)
