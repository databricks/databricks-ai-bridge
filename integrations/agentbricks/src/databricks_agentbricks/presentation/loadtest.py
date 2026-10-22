"""Human-readable load-test results without changing Locust's JSON stdout."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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


def _mlflow_rows(summary: dict) -> list[tuple[str, str]]:
    mlflow = summary.get("mlflow")
    if not mlflow:
        return []
    if mlflow.get("status") == "not_run":
        return [("MLflow status", f"not run ({mlflow.get('reason', 'unavailable')})")]
    captured = str(mlflow["sampled_successes"])
    if "eligible_successes" in mlflow:
        captured += f"/{mlflow['eligible_successes']} successful requests"
    rows = [
        ("MLflow experiment", str(mlflow["experiment_id"])),
        (
            "Successful captured samples",
            f"{captured} captured; first "
            f"{mlflow['sample_limit']} dispatched load requests eligible",
        ),
    ]
    evaluation = mlflow.get("evaluation", {})
    rows.append(("Evaluation status", str(evaluation.get("status", "disabled"))))
    if evaluation.get("error_types"):
        rows.append(("Evaluation error types", ", ".join(evaluation["error_types"])))
    for scorer in evaluation.get("scorers", []):
        scored = "unknown" if scorer["scored"] is None else str(scorer["scored"])
        rows.append(
            (
                f"Scorer {scorer['name']}",
                f"{scorer['status']}: {scored}/{scorer['selected']} scored, "
                f"{scorer['errors']} errors",
            )
        )
        for name, value in scorer.get("metrics", {}).items():
            rows.append((f"{scorer['name']}: {name}", _number(value)))
        if scorer.get("error_types"):
            rows.append((f"{scorer['name']}: error types", ", ".join(scorer["error_types"])))
    traces = mlflow.get("traces", {})
    rows.append(("Trace status", str(traces.get("status", "disabled"))))
    if traces.get("status") != "disabled":
        rows.append(
            (
                "Trace coverage",
                f"{traces.get('fetched', 0)}/{traces.get('selected', 0)} fetched, "
                f"{traces.get('correlated', 0)} explicitly correlated, "
                f"{traces.get('missing', 0)} missing, {traces.get('errors', 0)} errors",
            )
        )
        for span in traces.get("spans", []):
            rows.append(
                (
                    f"Span type {span['span_type']}",
                    f"{span['count']} spans, {_number(span['total_duration_ms'])}ms total, "
                    f"{_number(span['mean_duration_ms'])}ms average",
                )
            )
        if traces.get("error_types"):
            rows.append(("Trace error types", ", ".join(traces["error_types"])))
    return rows


def _table(rows: list[tuple[str, str]]) -> str:
    return (
        "<table>"
        + "".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows
        )
        + "</table>"
    )


def _mlflow_links(summary: dict) -> list[tuple[str, str]]:
    mlflow = summary.get("mlflow", {})
    urls = [("MLflow evaluation run", mlflow.get("evaluation", {}).get("run_url"))]
    urls.extend(
        (f"MLflow trace {number}", url)
        for number, url in enumerate(mlflow.get("traces", {}).get("urls", []), start=1)
    )
    result = []
    for label, url in urls:
        if not isinstance(url, str):
            continue
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        if parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username:
            result.append((label, url))
    return result


def _mlflow_html(summary: dict, rows: list[tuple[str, str]]) -> str:
    if not rows:
        return ""
    result = "<h2>MLflow results</h2>" + _table(rows)
    if summary["mlflow"].get("status") == "not_run":
        return result
    result += (
        "<p>Requested evaluation and trace reads run after load, on bounded captured successful outputs. "
        "They do not invoke the target again or contribute to performance timings. "
        "Evaluation errors are separate from request failures; metrics describe the scored sample. "
        "Span durations can overlap and do not sum to invocation latency. "
        "Request text, answers, judge rationales, and span contents are omitted from this report.</p>"
    )
    for label, url in _mlflow_links(summary):
        result += f'<p><a href="{escape(url, quote=True)}">{escape(label)}</a></p>'

    return result


def render_loadtest_summary(summary: dict, *, report_dir: Path | None = None) -> None:
    """Print final metrics to stderr and optionally write an HTML report overview."""
    if not summary:
        return
    rows = _rows(summary)
    mlflow_rows = _mlflow_rows(summary)
    errors = summary.get("errors", {})
    click.echo("\nLoad test report", err=True)
    for label, value in rows:
        click.echo(f"  {label}: {value}", err=True)
    if mlflow_rows:
        click.echo("  MLflow results (after load):", err=True)
        for label, value in mlflow_rows:
            click.echo(f"    {label}: {value}", err=True)
        for label, url in _mlflow_links(summary):
            click.echo(f"    {label}: {url}", err=True)
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
        + _mlflow_html(summary, mlflow_rows)
        + "<h2>Request errors</h2>"
        + errors_html
        + "<h2>Details</h2><ul><li><a href='locust.html'>Locust charts and statistics</a></li>"
        "<li><a href='locust_stats.csv'>Locust statistics CSV</a></li>"
        "<li><a href='summary.json'>Summary JSON</a></li></ul></html>",
        encoding="utf-8",
    )
    click.echo(f"  Report: {report_dir / 'index.html'}", err=True)
