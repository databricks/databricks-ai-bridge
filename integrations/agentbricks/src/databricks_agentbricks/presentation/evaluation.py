"""Content-free MLflow evaluation reports, separate from load-test performance."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import click


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


def _rows(summary: dict) -> list[tuple[str, str]]:
    mlflow = summary
    captured = str(mlflow["sampled_successes"])
    rows = [
        ("Outcome", str(mlflow.get("status", "completed"))),
        ("MLflow experiment", str(mlflow["experiment_id"])),
        (
            "Successful captured samples",
            f"{captured} captured responses selected (limit {mlflow['sample_limit']})",
        ),
    ]
    if summary.get("error_types"):
        rows.append(("Analysis error types", ", ".join(summary["error_types"])))
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


def _links(summary: dict) -> list[tuple[str, str]]:
    mlflow = summary
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


def render_evaluation_summary(summary: dict, *, report_dir: Path | None = None) -> None:
    """Print analysis coverage to stderr and optionally write a shareable HTML report."""
    rows = _rows(summary)
    click.echo("\nEvaluation report", err=True)
    for label, value in rows:
        click.echo(f"  {label}: {value}", err=True)
    links = _links(summary)
    for label, url in links:
        click.echo(f"  {label}: {url}", err=True)
    if report_dir is None:
        return
    links_html = "".join(
        f'<p><a href="{escape(url, quote=True)}">{escape(label)}</a></p>' for label, url in links
    )
    (report_dir / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Agent Bricks evaluation</title><style>"
        "body{font:16px system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;"
        "color:#202124}table{border-collapse:collapse;width:100%;margin:20px 0}"
        "th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #ddd;"
        "overflow-wrap:anywhere}th{background:#f6f7f8}a{color:#0759ac}"
        "</style><h1>Evaluation report</h1>"
        + _table(rows)
        + "<p>Analysis uses saved responses and never invokes the target agent. "
        "Metrics describe the scored sample. Trace reads use explicit trace IDs only. "
        "Span durations can overlap and do not sum to invocation latency. "
        "Request text, answers, judge rationales, and span contents are omitted from this report.</p>"
        + links_html
        + "<p><a href='summary.json'>Summary JSON</a></p></html>",
        encoding="utf-8",
    )
    click.echo(f"  Report: {report_dir / 'index.html'}", err=True)
