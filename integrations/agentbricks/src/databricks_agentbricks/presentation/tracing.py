"""Shared tracing presentation values used by deploy output."""

from __future__ import annotations

from typing import Optional

from databricks_agentbricks.presentation import render
from databricks_agentbricks.services.tracing_service import (
    TraceGetResult,
    TraceListResult,
    TraceSummary,
)
from databricks_agentkit import timefmt

# The command that binds (enables) tracing. Referenced parameter-free by deploy's "deployed without
# tracing" guidance, so the hint cannot go stale if the flags change; the command's own ``--help``
# documents the flags.
TRACING_BIND_COMMAND = "agentbricks tracing bind"


def experiment_url(host: Optional[str], experiment_id: str) -> Optional[str]:
    """The workspace MLflow experiment Traces page, or None when the host is unavailable."""
    if not host or host == "unknown":
        return None
    return f"{host.rstrip('/')}/ml/experiments/{experiment_id}?compareRunsMode=TRACES"


def _summary_json(summary: TraceSummary) -> dict:
    return {
        "trace_id": summary.trace_id,
        "status": summary.status,
        "execution_time_ms": summary.execution_time_ms,
        "timestamp_ms": summary.timestamp_ms,
    }


def _show_warning(warning: str | None, help: str | None) -> None:
    if warning is not None:
        render.diagnostic("warning", warning, help=help)


def show_trace_list(result: TraceListResult, output: str) -> None:
    _show_warning(result.warning, result.help)
    if output == "json":
        render.emit_json([_summary_json(summary) for summary in result.traces])
        return
    rows = [
        [
            summary.trace_id,
            render.status_pill(summary.status),
            summary.execution_time_ms,
            timefmt.relative(summary.timestamp_ms),
        ]
        for summary in result.traces
    ]
    where = " (local dev)" if result.local else ""
    render.resource_table(
        f"Agent Traces · {str(result.experiment_id) + where if result.experiment_id else 'no experiment yet'}",
        [
            ("Trace ID", "left"),
            ("Status", "left"),
            ("Latency (ms)", "left"),
            ("Created", "left"),
        ],
        rows,
    )


def show_trace_detail(result: TraceGetResult, output: str) -> None:
    _show_warning(result.warning, result.help)
    detail = result.detail
    summary = detail.summary
    if output == "json":
        render.emit_json(_summary_json(summary))
        return
    render.detail(
        "Agent Tracing",
        result.trace_id,
        {
            "Status": summary.status,
            "Latency (ms)": summary.execution_time_ms,
            "Spans": detail.span_count,
            "Request": detail.request,
            "Response": detail.response,
            "Created": timefmt.absolute(summary.timestamp_ms),
        },
        status=summary.status,
    )
