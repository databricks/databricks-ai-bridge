"""Framework-neutral workflows behind ``agentbricks tracing``."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from databricks_agentbricks.clients.mlflow_trace_client import MLflowTraceClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.tracing.targets import TraceTargetResolver


@dataclass(frozen=True)
class TraceReadRequest:
    source: pathlib.Path
    experiment_name: str | None = None
    experiment_id: str | None = None
    warehouse_id: str | None = None


@dataclass(frozen=True)
class TraceSummary:
    trace_id: Any
    status: str | None
    execution_time_ms: Any
    timestamp_ms: Any


@dataclass(frozen=True)
class TraceDetail:
    summary: TraceSummary
    span_count: int
    request: Any
    response: Any


@dataclass(frozen=True)
class TraceListResult:
    traces: tuple[TraceSummary, ...]
    experiment_id: str | None
    local: bool
    warning: str | None = None
    help: str | None = None


@dataclass(frozen=True)
class TraceGetResult:
    trace_id: str
    detail: TraceDetail
    warning: str | None = None
    help: str | None = None


def _attr(obj: Any, *paths: str, default: Any = None) -> Any:
    for path in paths:
        current = obj
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if current is not None:
            return current
    return default


def _status_str(status: Any) -> Optional[str]:
    if status is None:
        return None
    return getattr(status, "name", None) or str(status)


def _summary(trace: Any) -> TraceSummary:
    return TraceSummary(
        trace_id=_attr(trace, "info.trace_id", "info.request_id"),
        status=_status_str(_attr(trace, "info.status", "info.state")),
        execution_time_ms=_attr(trace, "info.execution_time_ms", "info.execution_duration_ms"),
        timestamp_ms=_attr(trace, "info.timestamp_ms", "info.request_time"),
    )


class TracingService:
    """Coordinate trace reads without Click or presentation dependencies."""

    def __init__(self, mlflow: MLflowTraceClient, targets: TraceTargetResolver) -> None:
        self._mlflow = mlflow
        self._targets = targets

    def list(self, request: TraceReadRequest, limit: int) -> TraceListResult:
        warehouse_id = self._mlflow.resolve_warehouse_id(request.warehouse_id)
        with self._targets.resolve(
            request.source, request.experiment_name, request.experiment_id, warehouse_id
        ) as target:
            traces = (
                self._mlflow.search_traces(target.experiment_id, limit)
                if target.experiment_id
                else []
            )
            return TraceListResult(
                traces=tuple(_summary(trace) for trace in traces),
                experiment_id=target.experiment_id,
                local=target.local,
                warning=target.warning,
                help=target.help,
            )

    def get(
        self,
        request: TraceReadRequest,
        trace_id: str,
        on_warning: Callable[[str, str | None], None] | None = None,
    ) -> TraceGetResult:
        warehouse_id = self._mlflow.resolve_warehouse_id(request.warehouse_id)
        with self._targets.resolve(
            request.source, request.experiment_name, request.experiment_id, warehouse_id
        ) as target:
            # Surface a failed local read server before the workspace fallback. The fallback may
            # raise or find no trace, in which case there is no result to carry the warning.
            if target.warning and on_warning:
                on_warning(target.warning, target.help)
            if target.tracking_uri is None:
                self._mlflow.select_fallback_workspace(warehouse_id)
            trace = self._mlflow.get_trace(trace_id)
            if trace is None:
                raise AgentCliError(f"No trace found with id {trace_id!r}.")
            detail = TraceDetail(
                summary=_summary(trace),
                span_count=len(_attr(trace, "data.spans", default=[]) or []),
                request=_attr(trace, "info.request_preview", "data.request"),
                response=_attr(trace, "info.response_preview", "data.response"),
            )
            return TraceGetResult(
                trace_id,
                detail,
                warning=target.warning if on_warning is None else None,
                help=target.help if on_warning is None else None,
            )
