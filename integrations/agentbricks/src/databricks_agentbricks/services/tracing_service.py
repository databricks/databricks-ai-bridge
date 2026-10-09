"""Framework-neutral workflows behind ``agentbricks tracing``."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Optional

from databricks_agentbricks.clients.mlflow_trace_client import MLflowTraceClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
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


def _check_experiment_flags(
    experiment_name: Optional[str], experiment_id: Optional[str], *, require_one: bool = False
) -> None:
    if experiment_name and experiment_id:
        raise AgentCliError(
            "Pass --experiment-name or --experiment-id, not both.",
            hint="They select the same experiment; use whichever identifier you have.",
        )
    if require_one and not (experiment_name or experiment_id):
        raise AgentCliError(
            "Pass --experiment-name or --experiment-id to bind tracing to an experiment.",
            hint="Absence of a bound experiment means tracing is off; `agentbricks tracing unbind` clears it.",
        )


class TracingService:
    """Coordinate project bindings and trace reads without Click or presentation dependencies."""

    def __init__(self, mlflow: MLflowTraceClient, targets: TraceTargetResolver) -> None:
        self._mlflow = mlflow
        self._targets = targets

    def bind(
        self,
        source: pathlib.Path,
        experiment_name: str | None,
        experiment_id: str | None,
    ) -> str:
        _check_experiment_flags(experiment_name, experiment_id, require_one=True)
        name = experiment_name
        if experiment_id:
            self._mlflow.select_workspace()
            experiment = self._mlflow.experiment_by_id(experiment_id)
            if experiment is None:
                raise AgentCliError(
                    f"No MLflow experiment found with id {experiment_id!r}.",
                    hint="Pass an existing experiment id, or use --experiment-name.",
                )
            name = experiment.name
        elif name and not name.startswith("/"):
            raise AgentCliError(
                f"Experiment name must be an absolute workspace path, got {name!r}.",
                hint="Use a path like /Shared/agentbricks_traces/<agent> or "
                "/Users/<you>/agentbricks_traces/<agent>.",
            )
        assert name is not None
        project = AgentProject.load(source)
        project.bind_tracing(name)
        project.write()
        return name

    def unbind(self, source: pathlib.Path) -> None:
        project = AgentProject.load(source)
        project.unbind_tracing()
        project.write()

    def list(self, request: TraceReadRequest, limit: int) -> TraceListResult:
        _check_experiment_flags(request.experiment_name, request.experiment_id)
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

    def get(self, request: TraceReadRequest, trace_id: str) -> TraceGetResult:
        _check_experiment_flags(request.experiment_name, request.experiment_id)
        warehouse_id = self._mlflow.resolve_warehouse_id(request.warehouse_id)
        with self._targets.resolve(
            request.source, request.experiment_name, request.experiment_id, warehouse_id
        ) as target:
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
            return TraceGetResult(trace_id, detail, warning=target.warning, help=target.help)
