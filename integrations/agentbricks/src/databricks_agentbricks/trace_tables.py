"""Shared data types and discovery for a UC-backed experiment's trace tables.

Lives at the package top - NOT under ``databricks_agentbricks.cli`` - so lower-level modules like
``app_resources`` can name these without importing the CLI package. Importing anything from
``databricks_agentbricks.cli`` runs ``cli/__init__`` -> ``cli.app`` -> ``cli.deploy`` -> ``app_resources``,
so a ``cli`` import from ``app_resources`` would be circular (an import-order-dependent failure).
Keeping the table model and tag parsing here lets deploy collaborators discover their grants without
importing the Click command package. ``cli.tracing`` re-exports these names for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class TraceTableKind(str, Enum):
    """The kind of UC OTEL base table backing an experiment's traces.

    A ``str`` mixin so a member compares equal to its literal value; use ``.value`` when building
    strings (e.g. the app-resource name) so the rendering is the bare kind, not ``TraceTableKind.X``.
    """

    SPANS = "spans"
    LOGS = "logs"
    ANNOTATIONS = "annotations"
    METRICS = "metrics"


@dataclass(frozen=True)
class TraceTable:
    """One UC OTEL base table for a trace experiment - its ``kind`` and its fully-qualified
    ``catalog.schema.table`` name."""

    kind: TraceTableKind
    full_name: str


# An experiment linked to a UC schema carries this destination tag. Managed experiments do not.
_UC_TRACE_TAG = "mlflow.experiment.databricksTraceDestinationPath"
_UC_TRACE_SPAN_TAG = "mlflow.experiment.databricksTraceSpanStorageTable"
_UC_TRACE_LOG_TAG = "mlflow.experiment.databricksTraceLogStorageTable"
_UC_TRACE_ANNOTATION_TAG = "mlflow.experiment.databricksTraceAnnotationStorageTable"
_UC_TRACE_METRIC_TAG = "mlflow.experiment.databricksTraceMetricStorageTable"


@dataclass(frozen=True)
class MLflowTraceTables:
    """UC OTEL base tables backing an experiment; fields are absent for managed storage."""

    spans: Optional[str] = None
    logs: Optional[str] = None
    annotations: Optional[str] = None
    metrics: Optional[str] = None

    def otel_tables(self) -> list[TraceTable]:
        """Return present base tables in the stable order used for app-resource grants."""
        return [
            TraceTable(kind=kind, full_name=name)
            for kind, name in (
                (TraceTableKind.SPANS, self.spans),
                (TraceTableKind.LOGS, self.logs),
                (TraceTableKind.ANNOTATIONS, self.annotations),
                (TraceTableKind.METRICS, self.metrics),
            )
            if name
        ]

    def __bool__(self) -> bool:
        return bool(self.otel_tables())


def _metrics_table_from_destination(tags: dict[str, Any]) -> Optional[str]:
    """Derive the metrics base-table name because MLflow does not currently tag it directly."""
    parts = (tags.get(_UC_TRACE_TAG) or "").split(".")
    if len(parts) == 3:
        return f"{'.'.join(parts)}_otel_metrics"
    if len(parts) == 2:
        return f"{'.'.join(parts)}.mlflow_experiment_trace_otel_metrics"
    return None


def uc_trace_tables(experiment: Any) -> MLflowTraceTables:
    """Discover the writable UC OTEL base tables for an MLflow experiment.

    Per-kind tags are authoritative when present. Older layouts expose only a destination path, in
    which case the conventional table names are derived. Managed experiments return an empty value.
    """
    tags = getattr(experiment, "tags", None) or {}
    if _UC_TRACE_TAG not in tags:
        return MLflowTraceTables()
    tables = MLflowTraceTables(
        spans=tags.get(_UC_TRACE_SPAN_TAG) or None,
        logs=tags.get(_UC_TRACE_LOG_TAG) or None,
        annotations=tags.get(_UC_TRACE_ANNOTATION_TAG) or None,
        metrics=tags.get(_UC_TRACE_METRIC_TAG) or _metrics_table_from_destination(tags),
    )
    if tables.spans or tables.logs or tables.annotations:
        return tables
    parts = (tags.get(_UC_TRACE_TAG) or "").split(".")
    if len(parts) == 3:
        prefix = ".".join(parts)
        return MLflowTraceTables(
            spans=f"{prefix}_otel_spans",
            logs=f"{prefix}_otel_logs",
            annotations=f"{prefix}_otel_annotations",
            metrics=f"{prefix}_otel_metrics",
        )
    if len(parts) == 2:
        base = ".".join(parts)
        return MLflowTraceTables(
            spans=f"{base}.mlflow_experiment_trace_otel_spans",
            logs=f"{base}.mlflow_experiment_trace_otel_logs",
            annotations=f"{base}.mlflow_experiment_trace_otel_annotations",
            metrics=f"{base}.mlflow_experiment_trace_otel_metrics",
        )
    return MLflowTraceTables()
