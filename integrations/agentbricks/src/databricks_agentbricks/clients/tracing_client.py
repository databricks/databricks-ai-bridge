"""MLflow experiment discovery and Apps trace-resource reconciliation.

This module owns the tracing adapter boundary shared by deploy and tracing commands. MLflow is
imported lazily; the injected client remains render-free and leaves deployment phase decisions to
the tracing provisioner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.apps_client import AppsClient


class TraceTableKind(str, Enum):
    """The kind of UC OTEL base table backing an experiment's traces."""

    SPANS = "spans"
    LOGS = "logs"
    ANNOTATIONS = "annotations"
    METRICS = "metrics"


@dataclass(frozen=True)
class TraceTable:
    """One UC OTEL base table and its fully qualified name."""

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


def _mlflow():
    """Import MLflow lazily so unrelated CLI commands do not pay its startup cost."""
    import mlflow  # noqa: PLC0415 - intentional lazy import

    return mlflow


def _workspace_uri(profile: Optional[str]) -> str:
    """MLflow tracking URI for the selected Databricks profile."""
    return f"databricks://{profile}" if profile else "databricks"


def _set_tracking_uri(mlflow, profile: Optional[str]) -> None:
    """Point MLflow at the selected workspace."""
    mlflow.set_tracking_uri(_workspace_uri(profile))


@dataclass(frozen=True)
class ResolvedTraceExperiment:
    """A workspace experiment id and any UC OTEL base tables backing its traces."""

    experiment_id: str
    tables: MLflowTraceTables


def create_experiment_idempotent(
    profile: Optional[str], client: Any, name: str
) -> ResolvedTraceExperiment:
    """Resolve an experiment by name or create it, including its parent directory when needed."""
    mlflow = _mlflow()
    _set_tracking_uri(mlflow, profile)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment:
        return ResolvedTraceExperiment(
            experiment_id=experiment.experiment_id, tables=uc_trace_tables(experiment)
        )
    parent = name.rsplit("/", 1)[0]
    if parent:
        client.ensure_workspace_dir(parent)
    return ResolvedTraceExperiment(
        experiment_id=mlflow.create_experiment(name), tables=MLflowTraceTables()
    )


_TRACE_EXPERIMENT_RESOURCE = "agentbricks-trace-experiment"
_TRACE_RESOURCE_PREFIX = "agentbricks-trace-"


class TracingClient:
    """Resolve a project's bound tracing experiment and grant the app access to it, for a ``profile``.

    Holds the per-command API client provider, invoked lazily on first use, so its methods take
    only the resource names.
    """

    def __init__(
        self,
        api_client_provider: ApiClientProvider,
        apps_client: AppsClient,
        profile: Optional[str],
    ) -> None:
        self._api_client_provider = api_client_provider
        self._apps_client = apps_client
        self._profile = profile

    def ensure_experiment(
        self, experiment_name: Optional[str]
    ) -> Optional[ResolvedTraceExperiment]:
        """Get or create the explicitly bound experiment, or return None when tracing is unbound.

        Project configuration is resolved by the caller; this client owns only the MLflow operation.
        Resolution is by name, never a workspace-local stored id.
        """
        if not experiment_name:
            return None
        return create_experiment_idempotent(
            self._profile, self._api_client_provider.get(), experiment_name
        )

    def reconcile_app_resources(
        self,
        name: str,
        experiment_id: Optional[str],
        otel_tables: Sequence[TraceTable],
    ) -> Optional[str]:
        """Reconcile the app's agentbricks-owned trace resources to the desired state.

        Builds the agentbricks-owned trace resources, then delegates safe array reconciliation to
        ``AppsClient``. Grants the app's service principal the ``experiment`` resource (and MODIFY
        on any UC OTEL tables), or prunes every ``agentbricks-trace-`` resource on a clean unbind
        (``experiment_id`` is None). Returns None on success, else a human-readable reason.
        """
        desired: list[dict] = []
        if experiment_id is not None:
            desired.append(
                {
                    "name": _TRACE_EXPERIMENT_RESOURCE,
                    "experiment": {"experiment_id": experiment_id, "permission": "CAN_EDIT"},
                }
            )
        desired.extend(
            [
                {
                    "name": f"{_TRACE_RESOURCE_PREFIX}{table.kind.value}",
                    "uc_securable": {
                        "securable_full_name": table.full_name,
                        "securable_type": "TABLE",
                        "permission": "MODIFY",
                    },
                }
                for table in otel_tables
            ]
        )
        return self._apps_client.reconcile_resources(
            name,
            desired,
            owned_names=(_TRACE_EXPERIMENT_RESOURCE,),
            owned_prefixes=(_TRACE_RESOURCE_PREFIX,),
        )
