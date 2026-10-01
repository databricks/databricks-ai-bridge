"""Framework-neutral MLflow experiment resolution shared by deploy and tracing commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from databricks_agentbricks.trace_tables import MLflowTraceTables, uc_trace_tables


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
    """Resolve ``name`` or create it, including its parent workspace directory when needed."""
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
