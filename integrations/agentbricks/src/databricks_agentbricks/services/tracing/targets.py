"""Choose a workspace or local target for one trace read."""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from databricks_agentbricks.clients.local_tracing_client import LocalTracingClient
from databricks_agentbricks.clients.mlflow_trace_client import MLflowTraceClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject


@dataclass(frozen=True)
class TraceReadTarget:
    tracking_uri: str | None
    experiment_id: str | None
    local: bool = False
    warning: str | None = None
    help: str | None = None


class TraceTargetResolver:
    """Apply explicit, project-bound, then local-dev trace target precedence."""

    def __init__(self, mlflow: MLflowTraceClient, local_tracing: LocalTracingClient) -> None:
        self._mlflow = mlflow
        self._local_tracing = local_tracing

    @contextmanager
    def resolve(
        self,
        source: pathlib.Path,
        experiment_name: Optional[str],
        experiment_id: Optional[str],
        warehouse_id: Optional[str],
    ) -> Iterator[TraceReadTarget]:
        if experiment_id or experiment_name:
            self._mlflow.select_workspace()
            if experiment_id:
                experiment = self._mlflow.experiment_by_id(experiment_id)
                if experiment is None:
                    raise AgentCliError(
                        f"No MLflow experiment found with id {experiment_id!r} in this workspace.",
                        hint="Check the id, or omit it to use this project's experiment.",
                    )
                resolved_id = experiment_id
            else:
                assert experiment_name is not None
                experiment = self._mlflow.experiment_by_name(experiment_name)
                if experiment is None:
                    raise AgentCliError(
                        f"No MLflow experiment named {experiment_name!r} in this workspace.",
                        hint="Check the name, or omit it to use this project's experiment.",
                    )
                resolved_id = experiment.experiment_id
            self._mlflow.require_warehouse_for_uc_read(experiment, warehouse_id)
            yield TraceReadTarget(self._mlflow.workspace_uri, resolved_id)
            return

        try:
            project = AgentProject.load(source)
        except AgentCliError:
            project = None
        bound_name = project.trace_experiment_name if project is not None else None
        if bound_name:
            self._mlflow.select_workspace()
            try:
                experiment = self._mlflow.experiment_by_name(bound_name)
            except Exception:
                experiment = None
            if experiment is not None:
                self._mlflow.require_warehouse_for_uc_read(experiment, warehouse_id)
                yield TraceReadTarget(self._mlflow.workspace_uri, experiment.experiment_id)
                return

        db = source.resolve() / ".agentbricks" / "mlflow.db"
        if not db.exists():
            yield TraceReadTarget(None, None)
            return
        launch = self._local_tracing.start_read(db)
        if launch.server is None or launch.base_url is None:
            yield TraceReadTarget(None, None, local=True, warning=launch.warning, help=launch.help)
            return
        try:
            self._mlflow.select_local(launch.base_url)
            local = self._mlflow.experiment_by_name(source.resolve().name)
            yield TraceReadTarget(
                launch.base_url,
                local.experiment_id if local else None,
                local=True,
            )
        finally:
            self._local_tracing.stop(launch.server)
