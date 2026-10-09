"""MLflow lookups and reads for the tracing command."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Optional

from databricks_agentbricks.clients.tracing_client import _mlflow, _set_tracking_uri, _workspace_uri
from databricks_agentbricks.errors import AgentCliError

_UC_TRACE_TAG = "mlflow.experiment.databricksTraceDestinationPath"
_SQL_WAREHOUSE_ENV = "MLFLOW_TRACING_SQL_WAREHOUSE_ID"


def _get_experiment_by_id(mlflow: Any, experiment_id: str) -> Any | None:
    from mlflow.exceptions import MlflowException

    try:
        return mlflow.get_experiment(experiment_id)
    except MlflowException as exc:
        if getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise


class MLflowTraceClient:
    """Keep MLflow's profile, tracking URI, and warehouse effects behind one adapter."""

    def __init__(
        self,
        profile: Optional[str],
        *,
        mlflow_loader: Callable[[], Any] = _mlflow,
        tracking_uri_setter: Callable[[Any, Optional[str]], None] = _set_tracking_uri,
    ) -> None:
        self._profile = profile
        self._mlflow_loader = mlflow_loader
        self._tracking_uri_setter = tracking_uri_setter
        self._module: Any | None = None

    @property
    def module(self) -> Any:
        if self._module is None:
            self._module = self._mlflow_loader()
        return self._module

    @property
    def workspace_uri(self) -> str:
        return _workspace_uri(self._profile)

    def select_workspace(self) -> None:
        self._tracking_uri_setter(self.module, self._profile)

    def select_local(self, base_url: str) -> None:
        self.module.set_tracking_uri(base_url)

    def experiment_by_id(self, experiment_id: str) -> Any | None:
        return _get_experiment_by_id(self.module, experiment_id)

    def experiment_by_name(self, experiment_name: str) -> Any | None:
        return self.module.get_experiment_by_name(experiment_name)

    def search_traces(self, experiment_id: str, limit: int) -> list[Any]:
        return self.module.search_traces(
            locations=[experiment_id], max_results=limit, return_type="list"
        )

    def get_trace(self, trace_id: str) -> Any | None:
        return self.module.get_trace(trace_id)

    def resolve_warehouse_id(self, warehouse_id: Optional[str]) -> Optional[str]:
        return warehouse_id or os.environ.get(_SQL_WAREHOUSE_ENV)

    def require_warehouse_for_uc_read(self, experiment: Any, warehouse_id: Optional[str]) -> None:
        if _UC_TRACE_TAG not in (getattr(experiment, "tags", None) or {}):
            return
        if warehouse_id is None:
            raise AgentCliError(
                "Reading traces from a UC-backed experiment requires a SQL warehouse.",
                hint=f"Pass --warehouse <id> (or set {_SQL_WAREHOUSE_ENV}).",
            )
        os.environ[_SQL_WAREHOUSE_ENV] = warehouse_id

    def select_fallback_workspace(self, warehouse_id: Optional[str]) -> None:
        self.module.set_tracking_uri(self.workspace_uri)
        if warehouse_id:
            os.environ[_SQL_WAREHOUSE_ENV] = warehouse_id
