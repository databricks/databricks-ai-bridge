"""Provision and grant a deployment's MLflow tracing experiment.

Render-free collaborator wrapping the tracing get-or-create and the trace-resource grant so a service
can drive them without importing ``click`` or ``render`` (a caller wraps ``reporter.status(...)``
around each call).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.app_resources import apply_trace_resources
from databricks_agentbricks.clients.trace_experiment import (
    ResolvedTraceExperiment,
    create_experiment_idempotent,
)
from databricks_agentbricks.trace_tables import TraceTable


class TracingClient:
    """Resolve a project's bound tracing experiment and grant the app access to it, for a ``profile``.

    Holds the per-command API client provider, invoked lazily on first use, so its methods take
    only the resource names.
    """

    def __init__(self, api_client_provider: ApiClientProvider, profile: Optional[str]) -> None:
        self._api_client_provider = api_client_provider
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

        Delegates to ``app_resources.apply_trace_resources``: grants the app's service principal the
        ``experiment`` resource (and MODIFY on any UC OTEL tables), or prunes them on a clean unbind
        (``experiment_id`` is None). Returns None on success, else a human-readable reason.
        """
        return apply_trace_resources(name, experiment_id, otel_tables, self._profile)
