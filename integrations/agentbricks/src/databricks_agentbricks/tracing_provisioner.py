"""Provision and grant a deployment's MLflow tracing experiment.

Render-free collaborator wrapping the tracing get-or-create and the trace-resource grant so a service
can drive them without importing ``click`` or ``render`` (a caller wraps ``reporter.status(...)``
around each call).
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Optional

from databricks_agentbricks.app_resources import apply_trace_resources
from databricks_agentbricks.cli.tracing import (
    ResolvedTraceExperiment,
    create_experiment_idempotent,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.trace_tables import TraceTable


class TracingProvisioner:
    """Resolve a project's bound tracing experiment and grant the app access to it, for a ``profile``."""

    def __init__(self, profile: Optional[str]) -> None:
        self._profile = profile

    def get_or_create(self, source: pathlib.Path, client) -> Optional[ResolvedTraceExperiment]:
        """Get-or-create this project's bound MLflow experiment in the ``profile``'s workspace, or None
        when tracing is unbound (no ``experiment_name`` in agent.toml).

        Resolves by experiment **name**, never a stored id. ``source`` locates agent.toml. Nothing is
        written back to agent.toml. Raises if the experiment can't be created.
        """
        from databricks_agentbricks.agent_project import (
            AgentProject,  # noqa: PLC0415 - avoid import cycle
        )

        try:
            project = AgentProject.load(source)
        except AgentCliError:
            project = None
        name = project.trace_experiment_name if project is not None else None
        if not name:
            return None
        return create_experiment_idempotent(self._profile, client, name)

    def apply_resources(
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
