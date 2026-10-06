"""Deploy-time access for tools explicitly declared in ``agent.toml``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.projects.agent_project import ToolSpec
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.tool_access import (
    ToolAccessPlan,
    finalize_tool_access,
    plan_tool_access,
    reconcile_tool_access,
)


class ToolAccessProvisioner:
    """Reconcile direct tool grants before rollout, then prune stale App resources after it."""

    def __init__(self, apps_client: AppsClient, profile: Optional[str], reporter: Reporter) -> None:
        self._apps_client = apps_client
        self._profile = profile
        self._reporter = reporter

    def plan(self, tools: Sequence[ToolSpec]) -> ToolAccessPlan:
        """Describe the direct access required by this project's tools."""
        return plan_tool_access(tools)

    def reconcile_before_rollout(
        self, app_name: str, plan: ToolAccessPlan, workspace_client: Any
    ) -> None:
        """Apply required grants without revoking access used by the running version."""
        principal = self._apps_client.get_service_principal(app_name)
        with self._reporter.status("Granting the app access to its explicit tool resources…"):
            reconcile_tool_access(
                workspace_client,
                app_name,
                principal,
                plan,
                self._profile,
                apps_client=self._apps_client,
            )

    def finalize_after_rollout(self, app_name: str, plan: ToolAccessPlan) -> None:
        """Remove stale App tool resources only after the new source rolls out successfully."""
        with self._reporter.status("Finalizing the app's explicit tool resources…"):
            finalize_tool_access(app_name, plan, self._profile, apps_client=self._apps_client)
