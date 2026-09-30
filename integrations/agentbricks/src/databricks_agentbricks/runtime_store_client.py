"""Provision an Agent Bricks Runtime Store backend (legacy per-app Lakebase or service-managed).

Render-free collaborator wrapping the two runtime-store backends and the legacy Lakebase resource
grant so a service can drive them without importing ``click`` or ``render`` (a caller wraps
``reporter.status(...)`` around each call).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import databricks_agentbricks.lakebase_runtime_store as managed_runtime_store
import databricks_agentbricks.legacy_lakebase_runtime_store as legacy_runtime_store
from databricks_agentbricks.app_resources import LakebaseBackend, apply_postgres_resources
from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.lakebase_runtime_store import RuntimeStoreBackend


class RuntimeStoreClient:
    """Get-or-create the Runtime Store backend for a deployment, for a ``profile``.

    A temporary rollout switch chooses between the legacy per-app Lakebase project and the
    service-managed database; the switch is injected at construction (``use_managed``) so callers
    ask intent-level questions (``is_managed``) instead of reading the flag themselves. Holds its
    connections - the caching workspace-client factory, invoked lazily on first use, and the apps
    client the managed paths resolve the app's service principal through.
    """

    def __init__(
        self,
        api_client_factory: Callable[[], Any],
        profile: Optional[str],
        apps_client: AppsClient,
        use_managed: bool,
    ) -> None:
        self._api = api_client_factory
        self._profile = profile
        self._apps = apps_client
        self._use_managed = use_managed

    def is_managed(self) -> bool:
        """Whether the service-managed Runtime Store rollout switch is on."""
        return self._use_managed

    def legacy_backend(self, name: str) -> LakebaseBackend:
        """Reuse or create the dedicated per-app Lakebase project used by the legacy path."""
        return legacy_runtime_store.get_or_create_backend(name, self._profile)

    def apply_legacy_resource(self, name: str, backend: LakebaseBackend) -> Optional[str]:
        """Bind the legacy backend's database as a ``postgres`` app resource.

        Returns None on success or a human-readable reason on failure.
        """
        return apply_postgres_resources(name, [backend], self._profile)

    def managed_backend(self, name: str) -> RuntimeStoreBackend:
        """Create or reuse the service-managed Runtime Store owned by the app's service principal."""
        sp = self._apps.service_principal(name)
        return managed_runtime_store.get_or_create_backend(self._api(), name, sp)

    def delete_managed(self, name: str) -> None:
        """Drop the deployment's service-managed Runtime Store and its data.

        Called before the app itself is deleted: the delete needs the app's service principal, which
        stops resolving once the app is gone. If that identity can't be read we refuse outright
        rather than delete the app and orphan its data.
        """
        sp = self._apps.service_principal(name)
        if not sp:
            raise AgentCliError(
                "Could not resolve the app's service principal for Runtime Store cleanup.",
                hint="The deployment was retained. Check access to the app and retry deletion.",
            )
        managed_runtime_store.delete(self._api(), name, sp)
