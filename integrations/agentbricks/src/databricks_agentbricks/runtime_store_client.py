"""Provision an Agent Bricks Runtime Store backend (legacy per-app Lakebase or service-managed).

Render-free collaborator wrapping the two runtime-store backends and the legacy Lakebase resource
grant so a service can drive them without importing ``click`` or ``render`` (a caller wraps
``reporter.status(...)`` around each call).
"""

from __future__ import annotations

from typing import Optional

import databricks_agentbricks.lakebase_runtime_store as managed_runtime_store
import databricks_agentbricks.legacy_lakebase_runtime_store as legacy_runtime_store
from databricks_agentbricks.app_resources import LakebaseBackend, apply_postgres_resources
from databricks_agentbricks.lakebase_runtime_store import RuntimeStoreBackend


class RuntimeStoreClient:
    """Get-or-create the Runtime Store backend for a deployment, for a ``profile``.

    A temporary rollout switch chooses between the legacy per-app Lakebase project and the
    service-managed database; this exposes both paths render-free so the service picks one.
    """

    def __init__(self, profile: Optional[str]) -> None:
        self._profile = profile

    def legacy_backend(self, name: str) -> LakebaseBackend:
        """Reuse or create the dedicated per-app Lakebase project used by the legacy path."""
        return legacy_runtime_store.get_or_create_backend(name, self._profile)

    def apply_legacy_resource(self, name: str, backend: LakebaseBackend) -> Optional[str]:
        """Bind the legacy backend's database as a ``postgres`` app resource.

        Returns None on success or a human-readable reason on failure.
        """
        return apply_postgres_resources(name, [backend], self._profile)

    def managed_backend(
        self, client, name: str, service_principal_id: Optional[str]
    ) -> RuntimeStoreBackend:
        """Create or reuse the service-managed Runtime Store owned by the app's service principal."""
        return managed_runtime_store.get_or_create_backend(client, name, service_principal_id)

    def delete_managed(self, client, name: str, service_principal_id: str) -> None:
        """Drop the deployment's service-managed Runtime Store and its data.

        Called before the app itself is deleted: the delete needs the app's service principal, which
        stops resolving once the app is gone.
        """
        managed_runtime_store.delete(client, name, service_principal_id)
