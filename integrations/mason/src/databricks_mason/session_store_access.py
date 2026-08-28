"""Session-store Lakebase specifics for the deployed-app SP grant (see `store_access`)."""

from __future__ import annotations

from databricks_mason.store_access import LakebaseBackend


def backend(store: str) -> LakebaseBackend:
    """The Lakebase backend for a session store. The per-store database is named after the store."""
    return LakebaseBackend(
        project="databricks-internal-agent-session-store",
        branch="production",
        endpoint_id="primary",
        database=store,
        schema="public",
        tables=("sessions", "session_items"),
        resource_name="postgres",
    )
