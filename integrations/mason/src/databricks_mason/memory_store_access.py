"""Memory-store Lakebase specifics for the deployed-app SP grant (see `store_access`)."""

from __future__ import annotations

from databricks_mason.store_access import LakebaseBackend

_PROJECT = "databricks-internal-agent-memory-store"


def backend(database: str) -> LakebaseBackend:
    """The Lakebase backend for a memory store.

    Unlike the session store, the per-store database is not the store id; it's the last segment of
    the store's ``storage_backend.backend_id`` (e.g. ``memory-250cbddd``), so callers pass the
    resolved database name. Memory entries live in the ``memory.memory_entries`` table.
    """
    return LakebaseBackend(
        project=_PROJECT,
        branch="production",
        endpoint_id="primary",
        database=database,
        schema="memory",
        tables=("memory_entries",),
        resource_name="postgres-memory",
    )


def database_from_backend_id(backend_id: str) -> str:
    """Extract the per-store database name from a store's ``storage_backend.backend_id`` path."""
    return backend_id.split("/")[-1]
