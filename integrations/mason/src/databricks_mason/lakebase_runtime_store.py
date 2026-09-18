"""Resolve service-managed Runtime Store backends from the API response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from databricks_mason import models
from databricks_mason.errors import AgentCliError


@dataclass(frozen=True)
class RuntimeStoreBackend:
    """Connection coordinates returned by the Runtime Store API."""

    branch: str
    database_id: str
    username: str


def get_or_create_backend(
    client: Any, app: str, app_service_principal_id: str | None
) -> RuntimeStoreBackend:
    """Create or reuse the Runtime Store owned by a Databricks App."""
    if not app_service_principal_id:
        raise AgentCliError("Could not resolve the app's service principal for its Runtime Store.")
    try:
        store = client.create_runtime_store(
            app,
            app_service_principal_id,
            app_name=app,
            retry_transient=True,
        )
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
        store = client.get_runtime_store(app)
    return backend_from_api(app, app_service_principal_id, store)


def delete(client: Any, app: str, app_service_principal_id: str) -> None:
    """Delete the managed store while preserving the app when cleanup needs a retry."""
    try:
        store = client.get_runtime_store(app)
        validate_owner(app, app_service_principal_id, store)
        client.delete_runtime_store(app)
    except AgentCliError as exc:
        if exc.error_code == "NOT_FOUND":
            return
        raise AgentCliError(
            f"Could not delete Runtime Store '{app}': {exc.message}",
            error_code=exc.error_code,
            hint=f"The deployment was retained. Retry `mason deployments delete {app}` "
            "after resolving the Runtime Store error.",
        ) from exc


def _model(runtime_store: Any) -> models.RuntimeStore:
    if not isinstance(runtime_store, dict):
        raise AgentCliError("Runtime Store API returned an invalid resource.")
    return models.RuntimeStore(runtime_store)


def validate_owner(
    app: str, app_service_principal_id: str, runtime_store: Any
) -> models.RuntimeStore:
    """Refuse to reuse or delete a Runtime Store owned by another app identity."""
    store = _model(runtime_store)
    if store.name != f"runtime-stores/{app}":
        raise AgentCliError("Runtime Store API returned an unexpected resource name.")
    owner = store.owner
    app_owner = owner.app if owner is not None else None
    if (
        app_owner is None
        or app_owner.name != app
        or app_owner.service_principal_id != app_service_principal_id
    ):
        raise AgentCliError(f"Runtime Store '{app}' does not belong to this app identity.")
    return store


def backend_from_api(
    app: str, app_service_principal_id: str, runtime_store: Any
) -> RuntimeStoreBackend:
    """Return only the connection coordinates supplied by the Runtime Store API."""
    store = validate_owner(app, app_service_principal_id, runtime_store)
    storage_backend = store.storage_backend
    lakebase = storage_backend.lakebase if storage_backend is not None else None
    branch = lakebase.branch if lakebase is not None else None
    database_id = lakebase.database_id if lakebase is not None else None
    owner = store.owner
    app_owner = owner.app if owner is not None else None
    username = app_owner.service_principal_id if app_owner is not None else None
    if (
        not isinstance(branch, str)
        or not branch
        or not isinstance(database_id, str)
        or not database_id
        or not isinstance(username, str)
        or not username
    ):
        raise AgentCliError("Runtime Store API returned an incomplete Lakebase backend.")
    return RuntimeStoreBackend(
        branch=branch,
        database_id=database_id,
        username=username,
    )
