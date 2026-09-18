"""Provision legacy or resolve service-managed Lakebase Runtime Store backends."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from databricks_mason.app_resources import LakebaseBackend, _databricks
from databricks_mason.errors import AgentCliError

_LEGACY_BRANCH = "production"
_ENDPOINT = "primary"
_LEGACY_DATABASE = "databricks-postgres"
_RESOURCE_NAME = "postgres-runtime-store"
_RESOURCE_ID = re.compile(r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?")


def legacy_backend(app: str) -> LakebaseBackend:
    """Return the dedicated per-app backend used before the managed API rollout."""
    return LakebaseBackend(
        project=_legacy_project_id(app),
        branch=_LEGACY_BRANCH,
        endpoint_id=_ENDPOINT,
        database=_LEGACY_DATABASE,
        schema=get_lakebase_schema(app),
        tables=(),
        resource_name=_RESOURCE_NAME,
    )


def get_or_create_legacy_backend(app: str, profile: Optional[str]) -> LakebaseBackend:
    """Reuse or create the dedicated per-app project used by the legacy path."""
    selected = legacy_backend(app)
    project_path = f"projects/{selected.project}"
    existing = _databricks(
        ["postgres", "get-project", project_path], profile, capture=True, check=False
    )
    if existing.returncode == 0:
        return selected

    payload = {"spec": {"display_name": f"Mason Runtime Store for {app}"}}
    created = _databricks(
        ["postgres", "create-project", selected.project, "--json", json.dumps(payload)],
        profile,
        capture=True,
        check=False,
    )
    if created.returncode == 0:
        return selected

    # A concurrent deploy can win the create race. Resolve the project before surfacing failure.
    resolved = _databricks(
        ["postgres", "get-project", project_path], profile, capture=True, check=False
    )
    if resolved.returncode == 0:
        return selected
    detail = (created.stderr or created.stdout or "").strip() or "unknown error"
    raise AgentCliError(f"Could not create Runtime Store '{selected.project}'.", hint=detail)


def _legacy_project_id(app: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", app.lower()).strip("-") or "mason-app"
    if not normalized[0].isalpha():
        normalized = f"mason-{normalized}"
    return f"{normalized}-runtime-store"[:63].rstrip("-")


def runtime_store_id(app: str, app_service_principal_id: str) -> str:
    """Return the stable store ID for one Databricks App identity."""
    normalized = re.sub(r"[^a-z0-9-]+", "-", app.lower()).strip("-") or "mason-app"
    if not normalized[0].isalpha():
        normalized = f"mason-{normalized}"
    suffix = hashlib.sha256(app_service_principal_id.encode("utf-8")).hexdigest()[:12]
    return f"{normalized[: 62 - len(suffix)].rstrip('-')}-{suffix}"


def validate_owner(
    app: str, store_id: str, app_service_principal_id: str, runtime_store: Any
) -> None:
    """Refuse to reuse or delete a store belonging to a different app identity."""
    if (
        not isinstance(runtime_store, dict)
        or runtime_store.get("name") != f"runtime-stores/{store_id}"
    ):
        raise AgentCliError("Runtime Store API returned an unexpected resource name.")
    owner = runtime_store.get("owner")
    app_owner = owner.get("app") if isinstance(owner, dict) else None
    # Legacy EStore rows only persisted the SP. Its exact match is required even when the API
    # cannot return an app name; new rows must also agree with the deployment name.
    if (
        not isinstance(app_owner, dict)
        or app_owner.get("name") not in (None, "", app)
        or app_owner.get("service_principal_id") != app_service_principal_id
    ):
        raise AgentCliError(f"Runtime Store '{store_id}' does not belong to this app identity.")


def backend_from_api(
    app: str, store_id: str, app_service_principal_id: str, runtime_store: Any
) -> LakebaseBackend:
    """Validate and convert the Runtime Store API response."""
    validate_owner(app, store_id, app_service_principal_id, runtime_store)
    storage_backend = runtime_store.get("storage_backend")
    lakebase = storage_backend.get("lakebase") if isinstance(storage_backend, dict) else None
    project = lakebase.get("project_id") if isinstance(lakebase, dict) else None
    branch_path = lakebase.get("branch") if isinstance(lakebase, dict) else None
    database = lakebase.get("database_id") if isinstance(lakebase, dict) else None
    expected_branch = f"projects/{project}/branches/"
    if (
        not isinstance(project, str)
        or not _RESOURCE_ID.fullmatch(project)
        or not isinstance(branch_path, str)
        or not branch_path.startswith(expected_branch)
        or not _RESOURCE_ID.fullmatch(branch_path[len(expected_branch) :])
        or not isinstance(database, str)
        or not _RESOURCE_ID.fullmatch(database)
    ):
        raise AgentCliError("Runtime Store API returned an incomplete Lakebase backend.")
    return LakebaseBackend(
        project=project,
        branch=branch_path[len(expected_branch) :],
        endpoint_id=_ENDPOINT,
        database=database,
        schema=get_lakebase_schema(app),
        tables=(),
        resource_name=_RESOURCE_NAME,
    )


def get_lakebase_schema(app: str) -> str:
    """Return the schema owned by one deployed app's Runtime Store."""
    digest = hashlib.sha256(app.encode("utf-8")).hexdigest()[:12]
    return f"databricks_mason_runtime_{digest}"
