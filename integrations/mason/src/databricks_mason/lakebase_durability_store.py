"""Resolve the Lakebase database provisioned by the Runtime Store API."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from databricks_mason.errors import AgentCliError

_PROJECT = "databricks-internal-agent-runtime-store"
_BRANCH = "production"
_ENDPOINT = "primary"


@dataclass(frozen=True)
class LakebaseBackend:
    project: str
    branch: str
    endpoint_id: str
    database: str
    schema: str

    @property
    def branch_path(self) -> str:
        return f"projects/{self.project}/branches/{self.branch}"

    @property
    def database_path(self) -> str:
        return f"{self.branch_path}/databases/{self.database}"

    @property
    def endpoint_path(self) -> str:
        return f"{self.branch_path}/endpoints/{self.endpoint_id}"


def runtime_store_id(app: str, app_service_principal_id: str) -> str:
    """Return the stable store ID for one Databricks App identity."""
    normalized = re.sub(r"[^a-z0-9-]+", "-", app.lower()).strip("-") or "mason-app"
    if not normalized[0].isalpha():
        normalized = f"mason-{normalized}"
    suffix = hashlib.sha256(app_service_principal_id.encode("utf-8")).hexdigest()[:12]
    return f"{normalized[: 62 - len(suffix)].rstrip('-')}-{suffix}"


def backend(app: str, store_id: str) -> LakebaseBackend:
    """Return the deterministic backend for an already-existing Runtime Store."""
    return LakebaseBackend(
        project=_PROJECT,
        branch=_BRANCH,
        endpoint_id=_ENDPOINT,
        database=store_id,
        schema=get_lakebase_schema(app),
    )


def backend_from_api(app: str, store_id: str, runtime_store: Any) -> LakebaseBackend:
    """Validate and convert the Runtime Store API response."""
    lakebase = runtime_store.get("lakebase_backend") if isinstance(runtime_store, dict) else None
    project = lakebase.get("project_id") if isinstance(lakebase, dict) else None
    branch_path = lakebase.get("branch") if isinstance(lakebase, dict) else None
    database = lakebase.get("database_id") if isinstance(lakebase, dict) else None
    expected_branch = f"projects/{project}/branches/"
    if (
        not project
        or not isinstance(branch_path, str)
        or not branch_path.startswith(expected_branch)
        or not database
    ):
        raise AgentCliError("Runtime Store API returned an incomplete Lakebase backend.")
    if project != _PROJECT or database != store_id:
        raise AgentCliError("Runtime Store API returned an unexpected Lakebase backend.")
    return LakebaseBackend(
        project=project,
        branch=branch_path[len(expected_branch) :],
        endpoint_id=_ENDPOINT,
        database=database,
        schema=get_lakebase_schema(app),
    )


def get_lakebase_schema(app: str) -> str:
    """Return the schema owned by one deployed app's Runtime Store."""
    digest = hashlib.sha256(app.encode("utf-8")).hexdigest()[:12]
    return f"databricks_mason_runtime_{digest}"
