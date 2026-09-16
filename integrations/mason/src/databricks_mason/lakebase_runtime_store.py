"""Provision and locate the Lakebase database for a Mason Runtime Store."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

from databricks_mason.app_resources import LakebaseBackend, _databricks
from databricks_mason.errors import AgentCliError

_BRANCH = "production"
_ENDPOINT = "primary"
_DATABASE = "databricks-postgres"
_RESOURCE_NAME = "postgres-runtime-store"


def backend(app: str) -> LakebaseBackend:
    """Return the dedicated fallback backend for a Mason deployment."""
    project = _project_id(app)
    return LakebaseBackend(
        project=project,
        branch=_BRANCH,
        endpoint_id=_ENDPOINT,
        database=_DATABASE,
        schema=get_lakebase_schema(app),
        tables=(),
        resource_name=_RESOURCE_NAME,
    )


def get_or_create_backend(app: str, profile: Optional[str], *, create: bool) -> LakebaseBackend:
    """Reuse the deployment's Runtime Store project or create it when allowed."""
    selected = backend(app)
    project_path = f"projects/{selected.project}"
    existing = _databricks(
        ["postgres", "get-project", project_path], profile, capture=True, check=False
    )
    if existing.returncode == 0:
        return selected
    if not create:
        raise AgentCliError(
            f"Runtime Store '{selected.project}' does not exist.",
            hint="Bind a Session Store to reuse its Lakebase database.",
        )

    payload = {"spec": {"display_name": f"Mason Runtime Store for {app}"}}
    created = _databricks(
        ["postgres", "create-project", selected.project, "--json", json.dumps(payload)],
        profile,
        capture=True,
        check=False,
    )
    if created.returncode == 0:
        return selected

    resolved = _databricks(
        ["postgres", "get-project", project_path], profile, capture=True, check=False
    )
    if resolved.returncode == 0:
        return selected
    detail = (created.stderr or created.stdout or "").strip() or "unknown error"
    raise AgentCliError(f"Could not create Runtime Store '{selected.project}'.", hint=detail)


def _project_id(app: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", app.lower()).strip("-")
    normalized = normalized or "mason-app"
    if not normalized[0].isalpha():
        normalized = f"mason-{normalized}"
    return f"{normalized}-runtime-store"[:63].rstrip("-")


def get_lakebase_schema(app: str) -> str:
    """Return the schema owned by one deployed app's Runtime Store."""
    digest = hashlib.sha256(app.encode("utf-8")).hexdigest()[:12]
    return f"databricks_mason_runtime_{digest}"
