"""Legacy per-app Lakebase project provisioning for Agent Bricks Runtime Store."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

from databricks_agentbricks.app_resources import LakebaseBackend
from databricks_agentbricks.databricks_cli import _databricks
from databricks_agentbricks.errors import AgentCliError

_BRANCH = "production"
_DATABASE = "databricks-postgres"
_ENDPOINT = "primary"
_RESOURCE_NAME = "postgres-runtime-store"
_SCHEMA_PREFIX = "databricks_agentkit_runtime_"


def backend(app: str) -> LakebaseBackend:
    """Return the dedicated per-app backend used before the managed API rollout."""
    return LakebaseBackend(
        project=_project_id(app),
        branch=_BRANCH,
        endpoint_id=_ENDPOINT,
        database=_DATABASE,
        schema=get_lakebase_schema(app),
        tables=(),
        resource_name=_RESOURCE_NAME,
    )


def get_or_create_backend(app: str, profile: Optional[str]) -> LakebaseBackend:
    """Reuse or create the dedicated per-app project used by the legacy path."""
    selected = backend(app)
    project_path = f"projects/{selected.project}"
    existing = _databricks(
        ["postgres", "get-project", project_path], profile, capture=True, check=False
    )
    if existing.returncode == 0:
        return selected

    payload = {"spec": {"display_name": f"Agent Bricks Runtime Store for {app}"}}
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


def _project_id(app: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", app.lower()).strip("-") or "agentbricks-app"
    if not normalized[0].isalpha():
        normalized = f"agentbricks-{normalized}"
    return f"{normalized}-runtime-store"[:63].rstrip("-")


def get_lakebase_schema(app: str) -> str:
    """Return the per-app Runtime Store schema name."""
    digest = hashlib.sha256(app.encode("utf-8")).hexdigest()[:12]
    return f"{_SCHEMA_PREFIX}{digest}"
