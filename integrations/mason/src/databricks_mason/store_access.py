"""Lakebase/Apps resource plumbing for a deployed app.

Binds Databricks Apps resources onto an app so its service principal gets the platform-managed
grant: a `postgres` `database` resource for the durable runtime's Lakebase (see
`lakebase_durability_store`) and the tracing `experiment` resource.

Managed-store (session/memory) table access is NOT granted here. The deployed app reaches those
stores over the conversation-store REST API, which grants the app's service principal read/write
server-side (see `deploy._grant_store_access`), so no direct Lakebase grant is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from databricks_mason.databricks_cli import _databricks


@dataclass(frozen=True)
class LakebaseBackend:
    """A per-store Lakebase database: its shared project/branch, its endpoint, and its schema/tables."""

    project: str
    branch: str
    endpoint_id: str
    database: str
    schema: str
    tables: tuple[str, ...]
    resource_name: str  # the app-resource name (must be unique across an app's resources)

    @property
    def branch_path(self) -> str:
        return f"projects/{self.project}/branches/{self.branch}"

    @property
    def database_path(self) -> str:
        return f"{self.branch_path}/databases/{self.database}"

    @property
    def endpoint_path(self) -> str:
        return f"{self.branch_path}/endpoints/{self.endpoint_id}"

    def postgres_resource(self) -> dict:
        """The `postgres` app-resource entry that grants the SP a Lakebase role + CONNECT."""
        return {
            "name": self.resource_name,
            "postgres": {
                "branch": self.branch_path,
                "database": self.database_path,
                "permission": "CAN_CONNECT_AND_CREATE",
            },
        }


def _current_app_resources(app: str, profile: Optional[str]) -> list[dict]:
    """Read the app's existing resources array (empty list if it can't be read)."""
    result = _databricks(["apps", "get", app, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        return []
    try:
        resources = json.loads(result.stdout or "{}").get("resources", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return resources if isinstance(resources, list) else []


# The app-resource name for the trace experiment (unique across an app's resources, like a store's).
_TRACE_EXPERIMENT_RESOURCE = "mason-trace-experiment"


def apply_experiment_resource(
    app: str, experiment_id: str, profile: Optional[str]
) -> Optional[str]:
    """Bind the trace experiment as an `experiment` app resource so the SP can write traces.

    This is the platform-managed grant: declaring the experiment as a `CAN_EDIT` resource lets the
    app's service principal log traces to it (no manual SQL grant). Uses the same read-modify-write as
    `apply_postgres_resources` — `apps update` replaces the whole resource array, so preserve every
    resource we don't own (including the store `postgres` resources) and re-apply ours by name.
    Returns None on success or a human-readable reason on failure.
    """
    ours = {
        "name": _TRACE_EXPERIMENT_RESOURCE,
        "experiment": {"experiment_id": experiment_id, "permission": "CAN_EDIT"},
    }
    preserved = [
        r
        for r in _current_app_resources(app, profile)
        if isinstance(r, dict) and r.get("name") != _TRACE_EXPERIMENT_RESOURCE
    ]
    payload = {"resources": preserved + [ours]}
    result = _databricks(
        ["apps", "update", app, "--json", json.dumps(payload)], profile, capture=True, check=False
    )
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "unknown error"


def apply_postgres_resources(
    app: str, backends: list[LakebaseBackend], profile: Optional[str]
) -> Optional[str]:
    """Bind each backend's database as a `postgres` app resource in one update.

    `apps update --json` REPLACES the whole resources array, so we must send the complete set:
    read the app's current resources, drop the ones we manage (matched by name) so re-deploys
    update rather than duplicate them, keep every other (user-owned) resource, and append ours.
    Returns None on success or a human-readable reason on failure.
    """
    ours = [b.postgres_resource() for b in backends]
    our_names = {r["name"] for r in ours}
    preserved = [
        r
        for r in _current_app_resources(app, profile)
        if isinstance(r, dict) and r.get("name") not in our_names
    ]
    payload = {"resources": preserved + ours}
    result = _databricks(
        ["apps", "update", app, "--json", json.dumps(payload)], profile, capture=True, check=False
    )
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "unknown error"
