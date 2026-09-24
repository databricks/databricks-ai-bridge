"""Lakebase/Apps resource plumbing for a deployed app.

Binds Databricks Apps resources onto an app so its service principal gets platform-managed grants:
a `postgres` resource for the legacy per-app Runtime Store, the tracing `experiment` resource, and
`uc_securable` TABLE resources granting MODIFY on a UC-backed experiment's OTEL trace tables.
The service-managed Runtime Store path grants database access through Conversation Store instead.

Managed-store (session/memory) table access is NOT granted here. The deployed app reaches those
stores over the conversation-store REST API, which grants the app's service principal read/write
server-side (see `store_provisioner.StoreProvisioner.grant_store_access`), so no direct Lakebase grant is needed.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from databricks_mason.cli.tracing import TraceTable
from databricks_mason.databricks_cli import _databricks


@dataclass(frozen=True)
class LakebaseBackend:
    """A Lakebase database and endpoint bound as a `postgres` app resource."""

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
# Prefix for the per-table trace resources (one uc_securable resource per OTEL table).
_UC_TRACE_TABLE_RESOURCE_PREFIX = "mason-trace-table-"


def apply_trace_resources(
    app: str,
    experiment_id: Optional[str],
    tables: Sequence[TraceTable],
    profile: Optional[str],
) -> Optional[str]:
    """Reconcile the app's mason-owned trace resources to the desired state in one masked update.

    The desired set is the `experiment` resource (CAN_EDIT, so the SP can write traces) plus one
    `uc_securable` TABLE resource (MODIFY) per UC OTEL table when the experiment is UC-backed - or
    EMPTY when tracing is unbound (``experiment_id`` is None), so an unbind + redeploy prunes the
    stale `mason-trace-experiment` / `mason-trace-table-*` resources instead of leaving the SP with
    grants on an experiment it no longer uses. ``tables`` are ``TraceTable`` entries (``kind`` and
    ``full_name``, e.g. ``TraceTable("spans", "cat.schema.pfx_otel_spans")``); each kind names its
    resource ``mason-trace-table-<kind>``. Writing the complete mason-owned set every deploy also
    converges a UC rebind: a new experiment's tables replace the old ones in the same write. MODIFY
    grants MODIFY+SELECT and Databricks Apps auto-grants USE CATALOG/USE SCHEMA - no
    catalog/schema resource or SQL grant needed. Preserves every resource we don't own. None on
    success, else a reason.
    """
    ours = (
        [
            {
                "name": _TRACE_EXPERIMENT_RESOURCE,
                "experiment": {"experiment_id": experiment_id, "permission": "CAN_EDIT"},
            }
        ]
        if experiment_id is not None
        else []
    ) + [
        {
            "name": f"{_UC_TRACE_TABLE_RESOURCE_PREFIX}{t.kind}",
            "uc_securable": {
                "securable_full_name": t.full_name,
                "securable_type": "TABLE",
                "permission": "MODIFY",
            },
        }
        for t in tables
    ]
    preserved = [
        r
        for r in _current_app_resources(app, profile)
        if isinstance(r, dict)
        and r.get("name") != _TRACE_EXPERIMENT_RESOURCE
        and not str(r.get("name", "")).startswith(_UC_TRACE_TABLE_RESOURCE_PREFIX)
    ]
    result = _update_app_resources(app, preserved + ours, profile)
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "unknown error"


def apply_postgres_resources(
    app: str, backends: list[LakebaseBackend], profile: Optional[str]
) -> Optional[str]:
    """Bind each backend's database as a `postgres` app resource in one update.

    Writing the resources field REPLACES the whole array, so we must send the complete set:
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
    result = _update_app_resources(app, preserved + ours, profile)
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "unknown error"


def _update_app_resources(
    app: str, resources: list[dict], profile: Optional[str]
) -> subprocess.CompletedProcess:
    """Set the app's resources array without disturbing any other app field.

    `update_mask` scopes the write to `resources` only. A bare `apps update` replaces the whole app
    spec, so fields we don't send — notably `user_api_scopes` — would reset to their defaults every
    deploy, silently breaking OBO. The masked `create-update` still replaces the resources array
    (callers pass the complete desired set), but leaves everything else untouched.
    """
    payload = {"app": {"resources": resources}, "update_mask": "resources"}
    return _databricks(
        ["apps", "create-update", app, "--json", json.dumps(payload)],
        profile,
        capture=True,
        check=False,
    )
