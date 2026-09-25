"""Masked Apps resource reconciliation for a deployed app.

Binds Databricks Apps resources onto an app so its service principal gets platform-managed grants:
a `postgres` resource for the legacy per-app Runtime Store, the tracing `experiment` resource, and
`uc_securable` TABLE resources for tracing and direct tools declared in `agent.toml`. The
service-managed Runtime Store path grants database access through Conversation Store instead.

Managed-store (session/memory) table access is NOT granted here. The deployed app reaches those
stores over the conversation-store REST API, which grants the app's service principal read/write
server-side (see `deploy._grant_store_access`), so no direct Lakebase grant is needed.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from databricks_agentbricks.databricks_cli import _databricks
from databricks_agentbricks.trace_tables import TraceTable


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


class _AppResourcesReadError(RuntimeError):
    """The app's current resources couldn't be read.

    Raised (rather than returning None or []) so a failed read is unambiguous: an ``apps get`` error or
    unparseable output must NOT be mistaken for "no resources", because the resource write is a
    full-array replace and would then drop every resource the app has. Carries the underlying error so
    callers can surface it; a returned list (possibly empty) always reflects the app's real state.
    """


def _current_app_resources(app: str, profile: Optional[str]) -> list[dict]:
    """Read the app's existing resources array; raise ``_AppResourcesReadError`` if the read fails."""
    result = _databricks(["apps", "get", app, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or "apps get failed"
        raise _AppResourcesReadError(detail)
    try:
        resources = json.loads(result.stdout or "{}").get("resources", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        raise _AppResourcesReadError(f"unparseable apps get output: {exc}") from exc
    return resources if isinstance(resources, list) else []


def _read_failed_reason(exc: _AppResourcesReadError) -> str:
    """The non-fatal reason a reconcile skipped its write because current resources couldn't be read."""
    return (
        "skipped the resource update to avoid dropping the app's other resources "
        f"(could not read current resources: {exc})"
    )


# The app-resource name for the trace experiment (unique across an app's resources, like a store's).
_TRACE_EXPERIMENT_RESOURCE = "agentbricks-trace-experiment"
_TOOL_RESOURCE_PREFIX = "agentbricks-tool-"
# Prefix for the per-table trace resources (one uc_securable resource per OTEL table).
# Databricks Apps resource names must be 2-30 characters, so keep the prefix short: the longest
# resulting name, `agentbricks-trace-annotations` (29), must stay <= 30. `apps create-update` rejects
# the WHOLE resource array if any name is too long, so an over-length name silently drops every trace
# grant. (`agentbricks-trace-table-` + `annotations` = 35 chars, which is what regressed.)
_UC_TRACE_TABLE_RESOURCE_PREFIX = "agentbricks-trace-"


def _contains_expected_fields(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_expected_fields(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _contains_expected_fields(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _owned_resources_match(actual: Sequence[Any], expected: Sequence[dict[str, Any]]) -> bool:
    return len(actual) == len(expected) and all(
        _contains_expected_fields(actual_resource, expected_resource)
        for actual_resource, expected_resource in zip(actual, expected, strict=True)
    )


def _read_app_resources_strict(
    app: str, profile: Optional[str], *, action: str
) -> tuple[list[Any] | None, str | None]:
    result = _databricks(["apps", "get", app, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        reason = (result.stderr or result.stdout or "").strip() or "unknown error"
        return None, f"{action}: {reason}"
    try:
        resources = json.loads(result.stdout or "{}").get("resources", [])
    except (json.JSONDecodeError, AttributeError):
        return None, f"{action}: invalid Apps response"
    if not isinstance(resources, list):
        return None, f"{action}: invalid resources array"
    return resources, None


def apply_trace_resources(
    app: str,
    experiment_id: Optional[str],
    tables: Sequence[TraceTable],
    profile: Optional[str],
) -> Optional[str]:
    """Reconcile the app's agentbricks-owned trace resources to the desired state in one masked update.

    The desired set is the `experiment` resource (CAN_EDIT, so the SP can write traces) plus one
    `uc_securable` TABLE resource (MODIFY) per UC OTEL table when the experiment is UC-backed - or
    EMPTY when tracing is unbound (``experiment_id`` is None), so an unbind + redeploy prunes the stale
    `agentbricks-trace-experiment` / `agentbricks-trace-*` resources instead of leaving the SP
    with grants on an experiment it no longer uses. ``tables`` are ``TraceTable`` entries (``kind`` and
    ``full_name``, e.g. ``TraceTable(TraceTableKind.SPANS, "cat.schema.pfx_otel_spans")``); each kind
    names its resource ``agentbricks-trace-<kind>``. Writing the complete agentbricks-owned set
    every deploy also converges a UC rebind: a new experiment's tables replace the old ones in the same
    write. MODIFY grants MODIFY+SELECT and Databricks Apps auto-grants USE CATALOG/USE SCHEMA - no
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
            "name": f"{_UC_TRACE_TABLE_RESOURCE_PREFIX}{t.kind.value}",
            "uc_securable": {
                "securable_full_name": t.full_name,
                "securable_type": "TABLE",
                "permission": "MODIFY",
            },
        }
        for t in tables
    ]
    try:
        current = _current_app_resources(app, profile)
    except _AppResourcesReadError as exc:
        return _read_failed_reason(exc)
    preserved = [
        r
        for r in current
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
    try:
        current = _current_app_resources(app, profile)
    except _AppResourcesReadError as exc:
        return _read_failed_reason(exc)
    preserved = [r for r in current if isinstance(r, dict) and r.get("name") not in our_names]
    result = _update_app_resources(app, preserved + ours, profile)
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "unknown error"


def apply_tool_resources(
    app: str, resources: Sequence[dict[str, Any]], profile: Optional[str]
) -> Optional[str]:
    """Replace Agent Bricks-owned tool resources while preserving unrelated App resources."""
    current, read_error = _read_app_resources_strict(
        app, profile, action="Could not read existing App resources"
    )
    if read_error is not None:
        return read_error
    assert current is not None

    preserved = [
        resource
        for resource in current
        if not (
            isinstance(resource, dict)
            and isinstance(resource.get("name"), str)
            and resource["name"].startswith(_TOOL_RESOURCE_PREFIX)
        )
    ]
    owned = sorted(resources, key=lambda resource: str(resource.get("name", "")))
    current_owned = sorted(
        (
            resource
            for resource in current
            if isinstance(resource, dict)
            and isinstance(resource.get("name"), str)
            and resource["name"].startswith(_TOOL_RESOURCE_PREFIX)
        ),
        key=lambda resource: str(resource.get("name", "")),
    )
    if _owned_resources_match(current_owned, owned):
        return None
    reconciled = [*preserved, *owned]
    update = _update_app_resources(app, reconciled, profile)
    if update.returncode != 0:
        return (update.stderr or update.stdout or "").strip() or "unknown error"

    persisted, verify_error = _read_app_resources_strict(
        app, profile, action="Could not verify App tool resources"
    )
    if verify_error is not None:
        return verify_error
    assert persisted is not None
    persisted_owned = sorted(
        (
            resource
            for resource in persisted
            if isinstance(resource, dict)
            and isinstance(resource.get("name"), str)
            and resource["name"].startswith(_TOOL_RESOURCE_PREFIX)
        ),
        key=lambda resource: str(resource.get("name", "")),
    )
    if not _owned_resources_match(persisted_owned, owned):
        return "Could not verify App tool resources: Agent Bricks-owned resources do not match"
    return None


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
