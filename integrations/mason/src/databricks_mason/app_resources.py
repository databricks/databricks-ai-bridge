"""Databricks Apps resource plumbing for a deployed app."""

from __future__ import annotations

import json
from typing import Optional

from databricks_mason.databricks_cli import _databricks


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


_TRACE_EXPERIMENT_RESOURCE = "mason-trace-experiment"


def apply_experiment_resource(
    app: str, experiment_id: str, profile: Optional[str]
) -> Optional[str]:
    """Bind the trace experiment as an `experiment` app resource so the SP can write traces.

    This is the platform-managed grant: declaring the experiment as a `CAN_EDIT` resource lets the
    app's service principal log traces to it (no manual SQL grant). `apps update` replaces the whole
    resource array, so preserve every resource we don't own. Returns None on success or a
    human-readable reason on failure.
    """
    ours = {
        "name": _TRACE_EXPERIMENT_RESOURCE,
        "experiment": {"experiment_id": experiment_id, "permission": "CAN_EDIT"},
    }
    preserved = [
        resource
        for resource in _current_app_resources(app, profile)
        if isinstance(resource, dict) and resource.get("name") != _TRACE_EXPERIMENT_RESOURCE
    ]
    payload = {"resources": preserved + [ours]}
    result = _databricks(
        ["apps", "update", app, "--json", json.dumps(payload)], profile, capture=True, check=False
    )
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "").strip() or "unknown error"
