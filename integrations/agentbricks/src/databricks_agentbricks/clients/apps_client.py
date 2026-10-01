"""A small client over `databricks apps`, used by `deploy.py` and `endpoint.py`.

The `databricks apps get <name> -o json` call is the one read the CLI needs from the Apps
control plane, but three different pieces of information come out of it (service principal,
URL, compute state). Collapsing them behind a single `_get_json` accessor means the JSON is
fetched and parsed once per call site instead of being duplicated for each field. The
`databricks` runner is injected through the constructor (rather than imported as a module
global) so tests can substitute a fake runner instead of monkeypatching module-level
functions.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Callable, Optional

from databricks_agentbricks.clients.databricks_cli import _databricks
from databricks_agentbricks.errors import AgentCliError

DatabricksRunner = Callable[..., subprocess.CompletedProcess]


def _instance_args(instance_count: Optional[int]) -> list[str]:
    """Format an explicit fixed instance count for the Databricks Apps CLI.

    ``None`` leaves an existing App's scale untouched.
    """
    # TODO(vNext): consider defaulting omitted --instances to 1 in a versioned breaking release.
    if instance_count is None:
        return []
    return [
        "--compute-min-instances",
        str(instance_count),
        "--compute-max-instances",
        str(instance_count),
    ]


def _field(payload: dict, name: str):
    """Read an Apps JSON field across snake_case and camelCase CLI versions."""
    if name in payload:
        return payload[name]
    parts = name.split("_")
    return payload.get(parts[0] + "".join(part.title() for part in parts[1:]))


class AppsClient:
    """Reads and lifecycle calls over `databricks apps`, with the reads behind `apps get -o json`.

    Terminal-free: every method returns raw data (or nothing) and raises on failure, so a service
    can drive the Apps control plane without owning any presentation.

    The `runner` (the `databricks` CLI wrapper) is injected so callers - and tests - can
    supply a fake instead of monkeypatching a module-level function.
    """

    def __init__(self, profile: Optional[str], *, runner: DatabricksRunner = _databricks) -> None:
        self._profile = profile
        self._run = runner
        self._sp_cache: dict[str, Optional[str]] = {}

    def _get_json(self, name: str) -> Optional[dict]:
        result = self._run(
            ["apps", "get", name, "-o", "json"], self._profile, capture=True, check=False
        )
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def exists(self, name: str) -> bool:
        """Whether the workspace has an app by this name, probed with a non-raising `apps get`.

        A failed read is reported as "absent": the probe can't distinguish a missing app from a
        workspace it can't reach, and the callers treat both as "nothing to reuse".
        """
        return (
            self._run(["apps", "get", name], self._profile, capture=True, check=False).returncode
            == 0
        )

    def list_all(self) -> list[dict]:
        """Every app in the workspace, as the raw `apps list` payload's items.

        Unfiltered on purpose: which apps count as agent deployments is the caller's policy.
        """
        result = self._run(
            ["apps", "list", "-o", "json"],
            self._profile,
            capture=True,
            action="Could not list agent deployments.",
        )
        data = json.loads(result.stdout or "[]")
        items = data.get("apps", []) if isinstance(data, dict) else data
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise AgentCliError("Could not parse the Apps list response.")
        return items

    def get(self, name: str) -> dict:
        """The app's full `apps get` payload, raising when it can't be read.

        The read-or-None `_get_json` above backs the single-field accessors, which treat an
        unreadable app as "unknown"; a caller that shows the app to the user needs the failure.
        """
        result = self._run(
            ["apps", "get", name, "-o", "json"],
            self._profile,
            capture=True,
            action=f"Could not read deployment '{name}'.",
        )
        data = json.loads(result.stdout or "{}")
        if not isinstance(data, dict):
            raise AgentCliError(f"Could not parse deployment '{name}'.")
        return data

    def stream_logs(self, name: str) -> None:
        """Stream the app's logs straight to the terminal (uncaptured, so it tails live)."""
        self._run(
            ["apps", "logs", name], self._profile, action=f"Could not read logs for '{name}'."
        )

    def start(self, name: str) -> None:
        """Start the app's compute, raising if the command fails."""
        self._run(
            ["apps", "start", name], self._profile, action=f"Could not start deployment '{name}'."
        )

    def stop(self, name: str) -> None:
        """Stop the app's compute, raising if the command fails. Its deployed source is kept."""
        self._run(
            ["apps", "stop", name], self._profile, action=f"Could not stop deployment '{name}'."
        )

    def delete(self, name: str) -> None:
        """Delete the app, raising if the command fails. Unconfirmed here - the caller owns that."""
        self._run(
            ["apps", "delete", name], self._profile, action=f"Could not delete deployment '{name}'."
        )

    def create(self, name: str, instance_count: Optional[int]) -> str:
        """Create the app at an optional fixed scale; return the raw command output."""
        result = self._run(
            ["apps", "create", name, *_instance_args(instance_count)],
            self._profile,
            capture=True,
            action=f"Could not create deployment '{name}'.",
        )
        return result.stdout or ""

    def create_update_instances(self, name: str, instances: int) -> str:
        """Pin the app's compute to a fixed instance count; returns the raw command output."""
        update = {
            "app": {
                "compute_min_instances": instances,
                "compute_max_instances": instances,
            },
            "update_mask": "compute_min_instances,compute_max_instances",
        }
        result = self._run(
            ["apps", "create-update", name, "--json", json.dumps(update)],
            self._profile,
            capture=True,
            action=f"Could not update deployment '{name}'.",
        )
        return result.stdout or ""

    def sync_source(self, name: str, source_dir, ws_path: str) -> None:
        """Upload the local agent source to the app's workspace path."""
        # Don't ship uv.lock: it pins exact package URLs from whatever index the developer's machine
        # resolved against (often an internal proxy). The Apps build must resolve against its own
        # configured index, so let it lock fresh in-sandbox instead of inheriting the local lock.
        self._run(
            ["sync", str(source_dir), ws_path, "--exclude", "uv.lock"],
            self._profile,
            action=f"Could not upload the agent source for '{name}'.",
        )

    def deploy(self, name: str, ws_path: str) -> None:
        """Roll out the uploaded source as the app's active deployment."""
        self._run(
            ["apps", "deploy", name, "--source-code-path", ws_path],
            self._profile,
            action=f"Could not deploy '{name}'.",
        )

    def get_service_principal(self, app_name: str) -> Optional[str]:
        """The app's service principal client id (its Postgres role identity), or None if unavailable.

        Resolved once per app name and cached: within a deploy the memory grant, the session grant, and
        the managed Runtime Store all need the same SP, and an app's SP is stable for the run - so this
        collapses their separate `apps get` calls into one, and no caller has to thread the value around.
        """
        if app_name not in self._sp_cache:
            data = self._get_json(app_name)
            self._sp_cache[app_name] = _field(data, "service_principal_client_id") if data else None
        return self._sp_cache[app_name]

    def get_app_url(self, name: str) -> Optional[str]:
        """The deployed app's browsable URL, or None if it can't be read."""
        data = self._get_json(name)
        return (_field(data, "url") or None) if data else None

    def get_compute_state(self, name: str) -> Optional[str]:
        """The app's compute state (e.g. RUNNING), or None if it can't be read."""
        data = self._get_json(name)
        status = _field(data, "compute_status") if data else None
        return _field(status, "state") if isinstance(status, dict) else None

    def wait_for_active(self, name: str, timeout_s: int = 300) -> None:
        """Block until a just-created app's compute is ACTIVE (or raise on timeout).

        `apps create` returns before compute is provisioned, but `apps deploy` requires the app to be
        ACTIVE - so a first deploy races without this wait.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.get_compute_state(name) == "ACTIVE":
                return
            time.sleep(5)
        raise AgentCliError(
            f"App '{name}' did not reach a running state within {timeout_s}s.",
            hint=f"Check `agentbricks deployments get {name}`, then re-run deploy once it's running.",
        )
