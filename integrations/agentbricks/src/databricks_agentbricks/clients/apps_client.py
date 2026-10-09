"""A small client over `databricks apps`, used by deploy, dev, and endpoint commands.

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
import os
import pathlib
import subprocess
import time
from collections.abc import Collection, Sequence
from typing import Any, Callable, Optional

from databricks_agentbricks.clients.databricks_cli import _databricks
from databricks_agentbricks.clients.legacy_runtime_store import LakebaseBackend
from databricks_agentbricks.errors import AgentCliError

DatabricksRunner = Callable[..., subprocess.CompletedProcess]

_TOOL_RESOURCE_PREFIX = "agentbricks-tool-"
_TOOL_PERMISSION_STRENGTH = {
    ("FUNCTION", "EXECUTE"): 1,
    ("TABLE", "SELECT"): 1,
    ("TABLE", "MODIFY"): 2,
    ("VOLUME", "READ_VOLUME"): 1,
    ("VOLUME", "WRITE_VOLUME"): 2,
}


class _AppResourcesReadError(RuntimeError):
    """The app's current resources could not be read safely."""


def _read_failed_reason(exc: _AppResourcesReadError) -> str:
    """Explain why a resource write was skipped after an unsafe read."""
    return (
        "skipped the resource update to avoid dropping the app's other resources "
        f"(could not read current resources: {exc})"
    )


def _contains_expected_fields(actual: Any, expected: Any) -> bool:
    """Return whether ``actual`` contains the fields and values in ``expected``.

    Apps may enrich a resource after it is attached. Tool-resource verification therefore checks
    the desired fields as a subset rather than requiring byte-for-byte equality.
    """
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
    """Compare Agent Bricks-owned resources while permitting server-added fields."""
    return len(actual) == len(expected) and all(
        _contains_expected_fields(actual_resource, expected_resource)
        for actual_resource, expected_resource in zip(actual, expected, strict=True)
    )


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

    def reconcile_resources(
        self,
        name: str,
        desired_resources: Sequence[dict],
        *,
        owned_names: Collection[str],
        owned_prefixes: Collection[str],
    ) -> Optional[str]:
        """Safely replace only explicitly owned resources on an App.

        Apps resource updates replace the complete array, so the current resources must be read
        strictly before writing. A failed, malformed, or structurally invalid read returns an error
        and never writes; a successful read preserves every resource that does not match one of the
        caller's explicit names or prefixes. The masked update changes only ``resources``.
        """
        try:
            current = self._read_resources(name)
        except _AppResourcesReadError as exc:
            return _read_failed_reason(exc)

        owned_name_set = set(owned_names)
        owned_prefixes = tuple(owned_prefixes)
        preserved = [
            resource
            for resource in current
            if resource.get("name") not in owned_name_set
            and not (
                isinstance(resource.get("name"), str)
                and any(resource["name"].startswith(prefix) for prefix in owned_prefixes)
            )
        ]
        result = self._update_resources(name, [*preserved, *desired_resources])
        if result.returncode == 0:
            return None
        return (result.stderr or result.stdout or "").strip() or "unknown error"

    def attach_postgres_backends(
        self, name: str, backends: Sequence[LakebaseBackend]
    ) -> Optional[str]:
        """Attach the supplied legacy Runtime Store backends as ``postgres`` resources.

        Only the backend resource names passed for this deployment are managed; unrelated App
        resources, including other ``postgres`` entries, remain untouched.
        """
        desired = [backend.postgres_resource() for backend in backends]
        return self.reconcile_resources(
            name,
            desired,
            owned_names=[backend.resource_name for backend in backends],
            owned_prefixes=(),
        )

    def apply_tool_resources(self, name: str, resources: Sequence[dict[str, Any]]) -> Optional[str]:
        """Replace Agent Bricks-owned tool resources after a successful rollout."""
        return _apply_tool_resources(name, resources, self._profile, runner=self._run)

    def add_tool_resources_for_rollout(
        self, name: str, resources: Sequence[dict[str, Any]]
    ) -> Optional[str]:
        """Add or upgrade tool resources before rollout without pruning or downgrading grants."""
        return _add_tool_resources_for_rollout(name, resources, self._profile, runner=self._run)

    def _read_resources(self, name: str) -> list[dict]:
        """Read an App's resources array without treating malformed data as empty."""
        result = self._run(
            ["apps", "get", name, "-o", "json"],
            self._profile,
            capture=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip() or "apps get failed"
            raise _AppResourcesReadError(detail)
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise _AppResourcesReadError(f"unparseable apps get output: {exc}") from exc
        if not isinstance(payload, dict):
            raise _AppResourcesReadError("unparseable apps get output: expected an object")
        if "resources" not in payload:
            return []
        resources = payload["resources"]
        if not isinstance(resources, list) or not all(
            isinstance(resource, dict) for resource in resources
        ):
            raise _AppResourcesReadError(
                "unparseable apps get output: resources must be an array of objects"
            )
        return resources

    def _update_resources(
        self, name: str, resources: Sequence[dict]
    ) -> subprocess.CompletedProcess:
        """Replace only an App's resources field with a masked Apps update."""
        payload = {"app": {"resources": list(resources)}, "update_mask": "resources"}
        return self._run(
            ["apps", "create-update", name, "--json", json.dumps(payload)],
            self._profile,
            capture=True,
            check=False,
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

    def run_local(
        self,
        source_dir: pathlib.Path,
        entry_point_name: str,
        *,
        prepare_environment: bool,
        app_port: Optional[int],
    ) -> None:
        """Run an app locally from a prepared manifest, streaming its output."""
        args = ["apps", "run-local"]
        if prepare_environment:
            args.append("--prepare-environment")
        if app_port is not None:
            args += ["--app-port", str(app_port)]
        # run-local resolves this relative to cwd and rejects an absolute alternate-manifest path.
        args += ["--entry-point", entry_point_name]
        self._run(
            args,
            self._profile,
            cwd=str(source_dir),
            action="Could not start the agent locally.",
            env=_get_run_local_env(),
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


def _read_app_resources_strict(
    app: str,
    profile: Optional[str],
    *,
    action: str,
    runner: DatabricksRunner,
) -> tuple[list[Any] | None, str | None]:
    """Read an App resource array without treating malformed output as empty.

    Tool-resource reconciliation intentionally keeps this lower-level shape separate from
    :meth:`AppsClient._read_resources`: the Apps API can return server-owned entries that are not
    dictionaries, and those entries must be preserved byte-for-byte when Agent Bricks updates its
    own subset.
    """
    result = runner(["apps", "get", app, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        reason = (result.stderr or result.stdout or "").strip() or "unknown error"
        return None, f"{action}: {reason}"
    try:
        payload = json.loads(result.stdout or "{}")
        resources = payload.get("resources", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None, f"{action}: invalid Apps response"
    if not isinstance(resources, list):
        return None, f"{action}: invalid resources array"
    return resources, None


def _update_app_resources(
    app: str,
    resources: Sequence[dict[str, Any]],
    profile: Optional[str],
    *,
    runner: DatabricksRunner,
) -> subprocess.CompletedProcess:
    """Replace only an App's resource field through a masked Apps update."""
    payload = {"app": {"resources": list(resources)}, "update_mask": "resources"}
    return runner(
        ["apps", "create-update", app, "--json", json.dumps(payload)],
        profile,
        capture=True,
        check=False,
    )


def _tool_resource_permission_strength(resource: dict[str, Any]) -> int | None:
    """Return the relative strength of an Apps tool resource's permission."""
    uc_resource = resource.get("uc_securable")
    if isinstance(uc_resource, dict):
        return _TOOL_PERMISSION_STRENGTH.get(
            (uc_resource.get("securable_type"), uc_resource.get("permission"))
        )
    genie_resource = resource.get("genie_space")
    if isinstance(genie_resource, dict) and genie_resource.get("permission") == "CAN_RUN":
        return 1
    return None


def _apply_tool_resources(
    app: str,
    resources: Sequence[dict[str, Any]],
    profile: Optional[str],
    *,
    runner: DatabricksRunner,
) -> Optional[str]:
    """Replace Agent Bricks-owned tool resources while preserving unrelated App resources.

    This is the post-rollout phase: stale Agent Bricks tool resources are pruned and permission
    downgrades are applied only after the source has rolled out successfully. The helper keeps the
    historical function signature so the framework-agnostic tool-access planner remains independent
    of the CLI composition layer.
    """
    current, read_error = _read_app_resources_strict(
        app,
        profile,
        action="Could not read existing App resources",
        runner=runner,
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
    update = _update_app_resources(app, reconciled, profile, runner=runner)
    if update.returncode != 0:
        return (update.stderr or update.stdout or "").strip() or "unknown error"

    persisted, verify_error = _read_app_resources_strict(
        app,
        profile,
        action="Could not verify App tool resources",
        runner=runner,
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


def apply_tool_resources(
    app: str, resources: Sequence[dict[str, Any]], profile: Optional[str]
) -> Optional[str]:
    """Replace Agent Bricks-owned tool resources while preserving unrelated App resources."""
    return _apply_tool_resources(app, resources, profile, runner=_databricks)


def _add_tool_resources_for_rollout(
    app: str,
    resources: Sequence[dict[str, Any]],
    profile: Optional[str],
    *,
    runner: DatabricksRunner,
) -> Optional[str]:
    """Add or upgrade tool resources before rollout without pruning or downgrading grants.

    This is the pre-rollout phase: existing owned resources are retained when a requested grant is
    weaker, so a failed rollout cannot leave an app with less access than it started with.
    """
    current, read_error = _read_app_resources_strict(
        app,
        profile,
        action="Could not read existing App resources",
        runner=runner,
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
    current_owned = [
        resource
        for resource in current
        if isinstance(resource, dict)
        and isinstance(resource.get("name"), str)
        and resource["name"].startswith(_TOOL_RESOURCE_PREFIX)
    ]
    owned_by_name = {resource["name"]: resource for resource in current_owned}
    for desired in resources:
        name = desired.get("name")
        existing = owned_by_name.get(name)
        if existing is None:
            owned_by_name[name] = desired
            continue

        existing_strength = _tool_resource_permission_strength(existing)
        desired_strength = _tool_resource_permission_strength(desired)
        if (
            existing_strength is not None
            and desired_strength is not None
            and desired_strength < existing_strength
        ):
            continue
        owned_by_name[name] = desired

    owned = sorted(owned_by_name.values(), key=lambda resource: str(resource.get("name", "")))
    current_owned = sorted(current_owned, key=lambda resource: str(resource.get("name", "")))
    if _owned_resources_match(current_owned, owned):
        return None

    reconciled = [*preserved, *owned]
    update = _update_app_resources(app, reconciled, profile, runner=runner)
    if update.returncode != 0:
        return (update.stderr or update.stdout or "").strip() or "unknown error"

    persisted, verify_error = _read_app_resources_strict(
        app,
        profile,
        action="Could not verify App tool resources",
        runner=runner,
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


def add_tool_resources_for_rollout(
    app: str, resources: Sequence[dict[str, Any]], profile: Optional[str]
) -> Optional[str]:
    """Add or upgrade tool resources before rollout without pruning or downgrading grants."""
    return _add_tool_resources_for_rollout(app, resources, profile, runner=_databricks)


def _get_run_local_env() -> dict[str, str]:
    """Our environment without ``VIRTUAL_ENV``.

    ``run-local --prepare-environment`` creates the project's ``.venv`` but installs the app's
    requirements with ``uv pip``, which targets ``$VIRTUAL_ENV`` first: an activated venv would get
    the app's packages (and downgrades) instead of the project's own.
    """
    return {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
