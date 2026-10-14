"""Deleting what the E2E suites create: per App when a test ends, and per run afterwards.

Two entry points, both safe to repeat (a missing resource counts as deleted):

- ``delete_app_resources``: everything one App's deploy created. The ``agentbricks_cli`` fixture
  calls it for each project when its test ends.
- ``delete_run_leftovers``: whatever a crashed test or a failed teardown left. It runs once after
  every live run, and on its own with ``pytest tests/suites --cleanup-only --run-id <id>`` after a
  run that was killed before it could. It finds resources by the run's name prefix (Apps, source
  folders, schemas) and by the App identities per-test cleanup recorded in ``<output>/apps.jsonl``
  (Lakebase databases and roles, which outlive their App), so it touches nothing outside the run.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
from collections.abc import Callable, Collection
from typing import Any

from harness.logs import log
from harness.run import APP_PREFIX, E2EError, project_prefix, scratch_schema_prefix
from harness.workspace import AppIdentity, Workspace

_APPS_RECORD = "apps.jsonl"


# Identities recorded by per-test cleanup


def record_app_identity(output: pathlib.Path, app: str, identity: AppIdentity) -> None:
    """Append one line for leftover cleanup, which runs in another process and cannot see worker memory."""
    line = json.dumps({"app": app, **dataclasses.asdict(identity)}) + "\n"
    # One write per line: appends this small do not interleave across processes on POSIX.
    with (output / _APPS_RECORD).open("a", encoding="utf-8") as record:
        record.write(line)


def recorded_app_identities(output: pathlib.Path) -> dict[str, AppIdentity]:
    """Every App identity the run's workers recorded, by App name."""
    path = output / _APPS_RECORD
    identities: dict[str, AppIdentity] = {}
    if not path.exists():
        return identities
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
            identity = AppIdentity(
                data.get("service_principal"), data.get("branch"), data.get("database_id")
            )
            identities[data["app"]] = identities.get(data["app"], AppIdentity()).merged(identity)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return identities


# One App


def _delete_each(
    kind: str, find: Callable[[], list[str]], delete: Callable[[str], None]
) -> list[str]:
    """Delete everything ``find`` returns, continuing past failures; returns one line per failure."""
    try:
        names = find()
    except Exception as exc:
        return [f"{kind} lookup: {exc}"]
    failures = []
    for name in names:
        try:
            delete(name)
        except Exception as exc:
            failures.append(f"{kind} {name}: {exc}")
    return failures


def delete_lakebase_leftovers(
    workspace: Workspace,
    branch: str,
    *,
    principals: Collection[str],
    database_ids: Collection[str] = (),
) -> list[str]:
    """Delete runtime databases and then the principals' roles on one branch.

    A role that owns a database cannot be deleted, so databases go first. Only ``runtime-``
    databases are touched, and both kinds are matched by principal UUID, never by name pattern.
    """
    return _delete_each(
        "runtime database",
        lambda: workspace.runtime_databases(branch, principals, database_ids),
        workspace.delete_database,
    ) + _delete_each(
        "Lakebase role",
        lambda: workspace.roles_of(branch, principals),
        workspace.delete_role,
    )


def delete_app_resources(
    workspace: Workspace,
    name: str,
    *,
    delete_store: Callable[[str, str], subprocess.CompletedProcess[str]] | None = None,
    memory_store: str | None = None,
    session_store: str | None = None,
    has_app: bool = True,
    runtime_store: bool = False,
    identity: AppIdentity | None = None,
) -> list[str]:
    """Delete one project's stores and, if it has an App, everything the App's deploy created.

    ``delete_store(kind, store)`` is the agentbricks CLI's store delete, which has no SDK surface.
    ``identity`` is what was recorded earlier; it fills in whatever the live lookup misses.
    Every step runs regardless of earlier failures except those that need the App gone. Returns a
    description of each failure; an empty list means everything was deleted.
    """
    failures: list[str] = []

    def attempt(label: str, action: Callable[..., object], *args: Any) -> bool:
        try:
            action(*args)
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            return False
        return True

    def delete_cli_store(kind: str, store: str) -> None:
        assert delete_store is not None
        result = delete_store(kind, store)
        if result.returncode != 0:
            raise E2EError((result.stderr or result.stdout).strip())

    # Read before any delete: the principal and branch come from the App and its Runtime Store.
    known = (identity or AppIdentity()).merged(
        workspace.app_identity(name) if has_app else AppIdentity()
    )
    for kind, store in (("memory", memory_store), ("sessions", session_store)):
        if store and delete_store:
            attempt(f"{kind} store {store}", delete_cli_store, kind, store)
    if not has_app:
        return failures

    attempt(f"catalog grant {name}", workspace.revoke_catalog_grant, name)
    if runtime_store:
        attempt(f"runtime store {name}", workspace.delete_runtime_store, name)
    app_deleted = attempt(f"app {name}", workspace.delete_app, name)
    if app_deleted:
        attempt(f"deployment source {name}", workspace.delete_deployment_source, name)
    if known.branch:
        failures += delete_lakebase_leftovers(
            workspace,
            known.branch,
            principals=[known.service_principal or ""],
            database_ids=[known.database_id or ""],
        )
    return failures


# One run


def delete_run_leftovers(
    workspace: Workspace, run_id: str, output: pathlib.Path, catalog: str
) -> list[str]:
    """Delete what a whole run left behind; returns a description of each failure."""
    failures: list[str] = []
    steps = (
        ("leaked Apps", lambda: _delete_leaked_apps(workspace, run_id, output)),
        ("leaked Lakebase databases and roles", lambda: _delete_leaked_lakebase(workspace, output)),
        ("leaked source folders", lambda: _delete_leaked_deployment_sources(workspace, run_id)),
        ("leaked schemas", lambda: _delete_leaked_schemas(workspace, catalog, run_id)),
    )
    for label, action in steps:
        try:
            failures.extend(action())
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    return failures


def _app_prefix(run_id: str) -> str:
    return f"{APP_PREFIX}{project_prefix(run_id)}"


def _delete_leaked_apps(workspace: Workspace, run_id: str, output: pathlib.Path) -> list[str]:
    """Delete Apps whose per-test cleanup failed or never ran, found by the run's name prefix."""
    prefix = _app_prefix(run_id)
    recorded = recorded_app_identities(output)
    failures: list[str] = []
    for app in list(workspace.client.apps.list()):
        if app.name and app.name.startswith(prefix):
            log(f"deleting leaked App {app.name}")
            failures += delete_app_resources(
                workspace, app.name, runtime_store=True, identity=recorded.get(app.name)
            )
    return failures


def _delete_leaked_lakebase(workspace: Workspace, output: pathlib.Path) -> list[str]:
    """Delete runtime databases and roles of this run's Apps, whether or not the App still exists."""
    recorded = list(recorded_app_identities(output).values())
    failures: list[str] = []
    for branch in sorted({item.branch for item in recorded if item.branch}):
        failures += delete_lakebase_leftovers(
            workspace,
            branch,
            # A Runtime Store create that fails partway records no branch but can still leave the
            # App's role behind, so look for branchless identities on every branch this run used.
            principals=[
                item.service_principal
                for item in recorded
                if item.branch in (branch, None) and item.service_principal
            ],
            database_ids=[
                item.database_id for item in recorded if item.branch == branch and item.database_id
            ],
        )
    return failures


def _delete_leaked_deployment_sources(workspace: Workspace, run_id: str) -> list[str]:
    """Delete this run's synced source folders, which `apps delete` leaves behind."""
    failures: list[str] = []
    for path in workspace.deployment_sources_with_prefix(_app_prefix(run_id)):
        log(f"deleting leaked deployment source {path}")
        try:
            workspace.delete_workspace_path(path)
        except Exception as exc:
            failures.append(f"deployment source {path}: {exc}")
    return failures


def _delete_leaked_schemas(workspace: Workspace, catalog: str, run_id: str) -> list[str]:
    """Delete this run's temporary schemas whose test teardown did not run, found by name prefix."""
    failures: list[str] = []
    for schema in workspace.schemas_with_prefix(catalog, scratch_schema_prefix(run_id)):
        log(f"deleting leaked schema {schema}")
        try:
            workspace.delete_schema(schema)
        except Exception as exc:
            failures.append(f"schema {schema}: {exc}")
    return failures
