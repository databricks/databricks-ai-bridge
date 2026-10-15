"""Deleting what the E2E suites create: per App when a test ends, and per run afterwards.

Two entry points, both safe to repeat (a missing resource counts as deleted):

- ``delete_app_resources``: one App and its stores, deleted the way a user would, through the CLI.
  The ``project`` fixture calls it when its test ends.
- ``delete_run_leftovers``: everything else, including whatever the CLI leaves behind (synced
  source folders, Runtime Stores whose App is gone, Lakebase databases and roles) and whatever a
  crashed test or a failed teardown left. It runs once after every live run, and on its own with
  ``pytest tests/suites --cleanup-only --run-id <id>`` after a run that was killed before it could.
  It finds resources by the run's name prefix (Apps, source folders, schemas) and by the App
  identities per-test cleanup recorded in ``<output>/apps.jsonl`` (Runtime Stores, Lakebase
  databases and roles, which outlive their App), so it touches nothing outside the run.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from collections.abc import Callable, Collection
from typing import Any

from databricks.sdk.errors import NotFound

from harness.agentbricks_cli import AgentbricksCli
from harness.logs import log
from harness.run import APP_PREFIX, E2EError, project_prefix, scratch_schema_prefix
from harness.workspace import AppIdentity, Workspace

_APPS_RECORD = "apps.jsonl"
_MISSING = re.compile(r"NOT_FOUND|RESOURCE_DOES_NOT_EXIST|does not exist|not found", re.IGNORECASE)


# Identities recorded by per-test cleanup


def record_app_identity(output: pathlib.Path, app: str, identity: AppIdentity) -> None:
    """Append one line for leftover cleanup, which runs in another process and cannot see worker memory."""
    line = json.dumps({"app": app, **dataclasses.asdict(identity)}) + "\n"
    # The leftover sweep records too, and with --cleanup-only it may be the first to write.
    output.mkdir(parents=True, exist_ok=True)
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
    cli: AgentbricksCli,
    workspace: Workspace,
    name: str,
    identity: AppIdentity,
    *,
    memory_store: str | None = None,
    session_store: str | None = None,
) -> list[str]:
    """Delete one project's stores and App through the CLI, and the catalog grant its test made.

    ``identity`` is the App as it was before any delete; record it first, because
    ``delete_run_leftovers`` needs it for what the CLI leaves behind. Every step runs regardless of
    earlier failures. Returns a description of each failure; an empty list means all succeeded.
    """
    failures: list[str] = []

    def attempt(label: str, action: Callable[..., object], *args: Any) -> None:
        try:
            action(*args)
        except Exception as exc:
            failures.append(f"{label}: {exc}")

    def run(command: str) -> None:
        result = cli.run(command, timeout=1200, check=False)
        output = (result.stderr or result.stdout).strip()
        # A store the test declared but never deployed was never created.
        if result.returncode != 0 and not _MISSING.search(output):
            raise E2EError(output)

    if memory_store:
        attempt(f"memory store {memory_store}", run, f"memory stores delete {memory_store} --yes")
    if session_store:
        attempt(
            f"session store {session_store}", run, f"sessions stores delete {session_store} --yes"
        )
    # Before the App goes: the grant is revoked from the App's principal.
    attempt(f"catalog grant {name}", workspace.revoke_catalog_grant, name)
    if identity.service_principal:
        attempt(f"deployment {name}", run, f"deployments delete {name} --yes")
    return failures


# One run


def delete_run_leftovers(
    cli: AgentbricksCli, workspace: Workspace, run_id: str, output: pathlib.Path, catalog: str
) -> list[str]:
    """Delete what a whole run left behind; returns a description of each failure."""
    failures: list[str] = []
    steps = (
        # First: the rest outlives the App, and some of it can't be deleted while the App exists.
        ("leaked Apps", lambda: _delete_leaked_apps(cli, workspace, run_id, output)),
        ("orphaned Runtime Stores", lambda: _delete_orphaned_runtime_stores(workspace, output)),
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


def _delete_leaked_apps(
    cli: AgentbricksCli, workspace: Workspace, run_id: str, output: pathlib.Path
) -> list[str]:
    """Delete Apps whose per-test cleanup failed or never ran, then wait until every run App is gone.

    `deployments delete` returns while the App is still deleting, so a teardown's App can still be
    listed here; deleting it again only warns, and the wait is what decides success.
    """
    prefix = _app_prefix(run_id)
    for deployment in json.loads(cli.run("--output json deployments list").stdout or "[]"):
        name = str(deployment.get("name") or "")
        if not name.startswith(prefix):
            continue
        log(f"deleting leaked App {name}")
        # The principal stops resolving once the App is gone, and the later steps need it.
        record_app_identity(output, name, workspace.app_identity(name))
        result = cli.run(f"deployments delete {name} --yes", timeout=1200, check=False)
        if result.returncode != 0:
            log(f"cleanup warning | {name} | {(result.stderr or result.stdout).strip()}")
    failures: list[str] = []
    for app in workspace.client.apps.list():
        if app.name and app.name.startswith(prefix):
            try:
                workspace.wait_until_app_gone(app.name)
            except Exception as exc:
                failures.append(f"App {app.name}: {exc}")
    return failures


def _delete_orphaned_runtime_stores(workspace: Workspace, output: pathlib.Path) -> list[str]:
    """Delete the Runtime Stores of this run's Apps that outlived them.

    `deployments delete` refuses once the App is gone, because it checks the store's owner against
    the App's principal.
    """
    failures: list[str] = []
    for name in recorded_app_identities(output):
        try:
            workspace.client.api_client.do("DELETE", f"/api/2.0/agents/runtime-stores/{name}")
        except NotFound:
            pass
        except Exception as exc:
            failures.append(f"runtime store {name}: {exc}")
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
