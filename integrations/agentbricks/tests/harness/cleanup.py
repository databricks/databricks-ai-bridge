"""Deleting what the E2E suites create: per test when it ends, and per run afterwards.

Each test registers the delete commands for what it creates with the ``cleanup`` fixture
(``Cleanup``), which runs them when the test ends, pass or fail. ``delete_run_leftovers`` deletes
whatever is left: everything carrying the run's name prefix (stores, Apps, synced source folders,
schemas), found through the CLI where it can list them, and the Apps' Lakebase roles, found from
the identities the ``project`` fixture records because they outlive their App. It is safe to repeat
(a missing resource counts as deleted), runs once after every live run, and runs on its own with
``pytest tests/suites --cleanup-only --run-id <id>`` after a run that was killed before it could.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable, Iterator
from typing import Any

from harness.agentbricks_cli import AgentbricksCli, not_found
from harness.logs import log
from harness.run import APP_PREFIX, E2EError, project_prefix, scratch_schema_prefix
from harness.workspace import AppIdentity, Workspace

_APPS_RECORD = "apps.jsonl"


# One test


class Cleanup:
    """What a test registers runs when it ends, newest first, pass or fail, like Go's ``t.Cleanup``.

    Register right after the step that creates something, or before it when a failed step can
    still leave it behind. Something already gone counts as deleted; any other failure only warns,
    because ``delete_run_leftovers`` retries everything the run left.
    """

    def __init__(self, cli: AgentbricksCli):
        self._cli = cli
        self._steps: list[tuple[str, Callable[[], None]]] = []

    def __call__(self, command: str) -> None:
        """Run ``agentbricks <command>`` when the test ends."""
        self._steps.append((command, lambda: self._run(command)))

    def run_all(self) -> None:
        while self._steps:
            label, step = self._steps.pop()
            try:
                step()
            except Exception as exc:
                log(f"cleanup warning | {label} | {exc}")

    def _run(self, command: str) -> None:
        result = self._cli.run(command, timeout=1500, check=False)
        if result.returncode != 0 and not not_found(result):
            raise E2EError((result.stderr or result.stdout).strip())


# Identities recorded by per-test cleanup


def app_identity(cli: AgentbricksCli, workspace: Workspace, name: str) -> AppIdentity:
    """The App's principal and Lakebase branch as far as they still exist; never raises."""
    principal: str | None = None
    result = cli.run(f"--output json deployments get {name}", check=False)
    message = (result.stderr or result.stdout).strip()
    if result.returncode == 0:
        try:
            principal = json.loads(result.stdout).get("service_principal_client_id")
        except (ValueError, AttributeError):
            log(f"cleanup warning | principal lookup for {name} | {message}")
    elif not not_found(result):
        log(f"cleanup warning | principal lookup for {name} | {message}")
    return AppIdentity(principal, workspace.lakebase_branch(name))


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
            identity = AppIdentity(data.get("service_principal"), data.get("branch"))
            identities[data["app"]] = identities.get(data["app"], AppIdentity()).merged(identity)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return identities


# One run


def delete_run_leftovers(
    cli: AgentbricksCli, workspace: Workspace, run_id: str, output: pathlib.Path, catalog: str
) -> list[str]:
    """Delete what a whole run left behind; returns a description of each failure."""
    failures: list[str] = []
    steps = (
        ("leaked stores", lambda: _delete_leaked_stores(cli, run_id)),
        # Before the roles: the App's role is only safe to delete once the App is gone.
        ("leaked Apps", lambda: _delete_leaked_apps(cli, workspace, run_id, output)),
        ("leaked Lakebase roles", lambda: _delete_leaked_roles(workspace, output)),
        ("leaked source folders", lambda: _delete_leaked_deployment_sources(workspace, run_id)),
        ("leaked experiments", lambda: _delete_leaked_experiments(workspace, run_id)),
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


def _delete_leaked_stores(cli: AgentbricksCli, run_id: str) -> list[str]:
    """Delete this run's memory and session stores, found by name prefix.

    `agentbricks init` names a project's stores after its folder, which carries the run's prefix.
    """
    prefix = project_prefix(run_id)
    failures: list[str] = []
    for kind, key, name_field, address_field in (
        ("memory stores", "managed_memory_stores", "display_name", "name"),
        ("sessions stores", "session_stores", "session_store_name", "session_store_name"),
    ):
        for store in _list_all(cli, kind, key):
            if not str(store.get(name_field) or "").startswith(prefix):
                continue
            address = store[address_field]
            log(f"deleting leaked store {address}")
            result = cli.run(f"{kind} delete {address} --yes", timeout=1200, check=False)
            if result.returncode != 0:
                failures.append(f"{kind} {address}: {(result.stderr or result.stdout).strip()}")
    return failures


def _list_all(cli: AgentbricksCli, kind: str, key: str) -> Iterator[dict[str, Any]]:
    token = ""
    while True:
        page = json.loads(
            cli.run(
                f"--output json {kind} list --page-size 100{token and f' --page-token {token}'}"
            ).stdout
        )
        yield from page.get(key) or []
        token = page.get("next_page_token") or ""
        if not token:
            return


def _delete_leaked_apps(
    cli: AgentbricksCli, workspace: Workspace, run_id: str, output: pathlib.Path
) -> list[str]:
    """Delete Apps whose per-test cleanup failed or never ran.

    `deployments delete` waits until the App is gone and exits non-zero if it can't confirm that.
    """
    prefix = _app_prefix(run_id)
    failures: list[str] = []
    for deployment in json.loads(cli.run("--output json deployments list").stdout or "[]"):
        name = str(deployment.get("name") or "")
        if not name.startswith(prefix):
            continue
        log(f"deleting leaked App {name}")
        # The principal stops resolving once the App is gone, and the later steps need it.
        record_app_identity(output, name, app_identity(cli, workspace, name))
        result = cli.run(
            f"deployments delete {name} --delete-source --yes", timeout=1500, check=False
        )
        if result.returncode != 0 and not not_found(result):
            failures.append(f"App {name}: {(result.stderr or result.stdout).strip()}")
    return failures


def _delete_leaked_roles(workspace: Workspace, output: pathlib.Path) -> list[str]:
    """Delete the Lakebase roles of this run's Apps, which `deployments delete` leaves behind."""
    recorded = list(recorded_app_identities(output).values())
    failures: list[str] = []
    for branch in sorted({item.branch for item in recorded if item.branch}):
        # A Runtime Store create that fails partway records no branch but can still leave the App's
        # role behind, so look for branchless identities on every branch this run used.
        principals = [
            item.service_principal
            for item in recorded
            if item.branch in (branch, None) and item.service_principal
        ]
        try:
            roles = workspace.roles_of(branch, principals)
        except Exception as exc:
            failures.append(f"Lakebase role lookup on {branch}: {exc}")
            continue
        for role in roles:
            try:
                workspace.delete_role(role)
            except Exception as exc:
                failures.append(f"Lakebase role {role}: {exc}")
    return failures


def _delete_leaked_deployment_sources(workspace: Workspace, run_id: str) -> list[str]:
    """Delete this run's synced source folders that no `deployments delete --delete-source` removed.

    That covers a deploy that failed before its App recorded the folder, and an App delete that
    wasn't confirmed.
    """
    failures: list[str] = []
    for path in workspace.deployment_sources_with_prefix(_app_prefix(run_id)):
        log(f"deleting leaked deployment source {path}")
        try:
            workspace.delete_workspace_path(path)
        except Exception as exc:
            failures.append(f"deployment source {path}: {exc}")
    return failures


def _delete_leaked_experiments(workspace: Workspace, run_id: str) -> list[str]:
    """Delete this run's tracing experiments, which deploy creates and nothing deletes.

    `agentbricks init` names a project's experiment after its folder, under this shared path.
    """
    failures: list[str] = []
    for experiment_id in workspace.experiments_with_prefix(
        f"/Shared/agentbricks_traces/{project_prefix(run_id)}"
    ):
        log(f"deleting leaked experiment {experiment_id}")
        try:
            workspace.delete_experiment(experiment_id)
        except Exception as exc:
            failures.append(f"experiment {experiment_id}: {exc}")
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
