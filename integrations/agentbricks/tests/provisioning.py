"""pytest plugin: the workspace requirements shared by every matrix, and the end-of-run cleanup.

``catalog`` is the one fixture here; a matrix that needs more (the tools matrix's UC schema) builds
it from ``require_workspace`` in its own conftest, and tears it down in the test's own teardown.

Cleanup of leftovers happens once, in ``pytest_sessionfinish`` on the controller (the only process
that sees every worker finish): it sweeps what a crashed test or a failed per-test cleanup leaked,
found by the run's name prefix (Apps, source folders, schemas) and by the App identities the workers
recorded in ``apps.jsonl`` (Lakebase databases and roles, which outlive their App).
"""

from __future__ import annotations

import pytest
from common import APP_PREFIX, RunConfig, log, project_prefix, scratch_schema_prefix
from run_context import build_target_workspace, started_run_config
from target_workspace import TargetWorkspace, TargetWorkspaceUnavailable
from workspace_client import (
    Workspace,
    cleanup_app,
    delete_lakebase_leftovers,
    recorded_app_identities,
)


def require_workspace(target_workspace: TargetWorkspace, what: str) -> Workspace:
    workspace = target_workspace.client
    if workspace is None:
        pytest.skip(f"{what} needs a live workspace; got --workspace {target_workspace.name}")
    return workspace


@pytest.fixture(scope="session")
def catalog(request: pytest.FixtureRequest, target_workspace: TargetWorkspace) -> str:
    workspace = require_workspace(target_workspace, "a UC catalog")
    name = request.config.getoption("catalog")
    workspace.require_catalog(name)
    return name


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    config = session.config
    # Workers must not clean up: the controller outlives them and sees the last one finish.
    if hasattr(config, "workerinput"):
        return
    run = started_run_config(config)
    # The output dir exists once any test has run, so its absence means nothing was provisioned.
    if run is None or not run.output.exists():
        return
    if config.getoption("keep_resources"):
        log("Resources retained (--keep-resources); skipping cleanup.")
        return
    try:
        target_workspace = build_target_workspace(config)
    except TargetWorkspaceUnavailable:
        return
    workspace = target_workspace.client
    if workspace is None:
        return
    failures: list[str] = []
    sweeps = (
        ("leaked Apps", lambda: _sweep_apps(workspace, run)),
        ("leaked Lakebase databases and roles", lambda: _sweep_lakebase(workspace, run)),
        ("leaked source folders", lambda: _sweep_deployment_sources(workspace, run.run_id)),
        (
            "leaked schemas",
            lambda: _sweep_schemas(workspace, config.getoption("catalog"), run.run_id),
        ),
    )
    for label, action in sweeps:
        try:
            failures.extend(action())
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    for failure in failures:
        log(f"cleanup warning | {failure}")
    if failures and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _app_prefix(run_id: str) -> str:
    return f"{APP_PREFIX}{project_prefix(run_id)}"


def _sweep_apps(workspace: Workspace, run: RunConfig) -> list[str]:
    """Delete Apps whose per-test cleanup failed or never ran, found by the run's name prefix."""
    prefix = _app_prefix(run.run_id)
    recorded = recorded_app_identities(run.output)
    failures: list[str] = []
    for app in list(workspace.client.apps.list()):
        if app.name and app.name.startswith(prefix):
            log(f"sweeping leaked App {app.name}")
            failures += cleanup_app(
                workspace, app.name, runtime_store=True, identity=recorded.get(app.name)
            )
    return failures


def _sweep_lakebase(workspace: Workspace, run: RunConfig) -> list[str]:
    """Delete runtime databases and roles of this run's Apps, whether or not the App still exists."""
    recorded = list(recorded_app_identities(run.output).values())
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


def _sweep_deployment_sources(workspace: Workspace, run_id: str) -> list[str]:
    """Delete this run's synced source folders, which `apps delete` leaves behind."""
    failures: list[str] = []
    for path in workspace.deployment_sources_with_prefix(_app_prefix(run_id)):
        log(f"sweeping deployment source {path}")
        try:
            workspace.delete_workspace_path(path)
        except Exception as exc:
            failures.append(f"deployment source {path}: {exc}")
    return failures


def _sweep_schemas(workspace: Workspace, catalog: str, run_id: str) -> list[str]:
    """Delete this run's temporary schemas whose test teardown did not run, found by name prefix."""
    failures: list[str] = []
    for schema in workspace.schemas_with_prefix(catalog, scratch_schema_prefix(run_id)):
        log(f"sweeping leaked schema {schema}")
        try:
            workspace.delete_schema(schema)
        except Exception as exc:
            failures.append(f"schema {schema}: {exc}")
    return failures
