"""pytest plugin: the workspace requirements shared by every suite, and the end-of-run cleanup.

``catalog`` is the one fixture here; a suite that needs more (the tools suite's UC schema) builds
it from ``require_workspace`` in its own conftest, and tears it down in the test's own teardown.

Cleanup of leftovers happens once, in ``pytest_sessionfinish`` on the controller (the only process
that sees every worker finish), by ``sweep.sweep_run``. A killed run never gets here, so CI also
runs ``tests/sweep.py`` with the same ``--run-id`` afterwards.
"""

from __future__ import annotations

import pytest
from common import log
from run_context import build_target_workspace, started_run_config
from sweep import sweep_run
from target_workspace import TargetWorkspace, TargetWorkspaceUnavailable
from workspace_client import Workspace


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
    failures = sweep_run(workspace, run.run_id, run.output, config.getoption("catalog"))
    for failure in failures:
        log(f"cleanup warning | {failure}")
    if failures and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
