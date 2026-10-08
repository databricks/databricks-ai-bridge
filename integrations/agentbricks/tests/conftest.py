"""Fixtures shared by every matrix under ``matrices/``.

``run_context`` (options, target workspace, run identity) and ``provisioning`` (lazy workspace resources and
their teardown) are plugins. One command runs a matrix, from ``integrations/agentbricks``:

    uv run pytest tests/matrices/tools -n 6 --dist loadgroup \
        --databricks-profile <profile>

``--workspace fake`` runs against no workspace; whatever needs one is skipped.

The CLI wrapper is function-scoped: each test authors, runs and deploys its own projects, and the
fixture deletes every App and store they created when the test ends.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from agentbricks_cli import AgentbricksCli
from common import RunConfig, log
from target_workspace import TargetWorkspace
from workspace_client import Workspace, cleanup_app

pytest_plugins = ["run_context", "provisioning"]

_failed_tests: set[str] = set()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    # Includes subtest reports, so the App's logs are captured when any subtest fails.
    if report.failed:
        _failed_tests.add(report.nodeid)


@pytest.fixture(scope="session")
def workspace_client(target_workspace: TargetWorkspace) -> Workspace:
    workspace = target_workspace.client
    if workspace is None:
        pytest.skip(f"--workspace {target_workspace.name} has no Databricks SDK access")
    return workspace


@pytest.fixture
def agentbricks_cli(
    request: pytest.FixtureRequest, target_workspace: TargetWorkspace, run_config: RunConfig
) -> Iterator[AgentbricksCli]:
    cli = AgentbricksCli(target_workspace, run_config)
    yield cli
    if cli.projects and target_workspace.client is not None:
        _delete_projects(cli, target_workspace.client, failed=request.node.nodeid in _failed_tests)


def _delete_projects(cli: AgentbricksCli, workspace: Workspace, *, failed: bool) -> None:
    """Delete every App, store and role the test's projects created; the sweep gets the rest."""
    for project in cli.projects:
        if failed and project.app_registered:
            # The App is deleted next, so a failing test must capture its logs while it exists.
            workspace.app_logs(
                project.app_name, cli.logs_dir / f"deploy-runtime-{project.app_name}.log"
            )
        try:
            failures = cleanup_app(
                workspace,
                project.app_name,
                delete_store=cli.delete_store,
                memory_store=project.memory_store_name,
                session_store=project.session_store_name,
                has_app=project.app_registered,
                runtime_store=bool(project.memory_store_name or project.session_store_name),
            )
        except Exception as exc:
            failures = [str(exc)]
        for failure in failures:
            log(f"cleanup warning | {project.app_name} | {failure}")
