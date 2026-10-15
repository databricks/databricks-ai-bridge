"""All pytest wiring for the E2E suites: options, hooks, and the fixtures every suite shares.

pytest calls everything in this file by name; ``harness/`` is plain code, so following its imports
answers what calls what. One command runs a suite, from ``integrations/agentbricks``:

    uv run pytest tests/suites/tools -n 6 --dist loadgroup \
        --databricks-profile <profile>

``--workspace fake`` runs against no workspace; whatever needs one is skipped.

Each test deletes what it created when it ends (the ``agentbricks_cli`` fixture here, and suite
fixtures such as the tools suite's ``scratch_schema``). After every live run, the controller also
deletes whatever a crashed test or a failed teardown left (``harness.cleanup``). A run killed
before that point is cleaned up afterwards with ``--cleanup-only --run-id <its id>``.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from harness.agentbricks_cli import AgentbricksCli, not_found
from harness.cleanup import (
    Cleanup,
    app_identity,
    delete_run_leftovers,
    record_app_identity,
)
from harness.logs import log, log_command
from harness.project import Project
from harness.run import RUN_ID_PATTERN, E2EError, RunConfig, run_output_dir
from harness.target import (
    TargetWorkspace,
    TargetWorkspaceUnavailable,
    WorkspaceKind,
    make_target_workspace,
    require_workspace,
)
from harness.workspace import Workspace

PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[2]
_RUN_KEY = pytest.StashKey[RunConfig]()
_HANDOFF = "agentbricks_run"
_failed_tests: set[str] = set()


# Options


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agentbricks", "Agent Bricks E2E suites")
    group.addoption(
        "--workspace",
        type=WorkspaceKind,
        choices=list(WorkspaceKind),
        default=WorkspaceKind.LIVE,
        help="live runs against a Databricks workspace; fake skips whatever needs one.",
    )
    group.addoption(
        "--databricks-profile",
        help="Databricks CLI profile; default is ambient credentials (DATABRICKS_CONFIG_PROFILE).",
    )
    group.addoption(
        "--catalog",
        default="main",
        help="Existing catalog for temporary test schemas (default: main).",
    )
    group.addoption(
        "--wheel",
        type=pathlib.Path,
        help="Prebuilt databricks-agentbricks wheel (default: build one from this checkout).",
    )
    group.addoption(
        "--warehouse-id",
        help="SQL warehouse for UC setup (default: first running).",
    )
    group.addoption(
        "--genie-space-id",
        help="Existing Genie space for the Genie tool; without one that tool is skipped.",
    )
    group.addoption(
        "--run-id",
        help="Names every resource of the run, so --cleanup-only can delete what a killed run left "
        "(default: random). 1-8 lowercase letters or digits.",
    )
    group.addoption(
        "--keep-resources",
        action="store_true",
        help="Skip the end-of-run cleanup of leftovers; each test's own teardown still runs.",
    )
    group.addoption(
        "--cleanup-only",
        action="store_true",
        help="Run no tests; only delete what the run given by --run-id left behind.",
    )


def pytest_configure(config: pytest.Config) -> None:
    run_id = config.getoption("run_id")
    if run_id is not None and RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise pytest.UsageError("--run-id must be 1-8 lowercase letters or digits.")
    if config.getoption("cleanup_only"):
        if run_id is None:
            raise pytest.UsageError("--cleanup-only needs the --run-id of the run to clean up.")
        if config.getoption("keep_resources"):
            raise pytest.UsageError("--cleanup-only and --keep-resources contradict each other.")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("cleanup_only"):
        config.hook.pytest_deselected(items=list(items))
        items.clear()


# The run: one id and output directory, shared by the controller and every xdist worker


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """xdist hook: runs on the controller for each worker before it starts."""
    run = _base_run_config(node.config)
    node.workerinput[_HANDOFF] = {"run_id": run.run_id, "output": str(run.output)}


def _base_run_config(config: pytest.Config) -> RunConfig:
    """The run's config without a built wheel; the same value in the controller and every worker."""
    if _RUN_KEY in config.stash:
        return config.stash[_RUN_KEY]
    handoff = getattr(config, "workerinput", {}).get(_HANDOFF)
    if handoff:
        run_id, output = handoff["run_id"], pathlib.Path(handoff["output"])
    else:
        run_id = config.getoption("run_id") or uuid.uuid4().hex[:6]
        output = run_output_dir(run_id)
    option = config.getoption
    run = RunConfig(
        run_id=run_id,
        output=output,
        warehouse_id=option("warehouse_id"),
        wheel=option("wheel"),
    )
    config.stash[_RUN_KEY] = run
    return run


def _build_target_workspace(config: pytest.Config) -> TargetWorkspace:
    return make_target_workspace(
        config.getoption("workspace"),
        config.getoption("databricks_profile"),
        _base_run_config(config),
    )


@pytest.fixture(scope="session")
def run_config(request: pytest.FixtureRequest) -> RunConfig:
    run = _base_run_config(request.config)
    run.output.mkdir(parents=True, exist_ok=True)
    log(f"Agent Bricks E2E output: {run.output} (run {run.run_id})")
    if run.wheel:
        return run
    # A directory per process: xdist workers build at the same time and must not share an output.
    dist = run.output / "dist" / os.environ.get("PYTEST_XDIST_WORKER", "main")
    return dataclasses.replace(run, wheel=_build_wheel(dist))


def _build_wheel(dist: pathlib.Path) -> pathlib.Path:
    """Build this checkout's wheel, which every generated project installs instead of PyPI's."""
    argv = ["uv", "build", "--wheel", "--out-dir", str(dist)]
    log_command(argv, PACKAGE_DIR)
    result = subprocess.run(
        argv, cwd=PACKAGE_DIR, text=True, capture_output=True, timeout=600, check=False
    )
    if result.returncode != 0:
        raise E2EError(f"uv build failed ({result.returncode}):\n{result.stderr or result.stdout}")
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        raise E2EError("uv build produced no wheel")
    return wheels[-1]


# The workspace


@pytest.fixture(scope="session")
def target_workspace(request: pytest.FixtureRequest) -> TargetWorkspace:
    try:
        return _build_target_workspace(request.config)
    except TargetWorkspaceUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def workspace_client(target_workspace: TargetWorkspace) -> Workspace:
    return require_workspace(target_workspace, "Databricks SDK access")


@pytest.fixture(scope="session")
def catalog(request: pytest.FixtureRequest, target_workspace: TargetWorkspace) -> str:
    workspace = require_workspace(target_workspace, "a UC catalog")
    name = request.config.getoption("catalog")
    workspace.require_catalog(name)
    return name


# The CLI under test, and deleting what each test created


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    # Includes subtest reports, so the App's logs are captured when any subtest fails.
    if report.failed:
        _failed_tests.add(report.nodeid)


@pytest.fixture
def agentbricks_cli(
    request: pytest.FixtureRequest, target_workspace: TargetWorkspace, run_config: RunConfig
) -> AgentbricksCli:
    return AgentbricksCli(target_workspace, run_config, request.node.name)


@pytest.fixture
def cleanup(agentbricks_cli: AgentbricksCli) -> Iterator[Cleanup]:
    """Register the delete commands for what the test creates; they run when it ends."""
    registered = Cleanup(agentbricks_cli)
    yield registered
    registered.run_all()


@pytest.fixture
def project(
    request: pytest.FixtureRequest,
    cleanup: Cleanup,
    agentbricks_cli: AgentbricksCli,
    target_workspace: TargetWorkspace,
    run_config: RunConfig,
) -> Iterator[Project]:
    """A run-scoped path and App name for the test to `agentbricks init`.

    Depends on ``cleanup`` so that its teardown runs first, while the test's App still exists.
    """
    project = Project.allocate(run_config)
    workspace = target_workspace.client
    if workspace is not None:
        # The test deletes whatever App has this name, so it must not belong to anyone else.
        existing = agentbricks_cli.run(
            f"--output json deployments get {project.app_name}", check=False
        )
        if existing.returncode == 0:
            raise E2EError(f"App {project.app_name} already exists; refusing to deploy over it.")
        if not not_found(existing):
            raise E2EError(f"Could not check App {project.app_name}: {existing.stderr}")
    yield project
    failed = request.node.nodeid in _failed_tests
    if workspace is not None:
        _record_app(agentbricks_cli, workspace, project.app_name, failed=failed)
    # Otherwise a test that forgot the pin would pass against the release on PyPI.
    if not failed and (project.path / "pyproject.toml").exists() and not project.is_pinned():
        raise E2EError(
            f"{project.path} was never pinned to the runtime under test; call pin_runtime after init."
        )


def _record_app(cli: AgentbricksCli, workspace: Workspace, name: str, *, failed: bool) -> None:
    """Keep what is lost once the test's cleanup deletes its App."""
    identity = app_identity(cli, workspace, name)
    if identity.service_principal or identity.branch:
        # Leftover cleanup deletes the App's Lakebase role from this record.
        record_app_identity(cli.run_config.output, name, identity)
    if failed and identity.service_principal:
        workspace.app_logs(name, cli.logs_dir / f"deploy-runtime-{name}.log")


# Deleting what the whole run left behind


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    config = session.config
    # Workers must not clean up: the controller outlives them and sees the last one finish.
    if hasattr(config, "workerinput"):
        return
    cleanup_only = config.getoption("cleanup_only")
    if cleanup_only:
        run = _base_run_config(config)
    else:
        run = config.stash.get(_RUN_KEY, None)
        # The output dir exists once any test has run, so its absence means nothing was created.
        if run is None or not run.output.exists():
            return
        if config.getoption("keep_resources"):
            log("Resources retained (--keep-resources); skipping leftover cleanup.")
            return
    try:
        target_workspace = _build_target_workspace(config)
        workspace = target_workspace.client
    except TargetWorkspaceUnavailable:
        workspace = None
    if workspace is None:
        if cleanup_only:
            raise pytest.UsageError("--cleanup-only needs a live workspace.")
        return
    if cleanup_only and not (run.output / "apps.jsonl").exists():
        log(f"no apps.jsonl in {run.output}; Lakebase leftovers of deleted Apps can't be found")
    cli = AgentbricksCli(target_workspace, run, "cleanup")
    failures = delete_run_leftovers(
        cli, workspace, run.run_id, run.output, config.getoption("catalog")
    )
    for failure in failures:
        log(f"cleanup warning | {failure}")
    if failures:
        if session.exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    elif cleanup_only:
        # Every test was deselected, which pytest otherwise reports as "no tests collected".
        session.exitstatus = pytest.ExitCode.OK
