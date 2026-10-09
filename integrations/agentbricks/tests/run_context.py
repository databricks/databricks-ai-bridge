"""pytest plugin: command-line options, the run's identity, and the workspace the suite runs against.

Under xdist the controller picks the run id and output directory and hands them to each worker in
``workerinput``, so every process names, finds and later tears down the same resources.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import tempfile
import uuid
from typing import Any

import pytest
from common import MatrixError, RunConfig, log, run_command
from target_workspace import (
    WORKSPACE_KINDS,
    TargetWorkspace,
    TargetWorkspaceUnavailable,
    make_target_workspace,
)

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent
_RUN_KEY = pytest.StashKey[RunConfig]()
_HANDOFF = "agentbricks_run"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agentbricks", "Agent Bricks E2E matrices")
    group.addoption(
        "--workspace",
        choices=WORKSPACE_KINDS,
        default="live",
        help="live runs against a Databricks workspace; fake skips whatever needs one.",
    )
    group.addoption(
        "--databricks-profile",
        help="Databricks CLI profile; default is ambient credentials (DATABRICKS_CONFIG_PROFILE).",
    )
    group.addoption(
        "--app-auth-profile",
        help="OAuth profile for deployed App /api calls; defaults to --databricks-profile.",
    )
    group.addoption(
        "--catalog",
        default=os.environ.get("AGENTBRICKS_E2E_CATALOG", "main"),
        help="Existing catalog for temporary test schemas (AGENTBRICKS_E2E_CATALOG, default main).",
    )
    group.addoption("--bridge-sha", help="Immutable bridge commit for generated App dependencies.")
    group.addoption(
        "--wheel",
        type=pathlib.Path,
        help="Prebuilt databricks-agentbricks wheel (default: build one from this checkout).",
    )
    group.addoption("--warehouse-id", help="SQL warehouse for UC setup (default: first running).")
    group.addoption(
        "--preprovisioned-app-catalog-access",
        action="store_true",
        help="Skip per-App USE CATALOG grants because catalog access is pre-provisioned.",
    )
    group.addoption(
        "--keep-resources",
        action="store_true",
        help="Skip the end-of-run sweep of leaked Apps and temporary schemas.",
    )


def pytest_configure(config: pytest.Config) -> None:
    sha = config.getoption("bridge_sha")
    if sha is not None and re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise pytest.UsageError("--bridge-sha must be a 40-character lowercase Git commit SHA.")


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """xdist hook: runs on the controller for each worker before it starts."""
    run = base_run_config(node.config)
    node.workerinput[_HANDOFF] = {"run_id": run.run_id, "output": str(run.output)}


def base_run_config(config: pytest.Config) -> RunConfig:
    """The run's config without a built wheel; the same value in the controller and every worker."""
    if _RUN_KEY in config.stash:
        return config.stash[_RUN_KEY]
    handoff = getattr(config, "workerinput", {}).get(_HANDOFF)
    if handoff:
        run_id, output = handoff["run_id"], pathlib.Path(handoff["output"])
    else:
        run_id = uuid.uuid4().hex[:6]
        output = pathlib.Path(tempfile.gettempdir()) / f"agentbricks-e2e-{run_id}"
    option = config.getoption
    run = RunConfig(
        run_id=run_id,
        output=output,
        app_auth_profile=option("app_auth_profile") or option("databricks_profile"),
        warehouse_id=option("warehouse_id"),
        preprovisioned_app_catalog_access=bool(option("preprovisioned_app_catalog_access")),
        bridge_sha=option("bridge_sha"),
        wheel=option("wheel"),
    )
    config.stash[_RUN_KEY] = run
    return run


def started_run_config(config: pytest.Config) -> RunConfig | None:
    """The run's config if this process ever created one; None means no test needed it."""
    return config.stash.get(_RUN_KEY, None)


def build_target_workspace(config: pytest.Config) -> TargetWorkspace:
    return make_target_workspace(
        config.getoption("workspace"),
        config.getoption("databricks_profile"),
        base_run_config(config),
    )


@pytest.fixture(scope="session")
def target_workspace(request: pytest.FixtureRequest) -> TargetWorkspace:
    try:
        return build_target_workspace(request.config)
    except TargetWorkspaceUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def run_config(request: pytest.FixtureRequest) -> RunConfig:
    run = base_run_config(request.config)
    run.output.mkdir(parents=True, exist_ok=True)
    log(f"Agent Bricks E2E output: {run.output} (run {run.run_id})")
    if run.wheel or run.bridge_sha:
        return run
    # A directory per process: xdist workers build at the same time and must not share an output.
    dist = run.output / "dist" / os.environ.get("PYTEST_XDIST_WORKER", "main")
    run_command(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=PACKAGE_DIR, timeout=600)
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        raise MatrixError("uv build produced no wheel")
    return dataclasses.replace(run, wheel=wheels[-1])
