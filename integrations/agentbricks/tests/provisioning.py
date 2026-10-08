"""pytest plugin: the workspace resources tests ask for, as lazy session fixtures, and their teardown.

A fixture exists per requirement (``catalog``, ``scratch_schema``, ``uc_function``) and is only
created when a selected test needs it, so a run that never asks for ``uc_function`` never deploys a
bundle or creates a function. Each fixture skips when the target workspace
cannot provide it or the operator did not configure it.

Creation happens once per run across xdist workers (``shared_state``). Teardown happens once, in
``pytest_sessionfinish`` on the controller (the only process that sees every worker finish): it
destroys whatever the creators tracked and sweeps Apps a crashed test leaked.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest
from common import (
    APP_PREFIX,
    MatrixError,
    RunConfig,
    child_env,
    log,
    project_prefix,
    run_command,
)
from run_context import base_run_config, build_target_workspace, started_run_config
from shared_state import SharedResources, Track
from target_workspace import TargetWorkspace, TargetWorkspaceUnavailable
from workspace_client import Workspace, cleanup_app

BUNDLE_TARGET = "nightly"
SCHEMA_RESOURCE = "scratch_schema"


@dataclasses.dataclass(frozen=True)
class UcFunction:
    """The scratch UC objects the tools use."""

    function: str
    # Called by ``function`` and deliberately never declared in agent.toml.
    nested_function: str
    volume: str
    volume_file_path: str
    volume_marker: str


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


@pytest.fixture(scope="session")
def deploy_scratch_schema(
    catalog: str,
    target_workspace: TargetWorkspace,
    shared_resources: SharedResources,
    request: pytest.FixtureRequest,
) -> Callable[[Path], str]:
    """Deploy a matrix's bundle once per run and return its ``catalog.schema``.

    Every bundle names its schema ``ab_e2e_<run_id>``, so one run can deploy only one of them.
    """
    run = base_run_config(request.config)

    def deploy(bundle_dir: Path) -> str:
        value = shared_resources.get_or_create(
            "scratch_schema",
            lambda track: _deploy_bundle(target_workspace, run, catalog, bundle_dir, track),
        )
        if value["bundle_dir"] != str(bundle_dir.resolve()):
            raise MatrixError(
                f"This run already deployed {value['bundle_dir']}, whose schema has the same name "
                f"as the one in {bundle_dir}; run one bundle-backed matrix per invocation."
            )
        return value["schema"]

    return deploy


@pytest.fixture(scope="session")
def uc_function(
    scratch_schema: str, target_workspace: TargetWorkspace, shared_resources: SharedResources
) -> UcFunction:
    """Marker functions and a marker volume in ``scratch_schema`` (a fixture each matrix defines)."""
    workspace = require_workspace(target_workspace, "a UC function")
    value = shared_resources.get_or_create(
        "uc_function", lambda track: _create_uc_function(workspace, scratch_schema, track)
    )
    return UcFunction(**value)


# Creation


def _bundle_env(target_workspace: TargetWorkspace, catalog: str, run_id: str) -> dict[str, str]:
    return child_env(
        {**target_workspace.env, "BUNDLE_VAR_catalog_name": catalog, "BUNDLE_VAR_run_id": run_id}
    )


def _bundle(
    target_workspace: TargetWorkspace,
    bundle_dir: Path,
    env: dict[str, str],
    *args: str,
    timeout: float,
) -> str:
    argv = ["databricks", "bundle", *args, "-t", BUNDLE_TARGET, *target_workspace.cli_args]
    return run_command(argv, cwd=bundle_dir, env=env, timeout=timeout)


def _deploy_bundle(
    target_workspace: TargetWorkspace, run: RunConfig, catalog: str, bundle_dir: Path, track: Track
) -> dict[str, str]:
    bundle_dir = bundle_dir.resolve()
    env = _bundle_env(target_workspace, catalog, run.run_id)
    # Tracked first: a deploy that fails midway still leaves a schema to destroy.
    track(bundle_dir=str(bundle_dir), catalog=catalog, run_id=run.run_id)
    _bundle(target_workspace, bundle_dir, env, "deploy", timeout=900)
    summary = json.loads(
        _bundle(target_workspace, bundle_dir, env, "summary", "--output", "json", timeout=120)
    )
    resource = summary.get("resources", {}).get("schemas", {}).get(SCHEMA_RESOURCE)
    if not isinstance(resource, dict):
        raise MatrixError(f"bundle summary has no schemas.{SCHEMA_RESOURCE}: {summary}")
    resolved_id = str(resource.get("id") or "")
    if resolved_id.count(".") == 1:
        schema = resolved_id
    else:
        schema = f"{resource.get('catalog_name') or catalog}.{resource.get('name')}"
    if schema.count(".") != 1 or "None" in schema:
        raise MatrixError(f"Could not resolve the deployed schema from {resource}.")
    return {"schema": schema, "bundle_dir": str(bundle_dir)}


def _create_uc_function(workspace: Workspace, schema: str, track: Track) -> dict[str, str]:
    catalog, _, schema_name = schema.partition(".")
    suffix = uuid.uuid4().hex[:8]
    nested_name = f"agentbricks_nested_{suffix}"
    nested_function = f"{catalog}.{schema_name}.{nested_name}"
    track(nested_function=nested_function)
    workspace.sql(
        f"CREATE OR REPLACE FUNCTION `{catalog}`.`{schema_name}`.`{nested_name}`"
        "(value STRING) RETURNS STRING "
        "COMMENT 'Transitive Agent Bricks E2E marker; never declared in agent.toml' "
        "RETURN concat('AGENTBRICKS_UC_OK:', value)"
    )
    # Leave room for catalog and schema in the 64-character MCP tool name.
    function_name = f"ab_uc_{suffix}"
    function = f"{catalog}.{schema_name}.{function_name}"
    volume_name = f"agentbricks_volume_{suffix}"
    volume = f"{catalog}.{schema_name}.{volume_name}"
    track(volume=volume)
    workspace.sql(f"CREATE VOLUME `{catalog}`.`{schema_name}`.`{volume_name}`")
    marker = f"AGENTBRICKS_VOLUME_{uuid.uuid4().hex}"
    file_path = f"/Volumes/{catalog}/{schema_name}/{volume_name}/marker.txt"
    track(volume_file_path=file_path)
    workspace.upload(file_path, marker.encode())
    exposed_tool_name = function.replace(".", "__")
    if len(exposed_tool_name) > 64:
        raise MatrixError(
            "The UC function's MCP tool name would exceed 64 characters: "
            f"{exposed_tool_name!r}. Use a shorter catalog or run id."
        )
    track(function=function)
    workspace.sql(
        f"CREATE OR REPLACE FUNCTION `{catalog}`.`{schema_name}`.`{function_name}`"
        "(value STRING) RETURNS STRING "
        "COMMENT 'Deterministic Agent Bricks E2E marker tool' "
        f"RETURN `{catalog}`.`{schema_name}`.`{nested_name}`(value)"
    )
    return {
        "function": function,
        "nested_function": nested_function,
        "volume": volume,
        "volume_file_path": file_path,
        "volume_marker": marker,
    }


# Teardown


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    config = session.config
    # Workers must not tear down: the controller outlives them and sees the last one finish.
    if hasattr(config, "workerinput"):
        return
    run = started_run_config(config)
    if run is None:
        return
    tracked = SharedResources(run.output / "state").tracked()
    if not tracked and not (run.output / "projects").exists():
        return
    if config.getoption("keep_resources"):
        log("Resources retained (--keep-resources); skipping teardown.")
        return
    try:
        target_workspace = build_target_workspace(config)
    except TargetWorkspaceUnavailable:
        return
    if target_workspace.client is not None and _teardown(
        target_workspace, target_workspace.client, run, tracked
    ):
        if session.exitstatus == 0:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _teardown(
    target_workspace: TargetWorkspace,
    workspace: Workspace,
    run: RunConfig,
    tracked: dict[str, dict[str, str]],
) -> list[str]:
    """Remove what the run created and return a description of each failure."""
    failures: list[str] = []

    def attempt(label: str, action: Callable[[], object]) -> None:
        try:
            action()
        except Exception as exc:
            log(f"cleanup warning | {label} | {exc}")
            failures.append(f"{label}: {exc}")

    attempt("leaked Apps", lambda: failures.extend(_sweep_apps(workspace, run.run_id)))
    functions = tracked.get("uc_function", {})
    for key in ("function", "nested_function"):
        if name := functions.get(key):
            attempt(f"function {name}", partial(workspace.drop, "FUNCTION", name))
    if path := functions.get("volume_file_path"):
        attempt(f"file {path}", partial(workspace.delete_file, path))
    if volume := functions.get("volume"):
        attempt(f"volume {volume}", partial(workspace.drop, "VOLUME", volume))
    if bundle := tracked.get("scratch_schema"):
        env = _bundle_env(target_workspace, bundle["catalog"], bundle["run_id"])
        bundle_dir = Path(bundle["bundle_dir"])
        attempt(
            "bundle destroy",
            lambda: _bundle(
                target_workspace, bundle_dir, env, "destroy", "--auto-approve", timeout=900
            ),
        )
    return failures


def _sweep_apps(workspace: Workspace, run_id: str) -> list[str]:
    """Delete Apps whose per-test cleanup failed or never ran, found by the run's name prefix."""
    prefix = f"{APP_PREFIX}{project_prefix(run_id)}"
    failures: list[str] = []
    for app in list(workspace.client.apps.list()):
        if app.name and app.name.startswith(prefix):
            log(f"sweeping leaked App {app.name}")
            failures += cleanup_app(workspace, app.name, runtime_store=True)
    return failures
