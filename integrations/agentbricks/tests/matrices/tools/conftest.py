"""Tools matrix: every test runs once per authoring path, over the tools the target workspace can support.

Each test gets its own temporary UC schema (``ab_e2e_<run_id>_<random>``) holding the function and
volume its tools use, and deletes it, grants included, in its own teardown. The controller sweeps
any a crashed test left behind by the run's prefix (see ``provisioning``).
"""

from __future__ import annotations

import datetime as dt
import os
import re
import uuid
from collections.abc import Iterator

import pytest
from common import (
    AUTHORING_PATHS,
    MatrixError,
    RunConfig,
    log,
    now,
    scratch_schema_prefix,
)
from provisioning import require_workspace
from target_workspace import TargetWorkspace
from tools import TOOL_DEFINITIONS, Tool
from uc_objects import UcFunction, create_uc_function


@pytest.fixture(params=AUTHORING_PATHS)
def authoring(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def scratch_schema(
    catalog: str, target_workspace: TargetWorkspace, run_config: RunConfig
) -> Iterator[str]:
    """``catalog.schema`` of a schema private to this test, deleted with everything in it afterwards."""
    workspace = require_workspace(target_workspace, "a UC schema")
    name = f"{scratch_schema_prefix(run_config.run_id)}{uuid.uuid4().hex[:6]}"
    # RemoveAfter lets a separate sweeper delete schemas of runs that crashed before any cleanup.
    full_name = workspace.create_schema(catalog, name, remove_after=now() + dt.timedelta(hours=2))
    try:
        yield full_name
    finally:
        try:
            workspace.delete_schema(full_name)
        except Exception as exc:
            log(f"cleanup warning | schema {full_name} | {exc}")


@pytest.fixture
def uc_function(scratch_schema: str, target_workspace: TargetWorkspace) -> UcFunction:
    """The marker functions and volume in ``scratch_schema``."""
    return create_uc_function(require_workspace(target_workspace, "a UC function"), scratch_schema)


@pytest.fixture(scope="session")
def genie_space(target_workspace: TargetWorkspace) -> str:
    # A space needs a warehouse and seeded tables, so it is a long-lived one rather than per run.
    space_id = os.environ.get("AGENTBRICKS_E2E_GENIE_SPACE_ID")
    if not space_id:
        pytest.skip("no Genie space; set AGENTBRICKS_E2E_GENIE_SPACE_ID")
    if re.fullmatch(r"[0-9a-f]{32}", space_id) is None:
        raise MatrixError(f"Genie space id {space_id!r} is not 32 lowercase hex characters.")
    require_workspace(target_workspace, "a Genie space").require_genie_space(space_id)
    return space_id


@pytest.fixture
def tools(request: pytest.FixtureRequest) -> tuple[Tool, ...]:
    """The tools whose fixtures are all available; the rest are logged and left out."""
    available: list[Tool] = []
    for definition in TOOL_DEFINITIONS:
        try:
            resources = {name: request.getfixturevalue(name) for name in definition.requires}
        except pytest.skip.Exception as skipped:
            log(f"tool {definition.name} skipped | {skipped}")
            continue
        available.append(definition.build(**resources))
    if not available:
        pytest.skip("no tool is available")
    return tuple(available)
