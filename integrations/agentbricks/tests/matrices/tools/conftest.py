"""Tools matrix: every test runs once per authoring path, over the tools the target workspace can support."""

from __future__ import annotations

import os
import pathlib
import re

import pytest
from common import AUTHORING_PATHS, MatrixError, log
from provisioning import require_workspace
from target_workspace import TargetWorkspace
from tools import TOOL_DEFINITIONS, Tool


@pytest.fixture(params=AUTHORING_PATHS)
def authoring(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session")
def scratch_schema(deploy_scratch_schema) -> str:
    return deploy_scratch_schema(pathlib.Path(__file__).parent / "bundle")


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


@pytest.fixture(scope="session")
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
