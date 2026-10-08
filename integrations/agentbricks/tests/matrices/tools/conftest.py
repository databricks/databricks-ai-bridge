"""Tools matrix: every test runs once per authoring path, over the tools the backend can support."""

from __future__ import annotations

import pathlib

import pytest
from common import AUTHORING_PATHS, log
from tools import TOOL_DEFINITIONS, Tool


@pytest.fixture(params=AUTHORING_PATHS)
def authoring(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session")
def scratch_schema(deploy_scratch_schema) -> str:
    return deploy_scratch_schema(pathlib.Path(__file__).parent / "bundle")


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
