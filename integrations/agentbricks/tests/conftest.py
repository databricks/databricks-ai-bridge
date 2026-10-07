"""Fixtures shared by every matrix under ``matrices/``.

The CLI wrapper is function-scoped: each test authors, runs and deploys its own projects, and the
wrapper deletes every App and store it created when the test ends.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from agentbricks_cli import AgentbricksCli
from common import Evidence, Inputs, Transcript
from workspace_client import Workspace


@pytest.fixture(scope="session")
def inputs() -> Inputs:
    return Inputs.from_env()


@pytest.fixture(scope="session")
def transcript(inputs: Inputs) -> Transcript:
    return Transcript(inputs.output / "commands.log")


@pytest.fixture(scope="session")
def evidence(inputs: Inputs) -> Evidence:
    return Evidence(inputs.output)


@pytest.fixture(scope="session")
def workspace_client(inputs: Inputs, transcript: Transcript) -> Workspace:
    workspace = Workspace(
        inputs.databricks_profile,
        transcript,
        app_auth_profile=inputs.app_auth_profile,
        warehouse_id=inputs.warehouse_id,
        preprovisioned_app_catalog_access=inputs.preprovisioned_app_catalog_access,
    )
    workspace.check_app_auth()
    return workspace


@pytest.fixture
def agentbricks_cli(
    inputs: Inputs, workspace_client: Workspace, transcript: Transcript, evidence: Evidence
) -> Iterator[AgentbricksCli]:
    cli = AgentbricksCli(inputs, workspace_client, transcript, evidence)
    try:
        yield cli
    finally:
        cli.close()
