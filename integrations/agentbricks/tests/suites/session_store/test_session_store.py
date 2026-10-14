"""STUB: an `init` agent (default bindings); the session store the CLI declared was created.

Real case still to write: conversation state persists across invocations of one session_id and is
isolated between sessions.
"""

from __future__ import annotations

import pytest
from harness.cli import AgentbricksCli, AgentEndpoint
from harness.workspace import Workspace


def test_session_store_created(
    agentbricks_cli: AgentbricksCli, workspace_client: Workspace
) -> None:
    project = agentbricks_cli.new_project("cli")
    agentbricks_cli.run(f"deploy {project.app_name} --source {project.path}", timeout=2400)
    AgentEndpoint.deployed(workspace_client, project.app_name)
    resource = project.session_resource or {}

    assert resource.get("session_store_name") == project.session_store_config, resource


@pytest.mark.xfail(reason="STUB: no session-persistence case written yet", run=False)
def test_session_persists_across_invocations(agentbricks_cli: AgentbricksCli) -> None:
    raise NotImplementedError
