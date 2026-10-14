"""STUB: an `init` agent (default bindings); deploy created the session store agent.toml declares.

Real case still to write: conversation state persists across invocations of one session_id and is
isolated between sessions.
"""

from __future__ import annotations

import json

import pytest
from harness.agent_endpoint import AgentEndpoint
from harness.cli import AgentbricksCli
from harness.project import Project, pin_runtime
from harness.workspace import Workspace


def test_session_store_created(
    agentbricks_cli: AgentbricksCli, project: Project, workspace_client: Workspace
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    pin_runtime(project, agentbricks_cli.run_config)
    declared = project.declared_store("session")
    assert declared, "agentbricks init declared no session store"
    agentbricks_cli.run(f"deploy {project.app_name} --source {project.path}", timeout=2400)
    AgentEndpoint.deployed(workspace_client, project.app_name)

    store = json.loads(agentbricks_cli.run(f"--output json sessions stores get {declared}").stdout)
    assert declared in json.dumps(store), store


@pytest.mark.xfail(reason="STUB: no session-persistence case written yet", run=False)
def test_session_persists_across_invocations(agentbricks_cli: AgentbricksCli) -> None:
    raise NotImplementedError
