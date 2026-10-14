"""STUB: an `init` agent (default bindings); deploy created the session store agent.toml declares.

Real case still to write: conversation state persists across invocations of one session_id and is
isolated between sessions.
"""

from __future__ import annotations

import json

import pytest
from harness.agentbricks_cli import AgentbricksCli
from harness.project import Project


# Deploy needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
def test_session_store_created(agentbricks_cli: AgentbricksCli, project: Project) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    declared = project.declared_store("session")
    assert declared, "agentbricks init declared no session store"
    agentbricks_cli.run(f"deploy {project.app_name} --source {project.path}", timeout=2400)

    store = json.loads(agentbricks_cli.run(f"--output json sessions stores get {declared}").stdout)
    assert store.get("session_store_name") == declared, store


@pytest.mark.xfail(reason="STUB: no session-persistence case written yet", run=False)
def test_session_persists_across_invocations(agentbricks_cli: AgentbricksCli) -> None:
    raise NotImplementedError
