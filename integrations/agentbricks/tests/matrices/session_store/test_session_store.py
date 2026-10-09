"""STUB: an `init` agent (default bindings); the session store the CLI declared was created.

Real case still to write: conversation state persists across invocations of one session_id and is
isolated between sessions.
"""

from __future__ import annotations

import pytest
from agentbricks_cli import AgentbricksCli


@pytest.mark.usefixtures("workspace_client")
def test_session_store_created(agentbricks_cli: AgentbricksCli) -> None:
    project = agentbricks_cli.new_project("cli")
    agentbricks_cli.deploy(project)
    resource = project.session_resource or {}

    assert resource.get("session_store_name") == project.session_store_config, resource


@pytest.mark.xfail(reason="STUB: no session-persistence case written yet", run=False)
def test_session_persists_across_invocations(agentbricks_cli: AgentbricksCli) -> None:
    raise NotImplementedError
