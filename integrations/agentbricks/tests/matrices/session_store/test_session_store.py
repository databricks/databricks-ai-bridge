"""STUB: an `init` agent (default bindings); the session store the CLI declared was created.

Real case still to write: conversation state persists across invocations of one session_id and is
isolated between sessions.
"""

from __future__ import annotations

import json

import pytest
from agentbricks_cli import AgentbricksCli
from common import EXPECTED, Evidence, recorded_row


def test_session_store_created(agentbricks_cli: AgentbricksCli, evidence: Evidence) -> None:
    project = agentbricks_cli.new_project("cli")
    command = f"agentbricks sessions stores create --name {project.session_store_config}"
    with recorded_row(
        evidence,
        "cli",
        "deploy",
        "session_store",
        command=command,
        expected=EXPECTED["session_store"],
    ) as row:
        row.app_name = project.app_name
        row.log_path = agentbricks_cli.deploy(project).log_path
        resource = project.session_resource or {}
        row.serialized = json.dumps(resource, sort_keys=True)

        assert resource.get("session_store_name") == project.session_store_config, resource


@pytest.mark.xfail(reason="STUB: no session-persistence case written yet", run=False)
def test_session_persists_across_invocations(
    agentbricks_cli: AgentbricksCli, evidence: Evidence
) -> None:
    raise NotImplementedError
