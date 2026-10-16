"""`agentbricks sessions`: the deployed agent keeps a conversation's history, and only that one's.

The word is random, so only the session store can supply it.
"""

from __future__ import annotations

import json
import uuid

import pytest
from harness.agentbricks_cli import AgentbricksCli, excerpt, reply
from harness.cleanup import Cleanup
from harness.project import Project

QUESTION = "What secret word did I tell you earlier in this conversation? Reply with just the word."


# Deploy needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
def test_sessions(agentbricks_cli: AgentbricksCli, project: Project, cleanup: Cleanup) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name
    # Before deploy: a deploy that fails partway can still leave the App behind.
    cleanup(f"deployments delete {app} --yes")
    deployed = json.loads(
        agentbricks_cli.run(
            f"--output json deploy {app} --source {project.path}", timeout=2400
        ).stdout
    )
    cleanup(f"memory stores delete memory-stores/{deployed['memory_store']} --yes")
    cleanup(f"sessions stores delete {deployed['session_store']} --yes")

    session = str(uuid.uuid4())
    word = f"AGENTBRICKS_SESSION_{uuid.uuid4().hex[:12]}"

    def invoke(prompt: str, session_id: str) -> str:
        return agentbricks_cli.run(
            f"--output json endpoint invoke {app} --path /api/invocations "
            f"--json {agentbricks_cli.invocation_json(prompt, session_id=session_id)}",
            timeout=360,
            retry_server_errors=True,
        ).stdout

    invoke(f"The secret word for this conversation is {word}. Reply with just: noted.", session)
    same = invoke(QUESTION, session)
    assert word in reply(same), f"The session lost its history: {excerpt(same)}"
    other = invoke(QUESTION, str(uuid.uuid4()))
    assert word not in reply(other), f"Another session saw this one's history: {excerpt(other)}"
