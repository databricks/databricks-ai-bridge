"""`agentbricks memory`: the deployed agent remembers a fact in one session and recalls it in another.

Long-term memory is keyed by actor, so both invocations name the same one; the fact is random, so
only the memory store can supply it.
"""

from __future__ import annotations

import json
import uuid

import pytest
from harness.agentbricks_cli import AgentbricksCli, excerpt, tool_results
from harness.cleanup import Cleanup
from harness.project import Project


# Deploy needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
def test_memory(agentbricks_cli: AgentbricksCli, project: Project, cleanup: Cleanup) -> None:
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

    actor = f"e2e-{uuid.uuid4().hex[:8]}"
    codename = f"AGENTBRICKS_MEMORY_{uuid.uuid4().hex[:12]}"

    def invoke(prompt: str) -> str:
        # Each invocation is a new session; only the actor carries over.
        return agentbricks_cli.run(
            f"--output json endpoint invoke {app} --path /api/invocations "
            f"--json {agentbricks_cli.invocation_json(prompt, actor=actor)}",
            timeout=360,
            retry_server_errors=True,
        ).stdout

    remembered = invoke(
        f"Remember this fact about me in long-term memory: my codename is {codename}."
    )
    assert any(
        r.get("name") == "remember" and r.get("status") == "success"
        for r in tool_results(remembered)
    ), f"The agent did not remember the fact: {excerpt(remembered)}"

    recalled = invoke("What is my codename? Look it up in your long-term memory.")
    assert any(
        r.get("name") == "recall" and codename in str(r.get("content"))
        for r in tool_results(recalled)
    ), f"Long-term memory did not return the fact in a new session: {excerpt(recalled)}"
