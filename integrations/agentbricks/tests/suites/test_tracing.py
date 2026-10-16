"""`agentbricks tracing`: an invocation of the deployed App leaves a trace in init's experiment.

The experiment is the project's own, named after it, so any trace in it came from this test.
"""

from __future__ import annotations

import json
import time

import pytest
from harness.agentbricks_cli import AgentbricksCli
from harness.cleanup import Cleanup
from harness.project import Project


# Deploy needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
def test_tracing(agentbricks_cli: AgentbricksCli, project: Project, cleanup: Cleanup) -> None:
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
    agentbricks_cli.invoke(app, "Reply with one word: hello.")

    # Traces are exported asynchronously, so a new one can take a little while to show up.
    deadline = time.monotonic() + 180
    while True:
        traces = json.loads(
            agentbricks_cli.run(f"--output json tracing list --source {project.path}").stdout
        )
        if traces or time.monotonic() > deadline:
            break
        time.sleep(15)
    assert traces, "No trace in the project's experiment after invoking the App"
