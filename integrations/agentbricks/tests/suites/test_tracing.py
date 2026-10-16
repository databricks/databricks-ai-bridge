"""`agentbricks tracing`: an invocation of the deployed App leaves a trace in init's experiment.

The experiment is the project's own, named after it, so any trace in it came from this test.
"""

from __future__ import annotations

import json
import time

import pytest
from harness.agentbricks_cli import AgentbricksCli, invocation_json
from harness.cleanup import Cleanup
from harness.project import Project


# Deploy needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
def test_tracing(agentbricks_cli: AgentbricksCli, project: Project, cleanup: Cleanup) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name
    # Registered before deploy: a deploy that fails partway can still leave them behind.
    cleanup(f"memory stores delete memory-stores/{project.declared_store('memory')} --yes")
    cleanup(f"sessions stores delete {project.declared_store('session')} --yes")
    cleanup(f"deployments delete {app} --yes")
    agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
    agentbricks_cli.run(
        f"--output json endpoint invoke {app} --path /api/invocations "
        f"--json {invocation_json('Reply with one word: hello.')}",
        timeout=360,
        retry_server_errors=True,
    )

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
