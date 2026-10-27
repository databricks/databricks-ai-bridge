"""`agentbricks tracing`: an invocation of the deployed App leaves a trace in init's experiment,
and an invocation of `agentbricks dev` leaves one `tracing list` / `get` read locally.

The experiment is the project's own, named after it, so any trace in it came from this test.
"""

from __future__ import annotations

import json
import time

import pytest
from harness.agentbricks_cli import AgentbricksCli, free_port
from harness.cleanup import Cleanup
from harness.project import Project


# Deploy needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
def test_tracing(agentbricks_cli: AgentbricksCli, project: Project, cleanup: Cleanup) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name
    # Before deploy: a deploy that fails partway can still leave the App behind.
    cleanup(f"deployments delete {app} --delete-source --yes")
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


def test_tracing_reads_local_dev_traces(agentbricks_cli: AgentbricksCli, project: Project) -> None:
    """With tracing unbound, `agentbricks dev` still traces locally and `tracing list`/`get` read it."""
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    agentbricks_cli.run(f"tracing unbind --source {project.path}")

    port = free_port()
    with agentbricks_cli.start(
        f"dev --source {project.path} --app-port {port}", env={"PORT": str(port)}
    ) as server:
        server.wait_until_serving(port)
        agentbricks_cli.invoke(f"--url http://127.0.0.1:{port} --no-auth", "Reply with: hello.")
        # Traces are exported asynchronously, so a new one can take a little while to show up.
        deadline = time.monotonic() + 120
        while True:
            traces = json.loads(
                agentbricks_cli.run(
                    f"--output json tracing list --source {project.path}", timeout=180
                ).stdout
            )
            if traces or time.monotonic() > deadline:
                break
            time.sleep(5)
    assert traces, "No local trace after invoking the dev server"

    trace_id = traces[0]["trace_id"]
    got = json.loads(
        agentbricks_cli.run(
            f"--output json tracing get {trace_id} --source {project.path}", timeout=180
        ).stdout
    )
    assert got["trace_id"] == trace_id
