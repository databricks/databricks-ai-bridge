"""`agentbricks tracing`: an invocation of the deployed App leaves a trace in init's experiment,
and `tracing list` / `get` read the local dev store.

The experiment is the project's own, named after it, so any trace in it came from this test.
"""

from __future__ import annotations

import json
import pathlib
import shutil
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


def test_tracing_reads_the_local_dev_store(
    agentbricks_cli: AgentbricksCli, tmp_path: pathlib.Path
) -> None:
    """A trace written to the MLflow server `agentbricks dev` starts is read back by the CLI.

    The project has no [tracing] table, so list and get resolve to the local store without a
    workspace lookup.
    """
    if shutil.which("uvx") is None:
        pytest.skip("the local MLflow server runs through uvx")
    project = tmp_path / "agent"
    project.mkdir()
    (project / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "agentbricks"\n'
    )
    _write_local_trace(project)

    listed = json.loads(
        agentbricks_cli.run(f"--output json tracing list --source {project}", timeout=180).stdout
    )
    assert listed, "`agentbricks tracing list` returned no local traces"
    trace_id = listed[0]["trace_id"]
    assert trace_id.startswith("tr-")

    got = json.loads(
        agentbricks_cli.run(
            f"--output json tracing get {trace_id} --source {project}", timeout=180
        ).stdout
    )
    assert got["trace_id"] == trace_id
    assert got["status"]


def _write_local_trace(project: pathlib.Path) -> None:
    """Log one span through the same local MLflow server `agentbricks dev` runs, then stop it."""
    import mlflow

    from databricks_agentbricks.clients.local_tracing_client import (
        _wait_for_server,
        start_local_tracing_server,
        stop_local_tracing_server,
    )

    server, env = start_local_tracing_server(project)
    assert server is not None, "the local MLflow server could not be started"
    try:
        uri = env["MLFLOW_TRACKING_URI"]
        assert _wait_for_server(uri, server, timeout=180), "the local MLflow server never came up"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(env["MLFLOW_EXPERIMENT_NAME"])
        with mlflow.start_span(name="probe") as span:
            span.set_inputs({"q": "hi"})
            span.set_outputs({"a": "there"})
        mlflow.flush_trace_async_logging(terminate=True)
    finally:
        stop_local_tracing_server(server)
