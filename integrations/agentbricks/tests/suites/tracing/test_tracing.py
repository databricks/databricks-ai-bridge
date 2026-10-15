"""STUB: an `init` agent (default bindings); the tracing experiment is bound and attached to the App.

Real case still to write: an invocation exports a trace to the experiment.
"""

from __future__ import annotations

import json

import pytest
from harness.agentbricks_cli import AgentbricksCli
from harness.cleanup import Cleanup
from harness.project import Project
from harness.workspace import Workspace, app_resources


def test_tracing_experiment_bound(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    cleanup: Cleanup,
    workspace_client: Workspace,
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    # Registered before deploy: a deploy that fails partway can still leave them behind.
    cleanup(f"memory stores delete memory-stores/{project.declared_store('memory')} --yes")
    cleanup(f"sessions stores delete {project.declared_store('session')} --yes")
    cleanup(f"deployments delete {project.app_name} --yes")
    experiment_name = project.manifest().get("tracing", {}).get("experiment_name")
    agentbricks_cli.run(f"deploy {project.app_name} --source {project.path}", timeout=2400)
    assert experiment_name, "agent.toml binds no tracing experiment"
    deployment = json.loads(
        agentbricks_cli.run(f"--output json deployments get {project.app_name}").stdout
    )

    # Deploy attaches the experiment as an App resource.
    experiments = [r for r in app_resources(deployment) if "experiment" in r]
    assert experiments, "No experiment App resource"
    expected_id = workspace_client.experiment_id(experiment_name)
    assert any(str(r["experiment"].get("experiment_id")) == expected_id for r in experiments), (
        f"{experiment_name} ({expected_id}) not in {experiments}"
    )


@pytest.mark.xfail(reason="STUB: no trace-export case written yet", run=False)
def test_invocation_exports_trace(agentbricks_cli: AgentbricksCli) -> None:
    raise NotImplementedError
