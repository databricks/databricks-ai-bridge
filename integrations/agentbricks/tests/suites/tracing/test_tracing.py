"""STUB: an `init` agent (default bindings); the tracing experiment is bound and attached to the App.

Real case still to write: an invocation exports a trace to the experiment.
"""

from __future__ import annotations

import pytest
from harness.agent_endpoint import AgentEndpoint
from harness.agentbricks_cli import AgentbricksCli
from harness.project import Project
from harness.run import TOOL_RESOURCE_PREFIX
from harness.workspace import Workspace


def test_tracing_experiment_bound(
    agentbricks_cli: AgentbricksCli, project: Project, workspace_client: Workspace
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    experiment_name = project.manifest().get("tracing", {}).get("experiment_name")
    agentbricks_cli.run(f"deploy {project.app_name} --source {project.path}", timeout=2400)
    AgentEndpoint.deployed(workspace_client, project.app_name)
    assert experiment_name, "agent.toml binds no tracing experiment"

    # Deploy attaches the experiment as an App resource that Agent Bricks does not own.
    experiments = [
        r for r in workspace_client.non_tool_resources(project.app_name) if "experiment" in r
    ]
    assert experiments, "No experiment App resource among the non-tool resources"
    assert not any(r["name"].startswith(TOOL_RESOURCE_PREFIX) for r in experiments)
    expected_id = workspace_client.experiment_id(experiment_name)
    assert any(str(r["experiment"].get("experiment_id")) == expected_id for r in experiments), (
        f"{experiment_name} ({expected_id}) not in {experiments}"
    )


@pytest.mark.xfail(reason="STUB: no trace-export case written yet", run=False)
def test_invocation_exports_trace(agentbricks_cli: AgentbricksCli) -> None:
    raise NotImplementedError
