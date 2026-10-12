"""STUB: an `init` agent (default bindings); the tracing experiment is bound and attached to the App.

Real case still to write: an invocation exports a trace to the experiment.
"""

from __future__ import annotations

import pytest
from agentbricks_cli import AgentbricksCli
from common import TOOL_RESOURCE_PREFIX
from workspace_client import Workspace


def test_tracing_experiment_bound(
    agentbricks_cli: AgentbricksCli, workspace_client: Workspace
) -> None:
    project = agentbricks_cli.new_project("cli")
    agentbricks_cli.deploy(project)
    assert project.experiment_name, "agent.toml binds no tracing experiment"

    # Deploy attaches the experiment as an App resource that Agent Bricks does not own.
    experiments = [
        r for r in workspace_client.non_tool_resources(project.app_name) if "experiment" in r
    ]
    assert experiments, "No experiment App resource among the non-tool resources"
    assert not any(r["name"].startswith(TOOL_RESOURCE_PREFIX) for r in experiments)
    expected_id = workspace_client.experiment_id(project.experiment_name)
    assert any(str(r["experiment"].get("experiment_id")) == expected_id for r in experiments), (
        f"{project.experiment_name} ({expected_id}) not in {experiments}"
    )


@pytest.mark.xfail(reason="STUB: no trace-export case written yet", run=False)
def test_invocation_exports_trace(agentbricks_cli: AgentbricksCli) -> None:
    raise NotImplementedError
