"""STUB: an `init` agent (default bindings); the tracing experiment is bound and attached to the App.

Real case still to write: an invocation exports a trace to the experiment.
"""

from __future__ import annotations

import json

import pytest
from agentbricks_cli import AgentbricksCli
from common import EXPECTED, TOOL_RESOURCE_PREFIX, Evidence, recorded_row
from workspace_client import Workspace


def test_tracing_experiment_bound(
    agentbricks_cli: AgentbricksCli, workspace_client: Workspace, evidence: Evidence
) -> None:
    project = agentbricks_cli.new_project("cli")
    with recorded_row(
        evidence,
        "cli",
        "deploy",
        "tracing",
        command="agentbricks deploy (agent.toml [tracing] experiment_name)",
        expected=EXPECTED["tracing"],
    ) as row:
        row.app_name = project.app_name
        row.log_path = agentbricks_cli.deploy(project).log_path
        assert project.experiment_name, "agent.toml binds no tracing experiment"
        row.serialized = json.dumps({"experiment_name": project.experiment_name})

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
def test_invocation_exports_trace(agentbricks_cli: AgentbricksCli, evidence: Evidence) -> None:
    raise NotImplementedError
