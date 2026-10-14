"""STUB: an `init` agent (default bindings); the memory store the CLI declared was created.

Real case still to write: the agent writes a memory in one invocation and recalls it in another.
"""

from __future__ import annotations

import pytest
from harness.cli import AgentbricksCli, AgentEndpoint
from harness.workspace import Workspace


def test_memory_store_created(agentbricks_cli: AgentbricksCli, workspace_client: Workspace) -> None:
    project = agentbricks_cli.new_project("cli")
    agentbricks_cli.run(f"deploy {project.app_name} --source {project.path}", timeout=2400)
    AgentEndpoint.deployed(workspace_client, project.app_name)
    resource = project.memory_resource or {}

    assert resource.get("display_name") == project.memory_store_config, resource
    assert str(resource.get("name")).startswith("memory-stores/"), resource
    assert project.memory_store_name == resource["name"]


@pytest.mark.xfail(reason="STUB: no memory write/recall case written yet", run=False)
def test_memory_recalled_across_invocations(agentbricks_cli: AgentbricksCli) -> None:
    raise NotImplementedError
