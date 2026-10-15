"""STUB: an `init` agent (default bindings); deploy created the memory store agent.toml declares.

Real case still to write: the agent writes a memory in one invocation and recalls it in another.
"""

from __future__ import annotations

import json

import pytest
from harness.agentbricks_cli import AgentbricksCli
from harness.cleanup import Cleanup
from harness.project import Project


# Deploy needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
def test_memory_store_created(
    agentbricks_cli: AgentbricksCli, project: Project, cleanup: Cleanup
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    # Registered before deploy: a deploy that fails partway can still leave them behind.
    cleanup(f"memory stores delete memory-stores/{project.declared_store('memory')} --yes")
    cleanup(f"sessions stores delete {project.declared_store('session')} --yes")
    cleanup(f"deployments delete {project.app_name} --yes")
    declared = project.declared_store("memory")
    assert declared, "agentbricks init declared no memory store"
    agentbricks_cli.run(f"deploy {project.app_name} --source {project.path}", timeout=2400)

    store = json.loads(
        agentbricks_cli.run(f"--output json memory stores get memory-stores/{declared}").stdout
    )
    assert store.get("display_name") == declared, store
    assert store.get("name") == f"memory-stores/{declared}", store


@pytest.mark.xfail(reason="STUB: no memory write/recall case written yet", run=False)
def test_memory_recalled_across_invocations(agentbricks_cli: AgentbricksCli) -> None:
    raise NotImplementedError
