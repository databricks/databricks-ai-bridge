"""`agentbricks memory`: deploy creates the memory store `agentbricks init` declares."""

from __future__ import annotations

import json

import pytest
from harness.agentbricks_cli import AgentbricksCli
from harness.cleanup import Cleanup
from harness.project import Project


# Deploy needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
def test_memory_store(agentbricks_cli: AgentbricksCli, project: Project, cleanup: Cleanup) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    store = project.declared_store("memory")
    assert store, "agentbricks init declared no memory store"
    # Registered before deploy: a deploy that fails partway can still leave them behind.
    cleanup(f"memory stores delete memory-stores/{store} --yes")
    cleanup(f"sessions stores delete {project.declared_store('session')} --yes")
    cleanup(f"deployments delete {project.app_name} --yes")
    agentbricks_cli.run(f"deploy {project.app_name} --source {project.path}", timeout=2400)

    found = json.loads(
        agentbricks_cli.run(f"--output json memory stores get memory-stores/{store}").stdout
    )
    assert found.get("name") == f"memory-stores/{store}", found
