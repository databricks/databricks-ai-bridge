"""STUB: an `init` agent (default bindings); the memory store the CLI declared was created.

Real case still to write: the agent writes a memory in one invocation and recalls it in another.
"""

from __future__ import annotations

import json

import pytest
from agentbricks_cli import AgentbricksCli
from common import EXPECTED, Evidence, recorded_row


def test_memory_store_created(agentbricks_cli: AgentbricksCli, evidence: Evidence) -> None:
    project = agentbricks_cli.new_project("cli")
    command = f"agentbricks memory stores create --name {project.memory_store_config}"
    with recorded_row(
        evidence,
        "cli",
        "deploy",
        "memory_store",
        command=command,
        expected=EXPECTED["memory_store"],
    ) as row:
        row.app_name = project.app_name
        row.log_path = agentbricks_cli.deploy(project).log_path
        resource = project.memory_resource or {}
        row.serialized = json.dumps(resource, sort_keys=True)

        assert resource.get("display_name") == project.memory_store_config, resource
        assert str(resource.get("name")).startswith("memory-stores/"), resource
        assert project.memory_store_name == resource["name"]


@pytest.mark.xfail(reason="STUB: no memory write/recall case written yet", run=False)
def test_memory_recalled_across_invocations(
    agentbricks_cli: AgentbricksCli, evidence: Evidence
) -> None:
    raise NotImplementedError
