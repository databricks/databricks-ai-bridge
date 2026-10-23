"""`agentbricks deploy`: every tool works on the deployed App.

`agentbricks deploy` returns once the App is running its new source, so the test invokes it at once.
"""

from __future__ import annotations

import json

import pytest
from harness.agentbricks_cli import AgentbricksCli
from harness.cleanup import Cleanup
from harness.project import Project
from tool_definitions import Tool


# Deploy needs a live workspace.
@pytest.mark.usefixtures("workspace_client")
def test_deploy(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    cleanup: Cleanup,
    tools: tuple[Tool, ...],
    subtests,
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name
    # Before deploy: a deploy that fails partway can still leave the App behind.
    cleanup(f"deployments delete {app} --delete-source --yes")
    for tool in tools:
        if tool.add_command:
            agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
        project.write_files(tool.files)

    deployed = json.loads(
        agentbricks_cli.run(
            f"--output json deploy {app} --source {project.path}", timeout=2400
        ).stdout
    )
    cleanup(f"memory stores delete memory-stores/{deployed['memory_store']} --yes")
    cleanup(f"sessions stores delete {deployed['session_store']} --yes")
    for tool in tools:
        with subtests.test(tool=tool.name):
            output = agentbricks_cli.invoke(app, tool.prompt)
            tool.check(output)
