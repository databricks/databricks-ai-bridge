"""`agentbricks tools`: after `tools remove`, the agent no longer has the tool.

`tools add` is covered by `test_dev` and `test_deploy`, which invoke every added tool.
"""

from __future__ import annotations

import pytest
from harness.agentbricks_cli import AgentbricksCli, free_port
from harness.project import Project
from tool_definitions import Tool

# Local dev servers share run-local's proxy port, so all dev runs stay on one worker, in order.
pytestmark = pytest.mark.xdist_group("dev")


def test_tools_remove(
    agentbricks_cli: AgentbricksCli, project: Project, tools: tuple[Tool, ...], subtests
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    removed = [tool for tool in tools if tool.add_command and tool.tool_id]
    for tool in removed:
        agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
    for tool in removed:
        agentbricks_cli.run(f"tools remove {tool.tool_id} --source {project.path}")

    port = free_port()
    with agentbricks_cli.start(
        f"dev --source {project.path} --app-port {port}", env={"PORT": str(port)}
    ) as server:
        server.wait_until_serving(port)
        for tool in removed:
            with subtests.test(tool=tool.name):
                output = agentbricks_cli.invoke(
                    f"--url http://127.0.0.1:{port} --no-auth", tool.prompt
                )
                tool.assert_not_called(output)
