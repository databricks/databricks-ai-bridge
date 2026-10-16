"""`agentbricks dev`: every tool added with `tools add` works on the local dev server.

Each dev server gets a free port, passed as both --app-port and PORT: the template server binds
$PORT, so an inherited PORT would override --app-port.
"""

from __future__ import annotations

import pytest
from harness.agentbricks_cli import AgentbricksCli, free_port
from harness.project import Project
from tool_definitions import Tool

# Local dev servers share run-local's proxy port, so all dev runs stay on one worker, in order.
pytestmark = pytest.mark.xdist_group("dev")


def test_dev(
    agentbricks_cli: AgentbricksCli, project: Project, tools: tuple[Tool, ...], subtests
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    for tool in tools:
        if tool.add_command:
            agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
        project.write_files(tool.files)

    port = free_port()
    with agentbricks_cli.start(
        f"dev --source {project.path} --app-port {port}", env={"PORT": str(port)}
    ) as server:
        server.wait_until_serving(port)
        for tool in tools:
            with subtests.test(tool=tool.name):
                output = agentbricks_cli.invoke(
                    f"--url http://127.0.0.1:{port} --no-auth", tool.prompt
                )
                tool.check(output)
