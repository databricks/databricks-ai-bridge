"""`agentbricks dev`: every tool added with `tools add` works on the local dev server.

Each dev server gets its own free port, so dev runs can go in parallel.
"""

from __future__ import annotations

from harness.agentbricks_cli import AgentbricksCli, free_port
from harness.project import Project
from tool_definitions import Tool


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
    with agentbricks_cli.start(f"dev --source {project.path} --app-port {port}") as server:
        server.wait_until_serving(port)
        for tool in tools:
            with subtests.test(tool=tool.name):
                output = agentbricks_cli.invoke(
                    f"--url http://127.0.0.1:{port} --no-auth", tool.prompt
                )
                tool.check(output)
