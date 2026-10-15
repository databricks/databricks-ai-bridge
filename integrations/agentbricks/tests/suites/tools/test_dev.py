"""Each tool works in `agentbricks dev`, per authoring path.

CLI authoring is incremental: add one tool, restart dev, verify it, then add the next on the same
project. Direct authoring writes every tool into agent.toml and runs dev once. A subtest per tool
reports each tool independently and the sequence continues past a failure. Every check calls the
dev server with `agentbricks endpoint invoke`, the command the CLI tells users to run.

Each dev server gets a free port, passed as both --app-port and PORT: the template server binds
$PORT, so an inherited PORT would override --app-port.
"""

from __future__ import annotations

import pytest
from harness.agentbricks_cli import AgentbricksCli, free_port
from harness.project import Project
from tool_definitions import Tool, assert_declared, invocation_json

# Local dev servers share run-local's proxy port, so all dev runs stay on one worker, in order.
pytestmark = pytest.mark.xdist_group("dev")


def test_dev(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    tools: tuple[Tool, ...],
    authoring: str,
    subtests,
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()

    def check(tool: Tool, port: int) -> None:
        output = agentbricks_cli.run(
            f"--output json endpoint invoke --url http://127.0.0.1:{port} --no-auth "
            f"--path /api/invocations --json {invocation_json(tool.prompt)}",
            timeout=360,
            retry_server_errors=True,
        ).stdout
        tool.check(output)

    if authoring == "cli":
        for tool in tools:
            with subtests.test(tool=tool.name):
                if tool.add_command:
                    agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
                project.write_files(tool.files)
                assert_declared(project, tool)
                port = free_port()
                with agentbricks_cli.start(
                    f"dev --source {project.path} --app-port {port}", env={"PORT": str(port)}
                ) as server:
                    server.wait_until_serving(port)
                    check(tool, port)
    elif authoring == "direct":
        project.write_manifest(tools)
        port = free_port()
        with agentbricks_cli.start(
            f"dev --source {project.path} --app-port {port}", env={"PORT": str(port)}
        ) as server:
            server.wait_until_serving(port)
            for tool in tools:
                with subtests.test(tool=tool.name):
                    check(tool, port)
