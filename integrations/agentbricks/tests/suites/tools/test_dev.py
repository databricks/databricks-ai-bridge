"""Each tool works in `agentbricks dev`, per authoring path.

CLI authoring is incremental: add one tool, restart dev, verify it, then add the next on the same
project. Direct authoring writes every tool into agent.toml and runs dev once. A subtest per tool
reports each tool independently and the sequence continues past a failure.

Each dev server gets a free port, passed as both --app-port and PORT: the template server binds
$PORT, so an inherited PORT would override --app-port.
"""

from __future__ import annotations

import pytest
from harness.agent_endpoint import AgentEndpoint
from harness.agentbricks_cli import AgentbricksCli, free_port
from harness.project import Project
from tool_definitions import (
    Tool,
    assert_declared,
    assert_replied_ok,
    invocation_json,
    invoke_and_check,
)

# Local dev servers share run-local's proxy port, so all dev runs stay on one worker, in order.
pytestmark = pytest.mark.xdist_group("dev")

OK_PROMPT = "Reply with the single word OK."


def test_dev(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    tools: tuple[Tool, ...],
    authoring: str,
    subtests,
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()

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
                    invoke_and_check(AgentEndpoint.local(server, port), tool)
        with subtests.test(step="endpoint-invoke"):
            port = free_port()
            with agentbricks_cli.start(
                f"dev --source {project.path} --app-port {port}", env={"PORT": str(port)}
            ) as server:
                endpoint = AgentEndpoint.local(server, port)
                output = agentbricks_cli.run(
                    f"endpoint invoke --url {endpoint.base_url} --no-auth "
                    f"--path /api/invocations --json {invocation_json(OK_PROMPT)}"
                ).stdout
            assert_replied_ok(output)
        return

    project.write_manifest(tools)
    port = free_port()
    with agentbricks_cli.start(
        f"dev --source {project.path} --app-port {port}", env={"PORT": str(port)}
    ) as server:
        endpoint = AgentEndpoint.local(server, port)
        for tool in tools:
            with subtests.test(tool=tool.name):
                invoke_and_check(endpoint, tool)
        with subtests.test(step="endpoint-invoke"):
            output = agentbricks_cli.run(
                f"endpoint invoke --url {endpoint.base_url} --no-auth "
                f"--path /api/invocations --json {invocation_json(OK_PROMPT)}"
            ).stdout
            assert_replied_ok(output)
