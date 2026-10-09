"""Each tool works on a deployed App, and its UC grant is applied, per authoring path.

CLI authoring is incremental: add one tool, redeploy the same App, verify it, then add the next.
After every redeploy the grants of all tools added so far must still hold, which is the idempotency
check. Direct authoring writes every tool into agent.toml and deploys once. A subtest per tool
reports each tool independently and the sequence continues past a failure.
"""

from __future__ import annotations

from agentbricks_cli import AgentbricksCli
from tools import Tool, assert_endpoint_invoke, bind, invoke_and_check
from workspace_client import Workspace


def test_deploy(
    agentbricks_cli: AgentbricksCli,
    workspace_client: Workspace,
    tools: tuple[Tool, ...],
    authoring: str,
    subtests,
) -> None:
    project = agentbricks_cli.new_project(authoring)
    app = project.app_name

    def assert_granted(tool: Tool) -> None:
        if tool.grant is not None:
            assert workspace_client.granted(app, tool.grant), (
                f"{tool.name}: {tool.grant} not granted to {app}: "
                f"{sorted(workspace_client.granted_tuples(app))}"
            )

    def grant_transitive_if_needed(tool: Tool) -> None:
        # The nested function is deliberately outside Agent Bricks' grants; the tool only works
        # once this manual grant is in place.
        if tool.transitive_function:
            workspace_client.grant_transitive(app, tool.transitive_function)

    if authoring == "cli":
        for index, tool in enumerate(tools):
            with subtests.test(tool=tool.name):
                bind(agentbricks_cli, project, tool)
                agent = agentbricks_cli.deploy(project, app)
                grant_transitive_if_needed(tool)
                invoke_and_check(agent, tool)
                for added in tools[: index + 1]:
                    assert_granted(added)
        with subtests.test(step="endpoint-invoke"):
            assert_endpoint_invoke(agentbricks_cli, agent)
        return

    agentbricks_cli.write_manifest(project, tools)
    agent = agentbricks_cli.deploy(project, app)
    for tool in tools:
        with subtests.test(tool=tool.name):
            grant_transitive_if_needed(tool)
            invoke_and_check(agent, tool)
            assert_granted(tool)
    with subtests.test(step="endpoint-invoke"):
        assert_endpoint_invoke(agentbricks_cli, agent)
