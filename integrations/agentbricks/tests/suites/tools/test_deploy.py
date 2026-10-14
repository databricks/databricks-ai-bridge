"""Each tool works on a deployed App, and its UC grant is applied, per authoring path.

CLI authoring is incremental: add one tool, redeploy the same App, verify it, then add the next.
After every redeploy the grants of all tools added so far must still hold, which is the idempotency
check. Direct authoring writes every tool into agent.toml and deploys once. A subtest per tool
reports each tool independently and the sequence continues past a failure.
"""

from __future__ import annotations

from harness.agent_endpoint import AgentEndpoint
from harness.agentbricks_cli import AgentbricksCli
from harness.logs import log
from harness.project import Project
from harness.workspace import Workspace
from tool_definitions import Tool, assert_endpoint_invoke, bind, invoke_and_check


def test_deploy(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    workspace_client: Workspace,
    tools: tuple[Tool, ...],
    authoring: str,
    subtests,
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name

    def assert_granted(tool: Tool) -> None:
        if tool.grant is not None:
            assert workspace_client.granted(app, tool.grant), (
                f"{tool.name}: {tool.grant} not granted to {app}: "
                f"{sorted(workspace_client.granted_tuples(app))}"
            )

    def wait_for_grant(tool: Tool) -> None:
        if tool.grant is not None:
            waited = workspace_client.wait_until_effective(app, tool.grant)
            log(f"{tool.name}: {tool.grant[3]} effective for {app} after {waited:.0f}s")

    def grant_transitive_if_needed(tool: Tool) -> None:
        # The nested function is deliberately outside Agent Bricks' grants; the tool only works
        # once this manual grant is in place.
        if tool.transitive_function:
            workspace_client.grant_transitive(app, tool.transitive_function)

    if authoring == "cli":
        for index, tool in enumerate(tools):
            with subtests.test(tool=tool.name):
                bind(agentbricks_cli, project, tool)
                agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
                endpoint = AgentEndpoint.deployed(workspace_client, app)
                wait_for_grant(tool)
                grant_transitive_if_needed(tool)
                invoke_and_check(endpoint, tool)
                for added in tools[: index + 1]:
                    assert_granted(added)
        with subtests.test(step="endpoint-invoke"):
            assert_endpoint_invoke(agentbricks_cli, endpoint)
        return

    project.write_manifest(tools)
    agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
    endpoint = AgentEndpoint.deployed(workspace_client, app)
    for tool in tools:
        with subtests.test(tool=tool.name):
            wait_for_grant(tool)
            grant_transitive_if_needed(tool)
            invoke_and_check(endpoint, tool)
            assert_granted(tool)
    with subtests.test(step="endpoint-invoke"):
        assert_endpoint_invoke(agentbricks_cli, endpoint)
