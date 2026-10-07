"""Each tool works on a deployed App, and its UC grant is applied, per authoring path.

CLI authoring is incremental: add one tool, redeploy the same App, verify it, then add the next.
After every redeploy the grants of all tools added so far must still hold, which is the idempotency
check. Direct authoring writes every tool into agent.toml and deploys once. A subtest per tool
reports each tool independently and the sequence continues past a failure.
"""

from __future__ import annotations

from agentbricks_cli import AgentbricksCli
from common import Evidence, Inputs
from tools import bind, invoke_and_check, setup_failures, tool_row, tools_for
from workspace_client import Workspace


def test_deploy(
    agentbricks_cli: AgentbricksCli,
    workspace_client: Workspace,
    inputs: Inputs,
    evidence: Evidence,
    authoring: str,
    subtests,
) -> None:
    tools = tools_for(inputs)
    project = agentbricks_cli.new_project(authoring)
    app = project.app_name

    def assert_granted(tool) -> None:
        if tool.grant is not None:
            assert workspace_client.granted(app, tool.grant), (
                f"{tool.name}: {tool.grant} not granted to {app}: "
                f"{sorted(workspace_client.granted_tuples(app))}"
            )

    def grant_transitive_if_needed(tool) -> None:
        # The nested function is deliberately outside Agent Bricks' grants; the tool only works
        # once this manual grant is in place.
        if tool.needs_transitive_grant:
            workspace_client.grant_transitive(app, inputs.transitive_uc_function)

    if authoring == "cli":
        for index, tool in enumerate(tools):
            with (
                subtests.test(tool=tool.name),
                tool_row(evidence, authoring, "deploy", tool) as row,
            ):
                row.app_name = app
                bind(agentbricks_cli, project, tool)
                agent = agentbricks_cli.deploy(project, app)
                grant_transitive_if_needed(tool)
                invoke_and_check(row, agent, tool)
                for added in tools[: index + 1]:
                    assert_granted(added)
        return

    with setup_failures(evidence, authoring, "deploy", tools, app):
        agentbricks_cli.write_manifest(project, tools)
        agent = agentbricks_cli.deploy(project, app)
    for tool in tools:
        with subtests.test(tool=tool.name), tool_row(evidence, authoring, "deploy", tool) as row:
            grant_transitive_if_needed(tool)
            invoke_and_check(row, agent, tool)
            assert_granted(tool)
