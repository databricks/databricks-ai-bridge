"""Deploy grants that belong to no single tool (deploy-only): nothing transitive, nothing lost."""

from __future__ import annotations

from agentbricks_cli import AgentbricksCli
from common import TOOL_RESOURCE_PREFIX
from tools import bind, uc_function_tool
from uc_objects import UcFunction
from workspace_client import Workspace


def test_grants(
    agentbricks_cli: AgentbricksCli,
    workspace_client: Workspace,
    uc_function: UcFunction,
    authoring: str,
    subtests,
) -> None:
    tool = uc_function_tool(uc_function)
    # `init` and direct authoring both bind tracing, a resource Agent Bricks does not own.
    project = agentbricks_cli.new_project(authoring)
    app = project.app_name
    if authoring == "cli":
        bind(agentbricks_cli, project, tool)
    else:
        agentbricks_cli.write_manifest(project, [tool])
    agentbricks_cli.deploy(project, app)

    with subtests.test(check="transitive-function-not-auto-granted"):
        # Raises if the hidden nested function is already granted or listed as an App resource.
        workspace_client.transitive_state(app, uc_function.nested_function)

    with subtests.test(check="manual-grant-is-direct-and-effective"):
        granted = workspace_client.grant_transitive(app, uc_function.nested_function)
        assert "EXECUTE" in granted["direct_privileges"]
        assert "EXECUTE" in granted["effective_privileges"]

    with subtests.test(check="non-tool-resource-preserved"):
        unrelated = workspace_client.non_tool_resources(app)
        assert unrelated, "No non-tool App resource; the tracing experiment should be attached"
        assert not any(r["name"].startswith(TOOL_RESOURCE_PREFIX) for r in unrelated)
