"""Deploy grants that belong to no single tool (deploy-only): nothing transitive, nothing lost."""

from __future__ import annotations

from harness.agentbricks_cli import AgentbricksCli
from harness.project import Project
from harness.run import TOOL_RESOURCE_PREFIX
from harness.workspace import Workspace
from tool_definitions import assert_declared, uc_function_tool
from uc_objects import UcFunction


def test_grants(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    workspace_client: Workspace,
    uc_function: UcFunction,
    authoring: str,
    subtests,
) -> None:
    tool = uc_function_tool(uc_function)
    # `init` and direct authoring both bind tracing, a resource Agent Bricks does not own.
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name
    if authoring == "cli":
        agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
        assert_declared(project, tool)
    else:
        project.write_manifest([tool])
    agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)

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
