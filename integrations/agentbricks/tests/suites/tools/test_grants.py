"""Deploy grants only what a tool binds: a function the bound UC function calls needs a manual grant."""

from __future__ import annotations

import json

from harness.agentbricks_cli import AgentbricksCli
from harness.project import Project
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
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name
    if authoring == "cli":
        agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
        assert_declared(project, tool)
    else:
        project.write_manifest([tool])
    agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
    deployment = json.loads(agentbricks_cli.run(f"--output json deployments get {app}").stdout)

    with subtests.test(check="transitive-function-not-auto-granted"):
        # Raises if the hidden nested function is already granted or listed as an App resource.
        workspace_client.transitive_state(deployment, uc_function.nested_function)

    with subtests.test(check="manual-grant-is-direct-and-effective"):
        granted = workspace_client.grant_transitive(deployment, uc_function.nested_function)
        assert "EXECUTE" in granted["direct_privileges"]
        assert "EXECUTE" in granted["effective_privileges"]
