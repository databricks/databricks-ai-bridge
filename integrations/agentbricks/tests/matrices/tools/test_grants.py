"""Deploy grants that belong to no single tool (deploy-only): nothing transitive, nothing lost."""

from __future__ import annotations

from agentbricks_cli import AgentbricksCli
from common import TOOL_RESOURCE_PREFIX, Evidence, Inputs
from tools import bind, tools_for
from workspace_client import Workspace


def test_grants(
    agentbricks_cli: AgentbricksCli,
    workspace_client: Workspace,
    inputs: Inputs,
    evidence: Evidence,
    authoring: str,
    subtests,
) -> None:
    uc_function = next(tool for tool in tools_for(inputs) if tool.name == "uc_function")
    # `init` and direct authoring both bind tracing, a resource Agent Bricks does not own.
    project = agentbricks_cli.new_project(authoring)
    app = project.app_name
    if authoring == "cli":
        bind(agentbricks_cli, project, uc_function)
    else:
        agentbricks_cli.write_manifest(project, [uc_function])
    agentbricks_cli.deploy(project, app)
    state: dict = {}

    with subtests.test(check="transitive-function-not-auto-granted"):
        # Raises if the hidden nested function is already granted or listed as an App resource.
        state.update(workspace_client.transitive_state(app, inputs.transitive_uc_function))

    with subtests.test(check="manual-grant-is-direct-and-effective"):
        granted = workspace_client.grant_transitive(app, inputs.transitive_uc_function)
        if state:
            evidence.record_grant_check(
                app,
                authoring=authoring,
                service_principal_client_id=state["service_principal_client_id"],
                initial={
                    "transitive_direct_privileges": state["direct_privileges"],
                    "transitive_effective_privileges": state["effective_privileges"],
                },
                post_manual_transitive_grant=granted,
                manual_transitive_grant_applied=True,
            )
        assert "EXECUTE" in granted["direct_privileges"]
        assert "EXECUTE" in granted["effective_privileges"]

    with subtests.test(check="non-tool-resource-preserved"):
        unrelated = workspace_client.non_tool_resources(app)
        assert unrelated, "No non-tool App resource; the tracing experiment should be attached"
        assert not any(r["name"].startswith(TOOL_RESOURCE_PREFIX) for r in unrelated)
