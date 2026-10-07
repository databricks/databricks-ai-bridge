"""MULTI-TOOL SELECTION check (deploy-only): with every tool bound, the agent picks the one asked for.

The incremental tests cannot catch a regression where binding several tools makes the model, or
the tool surface, route to the wrong one, so this is the one test that binds them all at once.
"""

from __future__ import annotations

from agentbricks_cli import AgentbricksCli
from common import EXPECTED, Evidence, Inputs, recorded_row
from tools import PYTHON_MARKER, UC_MARKER, bind, tools_for
from workspace_client import Workspace

# Names the Unity Catalog function without its tool id, and the Python helper only to exclude it.
PROMPT = (
    "Use the Unity Catalog function tool to compute the Agent Bricks marker for the value "
    "'matrix', not the local Python helper tool. Return the called tool's exact result."
)


def test_selects_uc_function_not_python_marker(
    agentbricks_cli: AgentbricksCli,
    workspace_client: Workspace,
    inputs: Inputs,
    evidence: Evidence,
    authoring: str,
) -> None:
    tools = tools_for(inputs)
    project = agentbricks_cli.new_project(authoring)
    app = project.app_name
    with recorded_row(
        evidence,
        authoring,
        "deploy",
        "selection",
        command="agentbricks deploy",
        expected=EXPECTED["selection"],
        marker=UC_MARKER,
    ) as row:
        row.app_name = app
        if authoring == "cli":
            for tool in tools:
                bind(agentbricks_cli, project, tool)
        else:
            agentbricks_cli.write_manifest(project, tools)
        agent = agentbricks_cli.deploy(project, app)
        # The nested function is deliberately outside Agent Bricks' grants.
        workspace_client.grant_transitive(app, inputs.transitive_uc_function)
        row.command = agent.curl(PROMPT)
        row.log_path = agent.log_path
        row.app_url = agent.base_url
        row.serialized = agent.invoke(PROMPT, "deploy-selection")

        assert UC_MARKER in row.serialized, (
            f"The UC function tool was not called: {row.serialized[:2000]}"
        )
        assert PYTHON_MARKER not in row.serialized, (
            f"The Python marker tool was called too: {row.serialized[:2000]}"
        )
