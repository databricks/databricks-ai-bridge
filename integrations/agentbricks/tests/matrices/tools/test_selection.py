"""MULTI-TOOL SELECTION check (deploy-only): with every tool bound, the agent picks the one asked for.

The incremental tests cannot catch a regression where binding several tools makes the model, or
the tool surface, route to the wrong one, so this is the one test that binds them all at once.
"""

from __future__ import annotations

from agentbricks_cli import AgentbricksCli
from provisioning import UcFunction
from tools import PYTHON_MARKER, UC_MARKER, Tool, bind
from workspace_client import Workspace

# Names the Unity Catalog function without its tool id, and the Python helper only to exclude it.
PROMPT = (
    "Use the Unity Catalog function tool to compute the Agent Bricks marker for the value "
    "'matrix', not the local Python helper tool. Return the called tool's exact result."
)


def test_selects_uc_function_not_python_marker(
    agentbricks_cli: AgentbricksCli,
    workspace_client: Workspace,
    tools: tuple[Tool, ...],
    uc_function: UcFunction,
    authoring: str,
) -> None:
    project = agentbricks_cli.new_project(authoring)
    app = project.app_name
    if authoring == "cli":
        for tool in tools:
            bind(agentbricks_cli, project, tool)
    else:
        agentbricks_cli.write_manifest(project, tools)
    agent = agentbricks_cli.deploy(project, app)
    # The nested function is deliberately outside Agent Bricks' grants.
    workspace_client.grant_transitive(app, uc_function.nested_function)
    serialized = agent.invoke(PROMPT, "deploy-selection")

    assert UC_MARKER in serialized, f"The UC function tool was not called: {serialized[:2000]}"
    assert PYTHON_MARKER not in serialized, (
        f"The Python marker tool was called too: {serialized[:2000]}"
    )
