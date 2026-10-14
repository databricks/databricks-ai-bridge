"""MULTI-TOOL SELECTION check (deploy-only): with every tool bound, the agent picks the one asked for.

The incremental tests cannot catch a regression where binding several tools makes the model, or
the tool surface, route to the wrong one, so this is the one test that binds them all at once.
"""

from __future__ import annotations

from harness.cli import AgentbricksCli, AgentEndpoint
from harness.project import Project, pin_runtime
from harness.workspace import Workspace
from tool_definitions import PYTHON_MARKER, UC_MARKER, Tool, bind, excerpt
from uc_objects import UcFunction

# Names the Unity Catalog function without its tool id, and the Python helper only to exclude it.
PROMPT = (
    "Use the Unity Catalog function tool to compute the Agent Bricks marker for the value "
    "'matrix', not the local Python helper tool. Return the called tool's exact result."
)


def test_selects_uc_function_not_python_marker(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    workspace_client: Workspace,
    tools: tuple[Tool, ...],
    uc_function: UcFunction,
    authoring: str,
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    pin_runtime(project, agentbricks_cli.run_config)
    app = project.app_name
    if authoring == "cli":
        for tool in tools:
            bind(agentbricks_cli, project, tool)
    else:
        project.write_manifest(tools)
    agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
    endpoint = AgentEndpoint.deployed(workspace_client, app)
    # The nested function is deliberately outside Agent Bricks' grants.
    workspace_client.grant_transitive(app, uc_function.nested_function)
    serialized = endpoint.invoke(PROMPT, "deploy-selection")

    assert UC_MARKER in serialized, f"The UC function tool was not called: {excerpt(serialized)}"
    assert PYTHON_MARKER not in serialized, (
        f"The Python marker tool was called too: {excerpt(serialized)}"
    )
