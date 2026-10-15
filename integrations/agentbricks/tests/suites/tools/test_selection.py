"""MULTI-TOOL SELECTION check (deploy-only): with every tool bound, the agent picks the one asked for.

The incremental tests cannot catch a regression where binding several tools makes the model, or
the tool surface, route to the wrong one, so this is the one test that binds them all at once.
"""

from __future__ import annotations

import json

from harness.agentbricks_cli import AgentbricksCli
from harness.project import Project
from harness.workspace import Workspace
from tool_definitions import (
    PYTHON_MARKER,
    UC_MARKER,
    Tool,
    assert_declared,
    excerpt,
    invocation_json,
)
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
    project.pin_runtime()
    app = project.app_name
    if authoring == "cli":
        for tool in tools:
            if tool.add_command:
                agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
            project.write_files(tool.files)
            assert_declared(project, tool)
    else:
        project.write_manifest(tools)
    agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
    deployment = json.loads(agentbricks_cli.run(f"--output json deployments get {app}").stdout)
    # The nested function is deliberately outside Agent Bricks' grants.
    workspace_client.grant_transitive(deployment, uc_function.nested_function)
    serialized = agentbricks_cli.run(
        f"--output json endpoint invoke {app} --path /api/invocations "
        f"--json {invocation_json(PROMPT)}",
        timeout=360,
        retry_server_errors=True,
    ).stdout

    assert UC_MARKER in serialized, f"The UC function tool was not called: {excerpt(serialized)}"
    assert PYTHON_MARKER not in serialized, (
        f"The Python marker tool was called too: {excerpt(serialized)}"
    )
