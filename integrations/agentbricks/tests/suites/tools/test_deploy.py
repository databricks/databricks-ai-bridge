"""Each tool works on a deployed App, per authoring path: invoking it returns its marker.

CLI authoring adds every tool with `agentbricks tools add`; direct authoring writes them into
agent.toml. Either way the App is deployed once, and a subtest per tool invokes it, so each tool is
reported independently.

`agentbricks deploy` returns once the App is running its new source, so the next step can use it.
"""

from __future__ import annotations

import json
from typing import Any

from harness.agentbricks_cli import AgentbricksCli
from harness.cleanup import Cleanup
from harness.logs import log
from harness.project import Project
from harness.workspace import Workspace
from tool_definitions import Tool, assert_declared, invocation_json


def test_deploy(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    cleanup: Cleanup,
    workspace_client: Workspace,
    tools: tuple[Tool, ...],
    authoring: str,
    subtests,
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name
    # Registered before deploy: a deploy that fails partway can still leave them behind.
    cleanup(f"memory stores delete memory-stores/{project.declared_store('memory')} --yes")
    cleanup(f"sessions stores delete {project.declared_store('session')} --yes")
    cleanup(f"deployments delete {app} --yes")

    def invoke_and_check(tool: Tool, deployment: dict[str, Any]) -> None:
        if tool.grant is not None:
            # Deploy adds the grant before returning, but UC can apply it a little later, and a
            # tool called in between is denied inside a successful response.
            waited = workspace_client.wait_until_effective(deployment, tool.grant)
            log(f"{tool.name}: {tool.grant[3]} effective for {app} after {waited:.0f}s")
        output = agentbricks_cli.run(
            f"--output json endpoint invoke {app} --path /api/invocations "
            f"--json {invocation_json(tool.prompt)}",
            timeout=360,
            retry_server_errors=True,
        ).stdout
        tool.check(output)

    if authoring == "cli":
        for tool in tools:
            if tool.add_command:
                agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
            project.write_files(tool.files)
            assert_declared(project, tool)
    elif authoring == "direct":
        project.write_manifest(tools)
    agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
    deployment = json.loads(agentbricks_cli.run(f"--output json deployments get {app}").stdout)
    for tool in tools:
        with subtests.test(tool=tool.name):
            invoke_and_check(tool, deployment)
