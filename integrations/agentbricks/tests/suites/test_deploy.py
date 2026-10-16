"""`agentbricks deploy` and `deployments`: every tool works on the deployed App, then delete removes it.

`agentbricks deploy` returns once the App is running its new source, so the test invokes it at once.
"""

from __future__ import annotations

import json

from harness.agentbricks_cli import AgentbricksCli, invocation_json, not_found
from harness.cleanup import Cleanup, wait_until_app_gone
from harness.logs import log
from harness.project import Project
from harness.workspace import Workspace
from tool_definitions import Tool, assert_declared


def test_deploy(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    cleanup: Cleanup,
    workspace_client: Workspace,
    tools: tuple[Tool, ...],
    subtests,
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name
    # Registered before deploy: a deploy that fails partway can still leave them behind.
    cleanup(f"memory stores delete memory-stores/{project.declared_store('memory')} --yes")
    cleanup(f"sessions stores delete {project.declared_store('session')} --yes")
    cleanup(f"deployments delete {app} --yes")
    for tool in tools:
        if tool.add_command:
            agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
        project.write_files(tool.files)
        assert_declared(project, tool)

    agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
    deployment = json.loads(agentbricks_cli.run(f"--output json deployments get {app}").stdout)
    for tool in tools:
        with subtests.test(tool=tool.name):
            if tool.grant is not None:
                # Deploy adds the grant before returning, but UC can apply it a little later, and
                # a tool called in between is denied inside a successful response.
                waited = workspace_client.wait_until_effective(deployment, tool.grant)
                log(f"{tool.name}: {tool.grant[3]} effective for {app} after {waited:.0f}s")
            output = agentbricks_cli.run(
                f"--output json endpoint invoke {app} --path /api/invocations "
                f"--json {invocation_json(tool.prompt)}",
                timeout=360,
                retry_server_errors=True,
            ).stdout
            tool.check(output)

    agentbricks_cli.run(f"deployments delete {app} --yes", timeout=1200)
    # `deployments delete` returns while the App is still deleting.
    wait_until_app_gone(agentbricks_cli, app)
    assert not_found(agentbricks_cli.run(f"--output json deployments get {app}", check=False))
