"""Each tool works on a deployed App, and its UC grant is applied, per authoring path.

CLI authoring is incremental: add one tool, redeploy the same App, verify it, then add the next.
After every redeploy the grants of all tools added so far must still hold, which is the idempotency
check. Direct authoring writes every tool into agent.toml and deploys once. A subtest per tool
reports each tool independently and the sequence continues past a failure.

`agentbricks deploy` returns once the App is running its new source, so the next step can use it.
"""

from __future__ import annotations

import json
from typing import Any

from harness.agentbricks_cli import AgentbricksCli
from harness.cleanup import Cleanup
from harness.logs import log
from harness.project import Project
from harness.workspace import Workspace, granted, granted_tuples
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

    def assert_granted(tool: Tool, deployment: dict[str, Any]) -> None:
        if tool.grant is not None:
            assert granted(deployment, tool.grant), (
                f"{tool.name}: {tool.grant} not granted to {app}: "
                f"{sorted(granted_tuples(deployment))}"
            )

    def prepare_and_check(tool: Tool, deployment: dict[str, Any]) -> None:
        if tool.grant is not None:
            waited = workspace_client.wait_until_effective(deployment, tool.grant)
            log(f"{tool.name}: {tool.grant[3]} effective for {app} after {waited:.0f}s")
        if tool.transitive_function:
            # The nested function is deliberately outside Agent Bricks' grants; the tool only
            # works once this manual grant is in place.
            workspace_client.grant_transitive(deployment, tool.transitive_function)
            cleanup.call(workspace_client.revoke_catalog_grant, app)
        output = agentbricks_cli.run(
            f"--output json endpoint invoke {app} --path /api/invocations "
            f"--json {invocation_json(tool.prompt)}",
            timeout=360,
            retry_server_errors=True,
        ).stdout
        tool.check(output)

    if authoring == "cli":
        for index, tool in enumerate(tools):
            with subtests.test(tool=tool.name):
                if tool.add_command:
                    agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
                project.write_files(tool.files)
                assert_declared(project, tool)
                agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
                deployment = json.loads(
                    agentbricks_cli.run(f"--output json deployments get {app}").stdout
                )
                prepare_and_check(tool, deployment)
                for added in tools[: index + 1]:
                    assert_granted(added, deployment)
    elif authoring == "direct":
        project.write_manifest(tools)
        agentbricks_cli.run(f"deploy {app} --source {project.path}", timeout=2400)
        deployment = json.loads(agentbricks_cli.run(f"--output json deployments get {app}").stdout)
        for tool in tools:
            with subtests.test(tool=tool.name):
                prepare_and_check(tool, deployment)
                assert_granted(tool, deployment)
