"""Each tool works in `agentbricks dev`, per authoring path.

CLI authoring is incremental: add one tool, restart dev, verify it, then add the next on the same
project. Direct authoring writes every tool into agent.toml and runs dev once. A subtest per tool
reports each tool independently and the sequence continues past a failure.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
from harness.cli import AgentbricksCli, AgentEndpoint, Project, free_port
from tool_definitions import (
    Tool,
    assert_endpoint_invoke,
    bind,
    invoke_and_check,
    reject_unavailable_mcp,
)

# Local dev servers share run-local's proxy port, so all dev runs stay on one worker, in order.
pytestmark = pytest.mark.xdist_group("dev")


@contextlib.contextmanager
def dev(cli: AgentbricksCli, project: Project) -> Iterator[AgentEndpoint]:
    """`agentbricks dev` the project for the duration of the block."""
    port = free_port()
    # The template server binds $PORT, so an inherited PORT would override --app-port.
    with cli.start(
        f"dev --source {project.path} --app-port {port}", env={"PORT": str(port)}
    ) as server:
        yield AgentEndpoint.local(server, port)


def test_dev(
    agentbricks_cli: AgentbricksCli, tools: tuple[Tool, ...], authoring: str, subtests
) -> None:
    project = agentbricks_cli.new_project(authoring)

    if authoring == "cli":
        with subtests.test(step="rejects-unavailable-mcp"):
            reject_unavailable_mcp(agentbricks_cli, project)
        for tool in tools:
            with subtests.test(tool=tool.name):
                bind(agentbricks_cli, project, tool)
                with dev(agentbricks_cli, project) as endpoint:
                    invoke_and_check(endpoint, tool)
        with subtests.test(step="endpoint-invoke"), dev(agentbricks_cli, project) as endpoint:
            assert_endpoint_invoke(agentbricks_cli, endpoint)
        return

    project.write_manifest(tools)
    with dev(agentbricks_cli, project) as endpoint:
        for tool in tools:
            with subtests.test(tool=tool.name):
                invoke_and_check(endpoint, tool)
        with subtests.test(step="endpoint-invoke"):
            assert_endpoint_invoke(agentbricks_cli, endpoint)
