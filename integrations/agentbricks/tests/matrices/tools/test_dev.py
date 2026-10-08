"""Each tool works in `agentbricks dev`, per authoring path.

CLI authoring is incremental: add one tool, restart dev, verify it, then add the next on the same
project. Direct authoring writes every tool into agent.toml and runs dev once. A subtest per tool
reports each tool independently and the sequence continues past a failure.
"""

from __future__ import annotations

import pytest
from agentbricks_cli import AgentbricksCli
from tools import (
    Tool,
    assert_endpoint_invoke,
    bind,
    invoke_and_check,
    reject_unavailable_mcp,
)

# Local dev servers share run-local's proxy port, so all dev runs stay on one worker, in order.
pytestmark = pytest.mark.xdist_group("dev")


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
                with agentbricks_cli.dev(project) as agent:
                    invoke_and_check(agent, tool)
        with subtests.test(step="endpoint-invoke"), agentbricks_cli.dev(project) as agent:
            assert_endpoint_invoke(agentbricks_cli, agent)
        return

    agentbricks_cli.write_manifest(project, tools)
    with agentbricks_cli.dev(project) as agent:
        for tool in tools:
            with subtests.test(tool=tool.name):
                invoke_and_check(agent, tool)
        with subtests.test(step="endpoint-invoke"):
            assert_endpoint_invoke(agentbricks_cli, agent)
