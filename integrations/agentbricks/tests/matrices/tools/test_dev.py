"""Each tool works in `agentbricks dev`, per authoring path.

CLI authoring is incremental: add one tool, restart dev, verify it, then add the next on the same
project. Direct authoring writes every tool into agent.toml and runs dev once. A subtest per tool
reports each tool independently and the sequence continues past a failure.
"""

from __future__ import annotations

import contextlib

from agentbricks_cli import AgentbricksCli
from common import Evidence, Inputs
from tools import (
    bind,
    invoke_and_check,
    reject_unavailable_mcp,
    setup_failures,
    tool_row,
    tools_for,
)


def test_dev(
    agentbricks_cli: AgentbricksCli, inputs: Inputs, evidence: Evidence, authoring: str, subtests
) -> None:
    tools = tools_for(inputs)
    project = agentbricks_cli.new_project(authoring)

    if authoring == "cli":
        with subtests.test(step="rejects-unavailable-mcp"):
            reject_unavailable_mcp(agentbricks_cli, project)
        for tool in tools:
            with subtests.test(tool=tool.name), tool_row(evidence, authoring, "dev", tool) as row:
                bind(agentbricks_cli, project, tool)
                with agentbricks_cli.dev(project) as agent:
                    invoke_and_check(row, agent, tool)
        return

    with contextlib.ExitStack() as stack:
        with setup_failures(evidence, authoring, "dev", tools):
            agentbricks_cli.write_manifest(project, tools)
            agent = stack.enter_context(agentbricks_cli.dev(project))
        for tool in tools:
            with subtests.test(tool=tool.name), tool_row(evidence, authoring, "dev", tool) as row:
                invoke_and_check(row, agent, tool)
