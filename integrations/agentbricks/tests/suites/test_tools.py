"""`agentbricks tools`: `tools add` declares each tool in agent.toml, and `tools remove` takes it out."""

from __future__ import annotations

from harness.agentbricks_cli import AgentbricksCli
from harness.project import Project
from tool_definitions import Tool, assert_declared


def test_tools_add_and_remove(
    agentbricks_cli: AgentbricksCli, project: Project, tools: tuple[Tool, ...]
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    added = [tool for tool in tools if tool.add_command and tool.tool_id]
    for tool in added:
        agentbricks_cli.run(f"tools add {tool.add_command} --source {project.path}")
        assert_declared(project, tool)

    for tool in added:
        agentbricks_cli.run(f"tools remove {tool.tool_id} --source {project.path}")
    declared = [t["id"] for t in project.manifest().get("tools", [])]
    assert not declared, f"tools remove left {declared} in agent.toml"
