"""`agentbricks tools add` validates what it adds before it touches agent.toml."""

from __future__ import annotations

import json

import pytest
from harness.agentbricks_cli import AgentbricksCli
from harness.project import Project


@pytest.mark.usefixtures("workspace_client")
def test_rejects_unavailable_mcp(agentbricks_cli: AgentbricksCli, project: Project) -> None:
    """An MCP service that does not exist is refused, and agent.toml is left untouched."""
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    before = (project.path / "agent.toml").read_bytes()

    rejected = agentbricks_cli.run(
        "--output json tools add mcp system.ai.missing_service --name broken_mcp "
        f"--source {project.path}",
        check=False,
    )

    assert rejected.returncode != 0, "tools add accepted an unavailable MCP service"
    assert (project.path / "agent.toml").read_bytes() == before, (
        "a refused tools add changed agent.toml"
    )
    code = json.loads(rejected.stderr).get("error", {}).get("code")
    assert code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}, rejected.stderr

    agentbricks_cli.run(f"tools remove mcp system.ai.missing_service --source {project.path}")
    declared = [tool["id"] for tool in project.manifest().get("tools", [])]
    assert "broken_mcp" not in declared, "tools remove left the broken MCP binding in agent.toml"
