"""`agentbricks init --existing`: the migration skill and its pointer come from the installed package.

The other suites scaffold new projects; `--existing` copies its own bundled resources instead, so
this catches them missing from the built wheel.
"""

from __future__ import annotations

import pathlib

from harness.agentbricks_cli import AgentbricksCli


def test_init_existing(agentbricks_cli: AgentbricksCli, tmp_path: pathlib.Path) -> None:
    project = tmp_path / "existing-agent"
    project.mkdir()

    agentbricks_cli.run(f"init --existing --framework langgraph {project}", timeout=120)

    skill = project / "agent-bricks-migrate/SKILL.md"
    assert "name: agent-bricks-migrate" in skill.read_text()
    pointer = project / ".claude/skills/agent-bricks-migrate/SKILL.md"
    assert "../../../agent-bricks-migrate/SKILL.md" in pointer.read_text()
