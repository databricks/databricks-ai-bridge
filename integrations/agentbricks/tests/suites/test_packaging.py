"""The installed `agentbricks` package: what only breaks once it is built, installed and run.

CI runs the suites from a venv holding the built wheels, so these catch a broken console-script entry
point, package data missing from the wheel, or an import that fails only when installed. They need no
workspace. Detailed behavior of each command is the unit tests' job.
"""

from __future__ import annotations

import pathlib
from importlib.metadata import requires

import pytest
import tomli
from harness.agentbricks_cli import AgentbricksCli
from packaging.requirements import Requirement

from databricks_agentbricks.cli.app import agentbricks


def test_cli_and_every_command_help_load(agentbricks_cli: AgentbricksCli) -> None:
    assert "Usage" in agentbricks_cli.run("--help").stdout
    for command in sorted(agentbricks.commands):
        agentbricks_cli.run(f"{command} --help")


def test_installed_wheel_exposes_agentkit_sdk() -> None:
    from databricks_agentkit import AgentKitClient, MemoryStore, SessionStore

    assert AgentKitClient.__module__ == "databricks_agentkit.client"
    assert MemoryStore.__module__ == "databricks_agentkit.memory_store"
    assert SessionStore.__module__ == "databricks_agentkit.session_store"


def test_installed_langgraph_extra_pins_fixed_provider() -> None:
    requirements = [Requirement(dep) for dep in requires("databricks-agentbricks") or []]
    provider = next(dep for dep in requirements if dep.name == "databricks-langchain")
    assert str(provider.specifier) == "==0.21.0"
    assert provider.marker is not None
    assert provider.marker.evaluate({"extra": "langgraph"})
    assert not provider.marker.evaluate({"extra": "openai"})
    assert not provider.marker.evaluate({"extra": ""})


@pytest.mark.parametrize(
    "flags",
    [
        ["--framework", "langgraph"],
        ["--framework", "openai"],
        ["--framework", "langgraph", "--server", "custom"],
    ],
    ids=["langgraph", "openai", "langgraph-custom"],
)
def test_init_scaffolds_from_the_installed_templates(
    agentbricks_cli: AgentbricksCli, tmp_path: pathlib.Path, flags: list[str]
) -> None:
    project = tmp_path / "agent"
    agentbricks_cli.run(f"init {' '.join(flags)} {project}", timeout=120)
    for name in ("pyproject.toml", "app.yaml", "agent.toml"):
        assert (project / name).is_file(), name
    dependencies = tomli.loads((project / "pyproject.toml").read_text())["project"]["dependencies"]
    provider = [Requirement(dep) for dep in dependencies if "databricks-langchain" in dep]
    if "langgraph" in flags:
        assert [str(dep.specifier) for dep in provider] == ["==0.21.0"]
    else:
        assert not provider


def test_existing_init_includes_migration_skill_from_wheel(
    agentbricks_cli: AgentbricksCli, tmp_path: pathlib.Path
) -> None:
    project = tmp_path / "existing-agent"
    project.mkdir()

    agentbricks_cli.run(f"init --existing --framework langgraph {project}", timeout=120)

    skill = project / "agent-bricks-migrate/SKILL.md"
    assert "name: agent-bricks-migrate" in skill.read_text()
    pointer = project / ".claude/skills/agent-bricks-migrate/SKILL.md"
    assert "../../../agent-bricks-migrate/SKILL.md" in pointer.read_text()
