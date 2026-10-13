"""Bundled workflow guidance and project-local coding-agent discovery."""

from __future__ import annotations

import os
import shutil
import tempfile
from importlib import resources
from pathlib import Path

from databricks_agentbricks.errors import AgentCliError

WORKFLOW_SKILL_NAME = "agent-bricks-workflow"
SKILL_DISCOVERY_ROOTS = (".agents", ".claude", ".agent")


def bundled_workflow_skill() -> Path:
    """Return the version-matched skill shipped with the installed CLI."""
    manifest = (
        resources.files("databricks_agentbricks")
        .joinpath("templates")
        .joinpath(WORKFLOW_SKILL_NAME)
        .joinpath("SKILL.md")
    )
    if not manifest.is_file():
        raise AgentCliError("The installed Agent Bricks CLI does not bundle the workflow skill.")
    return Path(str(manifest))


def _skill_files(directory: Path) -> dict[Path, bytes]:
    files = {}
    for entry in directory.rglob("*"):
        if entry.is_symlink():
            raise AgentCliError(f"Cannot write skills through symbolic link '{entry}'.")
        if entry.is_file():
            files[entry.relative_to(directory)] = entry.read_bytes()
    return files


def _validate_destination(project: Path, destination: Path, source: Path) -> None:
    relative = destination.relative_to(project)
    for index in range(1, len(relative.parts) + 1):
        ancestor = project.joinpath(*relative.parts[:index])
        if ancestor.is_symlink() or (ancestor.exists() and not ancestor.is_dir()):
            raise AgentCliError(f"Cannot write skills at '{ancestor}'.")
    if destination.exists() and _skill_files(destination) != _skill_files(source):
        raise AgentCliError(
            f"Skill files already exist at '{destination}'.",
            hint="Move that skill directory before installing a different version.",
        )


def install_workflow_skill(project: Path) -> tuple[Path, ...]:
    """Install one shared bundle and discovery pointers without overwriting user files."""
    if not project.is_dir():
        raise AgentCliError(f"Project directory '{project}' was not found.")
    source = bundled_workflow_skill()
    destinations = tuple(
        project / root / "skills" / WORKFLOW_SKILL_NAME for root in SKILL_DISCOVERY_ROOTS
    )
    with tempfile.TemporaryDirectory(prefix="agent-bricks-workflow-") as temporary:
        staging = Path(temporary)
        bundle = staging / ".agents"
        shutil.copytree(source.parent, bundle, ignore=shutil.ignore_patterns("__pycache__"))
        _, frontmatter, _ = source.read_text(encoding="utf-8").split("---", 2)
        staged = [bundle]
        for root in SKILL_DISCOVERY_ROOTS[1:]:
            pointer = staging / root
            pointer.mkdir()
            target = Path(
                os.path.relpath(
                    destinations[0] / "SKILL.md", project / root / "skills" / WORKFLOW_SKILL_NAME
                )
            )
            (pointer / "SKILL.md").write_text(
                f"---{frontmatter}---\n\n"
                f"Read [{target.as_posix()}]({target.as_posix()}) for the Agent Bricks workflow. "
                "Resolve its references relative to that shared skill directory.\n",
                encoding="utf-8",
            )
            staged.append(pointer)

        for destination, prepared in zip(destinations, staged, strict=True):
            _validate_destination(project, destination, prepared)

        created: list[Path] = []
        try:
            for destination, prepared in zip(destinations, staged, strict=True):
                if destination.exists():
                    continue
                outermost = destination
                while outermost.parent != project and not outermost.parent.exists():
                    outermost = outermost.parent
                created.append(outermost)
                shutil.copytree(prepared, destination)
        except Exception:
            for directory in reversed(created):
                shutil.rmtree(directory, ignore_errors=True)
            raise
    return tuple(destination / "SKILL.md" for destination in destinations)
