"""Discover and explicitly adopt the bundled workflow skill without network access."""

from __future__ import annotations

from pathlib import Path

import click

from databricks_agentbricks.presentation import render
from databricks_agentbricks.skills import (
    WORKFLOW_SKILL_NAME,
    bundled_workflow_skill,
    install_workflow_skill,
)


@click.group()
def skills() -> None:
    """Read or install coding-agent guidance for Agent Bricks CLI workflows.

    Skills ship with this CLI version. No authentication or network access is required.
    Installation is project-local and never changes global coding-agent configuration.
    """


@skills.command("show")
@click.pass_obj
def show(obj) -> None:
    """Print the bundled workflow skill and its location.

    Any coding agent can read this guidance, even without native skill discovery.
    Resolve references relative to the printed skill path.
    """
    manifest = bundled_workflow_skill()
    content = manifest.read_text(encoding="utf-8")
    if obj.output == "json":
        render.emit_json({"name": WORKFLOW_SKILL_NAME, "path": str(manifest), "content": content})
        return
    click.echo(f"Skill: {manifest}\n\n{content}", nl=False)


@skills.command("install")
@click.argument(
    "directory",
    default=".",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.pass_obj
def install(obj, directory: Path) -> None:
    """Install the workflow skill in DIRECTORY (default: current directory).

    Write the shared bundle under .agents/skills and pointers under .claude/skills and
    .agent/skills. Identical installations are left unchanged; different existing skills
    are never overwritten. Restart your coding-agent session if it caches skill discovery.
    """
    manifests = install_workflow_skill(directory)
    if obj.output == "json":
        render.emit_json({"name": WORKFLOW_SKILL_NAME, "skills": [str(path) for path in manifests]})
        return
    render.success(
        "Installed Agent Bricks workflow guidance",
        fields={"Skill": str(manifests[0])},
        next_steps=["Restart your coding-agent session if it caches skill discovery"],
    )
