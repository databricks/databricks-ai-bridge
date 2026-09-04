"""`mason durability` — declare durable invocation storage for an agent project."""

from __future__ import annotations

import pathlib

import click

from databricks_mason import render
from databricks_mason.agent_project import AgentProject


@click.group()
def durability() -> None:
    """Configure durable invocation handling for an agent project."""


def _source_option(function):
    return click.option(
        "--source",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        default=pathlib.Path("."),
        show_default=True,
        help="Mason agent project containing agent.toml.",
    )(function)


@durability.command("bind")
@_source_option
@click.pass_obj
def durability_bind(obj, source: pathlib.Path) -> None:
    """Enable durable invocation storage for this agent project."""
    project = AgentProject.load(source)
    if project.bind_durability():
        project.write()
    if obj.output == "json":
        render.emit_json({"durability": True, "manifest": str(project.path)})
        return
    render.success(
        "Enabled durable invocation storage",
        fields={"agent.toml": str(project.path)},
        next_steps=["mason deploy <name>"],
    )


@durability.command("unbind")
@_source_option
@click.pass_obj
def durability_unbind(obj, source: pathlib.Path) -> None:
    """Disable durable invocation storage for this agent project."""
    project = AgentProject.load(source)
    if project.unbind_durability():
        project.write()
        if obj.output == "json":
            render.emit_json({"durability": False, "manifest": str(project.path)})
            return
        render.success(
            "Disabled durable invocation storage",
            fields={"agent.toml": str(project.path)},
        )
        return
    if obj.output == "json":
        render.emit_json({"durability": False, "manifest": str(project.path)})
        return
    click.echo(f"Durability is not bound in {project.path}.")
