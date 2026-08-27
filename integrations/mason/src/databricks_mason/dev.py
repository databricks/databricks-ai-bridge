"""`mason dev` — run a scaffolded agent locally."""

from __future__ import annotations

import os
import pathlib
import subprocess

import click

from databricks_mason.errors import AgentCliError


@click.command()
@click.option(
    "--source",
    default=".",
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="Agent project directory.",
)
@click.option("--port", type=int, default=None, help="Override the local server port.")
@click.pass_obj
def dev(obj, source: str, port: int | None) -> None:
    """Run the current agent locally with its `start-server` script."""
    source_dir = pathlib.Path(source).resolve()
    if not (source_dir / "pyproject.toml").exists():
        raise AgentCliError(
            f"No pyproject.toml found in '{source_dir}'.",
            hint="Run this inside a scaffolded agent project or pass --source.",
        )

    env = os.environ.copy()
    if obj.profile and "DATABRICKS_CONFIG_PROFILE" not in env:
        env["DATABRICKS_CONFIG_PROFILE"] = obj.profile
    if port is not None:
        env["PORT"] = str(port)

    try:
        result = subprocess.run(["uv", "run", "start-server"], cwd=source_dir, env=env)
    except FileNotFoundError as exc:
        raise AgentCliError("`uv` is required to run the agent locally.") from exc
    if result.returncode != 0:
        raise AgentCliError(f"Local agent exited with status {result.returncode}.")
