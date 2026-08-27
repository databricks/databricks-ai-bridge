"""`mason dev` — run a scaffolded agent locally, wrapping `databricks apps run-local`.

Runs the app from its ``app.yaml`` exactly as the Databricks Apps runtime would locally: reads the
manifest's command + env, and (with ``--prepare-environment``) builds the venv via uv. This is the
local counterpart to ``mason deploy`` — same source dir, same manifest — so what runs here matches
what ships. Delegating to ``apps run-local`` means mason inherits the Apps team's local-run behavior
rather than re-implementing it.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import click

from databricks_mason.deploy import _databricks
from databricks_mason.errors import AgentCliError


@click.command()
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Local source directory to run (containing app.yaml). Defaults to the current directory.",
)
@click.option(
    "--prepare-environment/--no-prepare-environment",
    default=None,
    help="Build the app's environment with uv before running. Default: build only if no .venv "
    "exists yet, and reuse it otherwise. Requires uv.",
)
@click.option("--app-port", type=int, default=None, help="Port to run the app on (default 8000).")
@click.pass_obj
def dev(obj, source: str, prepare_environment: Optional[bool], app_port: Optional[int]) -> None:
    """Run a scaffolded agent locally from its app.yaml (wraps `databricks apps run-local`).

    Reads the app's command + env from ``app.yaml`` and runs it the way the Apps runtime does — so
    local behavior matches a deployment. Auth uses the profile (``-p`` / ``mason login``), same as
    ``mason deploy``. The environment is built on first run and reused after; pass
    ``--prepare-environment`` to force a rebuild (e.g. after changing dependencies).
    """
    source_dir = pathlib.Path(source)
    if not (source_dir / "app.yaml").exists():
        raise AgentCliError(
            f"No app.yaml in '{source_dir}'.",
            hint="Run from a scaffolded project, or pass --source <dir> (see `mason init`).",
        )

    # Default: prepare only when there's no venv yet, so repeat runs don't rebuild. Explicit
    # --prepare-environment / --no-prepare-environment overrides the auto-detect.
    if prepare_environment is None:
        prepare_environment = not (source_dir / ".venv").exists()

    args = ["apps", "run-local"]
    if prepare_environment:
        args.append("--prepare-environment")
    if app_port is not None:
        args += ["--app-port", str(app_port)]
    # Run in the project dir so run-local finds app.yaml; stream output (no capture).
    _databricks(args, obj.profile, cwd=str(source_dir))
