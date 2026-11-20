"""Build a project's environment with ``uv sync`` for ``agentbricks dev``."""

from __future__ import annotations

import os
import pathlib
import subprocess

from databricks_agentbricks.errors import AgentCliError


def uv_sync(project_dir: pathlib.Path) -> None:
    """Install exactly the project's declared dependencies into its ``.venv``, streaming uv's output.

    ``VIRTUAL_ENV`` is dropped so an activated venv elsewhere never becomes the install target.
    """
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    try:
        result = subprocess.run(["uv", "sync"], cwd=project_dir, env=env, check=False)
    except FileNotFoundError as exc:
        raise AgentCliError(
            "Could not find `uv`.", hint="Install it: https://docs.astral.sh/uv/"
        ) from exc
    if result.returncode != 0:
        raise AgentCliError(
            "Could not install the agent's dependencies.", hint="See the uv output above."
        )
