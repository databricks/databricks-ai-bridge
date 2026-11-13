"""Prepare a project's dependency lock before publishing sandbox source."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from databricks_agentbricks.errors import AgentCliError


def ensure_lockfile(source: Path, *, note: Callable[[str], None] | None = None) -> None:
    """Create a missing lock with the sandbox Python version; preserve existing pins."""
    source = source.resolve()
    if not (source / "pyproject.toml").is_file():
        raise AgentCliError("Agent source must include pyproject.toml.")
    lock = source / "uv.lock"
    if lock.is_file():
        return
    uv = shutil.which("uv")
    if uv is None:
        raise AgentCliError(
            "Creating uv.lock requires uv.",
            hint="Install uv, then rerun agentbricks deploy --sandbox.",
        )
    command = [uv, "lock", "--python", "3.12", "--project", str(source)]
    retry = shlex.join(["uv", *command[1:]])
    if note:
        note(f"Generating uv.lock in {source} (resolving dependencies; no local installation)...")
    try:
        result = subprocess.run(
            command,
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if note and isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            note(exc.stderr.strip())
        raise AgentCliError(
            "Could not generate uv.lock; source was not uploaded.",
            hint=f"Run {retry} to check dependency and index configuration, then retry deployment.",
        ) from exc
    if note and result.stderr:
        note(result.stderr.strip())
    if not lock.is_file():
        raise AgentCliError(
            "uv did not create uv.lock in the selected source folder.",
            hint="Deploy a standalone project with its own pyproject.toml and uv.lock.",
        )
    if note:
        note("Created uv.lock. Keep it with your source for reproducible deployments.")
