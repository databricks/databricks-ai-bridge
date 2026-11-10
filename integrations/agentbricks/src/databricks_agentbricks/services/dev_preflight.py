"""Preflight check for ``agentbricks dev``: uv on PATH.

Scaffolded projects run ``uv run start-server`` (``app.yaml``), and run-local builds the venv with
uv. On a machine without uv, that otherwise only surfaces as run-local's generic "Could not start
the agent locally." long after the run began. The check runs before anything starts and names the
exact fix.
"""

from __future__ import annotations

import os
import pathlib
import shutil
from collections.abc import Callable

from databricks_agentbricks.errors import AgentCliError

# Only the executable token decides whether this run uses uv; `uv` appearing later in the command
# does not trigger the check. `uv.exe` covers the Windows spelling (`which` resolves it through
# PATHEXT there).
_UV_COMMAND_NAMES = frozenset({"uv", "uv.exe"})


def uses_uv(command_tokens: list[str]) -> bool:
    """Whether the app command's executable is uv."""
    first = command_tokens[0] if command_tokens else ""
    return os.path.basename(first).casefold() in _UV_COMMAND_NAMES


def run_dev_preflight(
    source_dir: pathlib.Path,
    command_tokens: list[str],
    prepare_environment: bool,
    *,
    which: Callable[[str], str | None] = shutil.which,
    os_name: str = os.name,
) -> None:
    """Run the preflight for one ``agentbricks dev`` invocation.

    The uv check applies only when this invocation will actually run uv: the app command runs
    under uv, or run-local is asked to build the environment (which also uses uv). A non-uv
    project reusing its ``.venv`` is never checked.
    """
    if not (uses_uv(command_tokens) or prepare_environment):
        return
    ensure_uv_available(source_dir, command_tokens, which=which, os_name=os_name)


def ensure_uv_available(
    source_dir: pathlib.Path,
    command_tokens: list[str],
    *,
    which: Callable[[str], str | None] = shutil.which,
    os_name: str = os.name,
) -> str:
    """Return the uv this run needs, or fail with the install command for the OS.

    run-local starts the command with ``source_dir`` as its working directory, so a relative
    path-form executable (``./bin/uv``) is resolved against the project, not the caller's cwd.
    """
    target = command_tokens[0] if uses_uv(command_tokens) else "uv"
    if os.path.dirname(target) and not os.path.isabs(target):
        target = str(source_dir / target)
    found = which(target)
    if found:
        return os.path.abspath(found)
    install = (
        'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
        if os_name == "nt"
        else "curl -LsSf https://astral.sh/uv/install.sh | sh"
    )
    raise AgentCliError(
        "uv is required to run this agent locally, but uv was not found on PATH.",
        hint=(
            f"Install it: {install}\n"
            "Then run `uv --version` to verify, and re-run `agentbricks dev`."
        ),
    )
