"""Preflight checks for ``agentbricks dev``: uv on PATH, and a Python uv can provide.

Scaffolded projects run ``uv run start-server`` (``app.yaml``), and run-local builds the venv with
uv. On a machine without uv, or where uv has no interpreter matching the project's
``requires-python`` and cannot fetch one, that otherwise only surfaces as run-local's generic
"Could not start the agent locally." long after the run began. Both checks run before anything
starts and name the exact fix, and both are fail-open: a check that cannot complete for an
unrelated reason (no pyproject, an invalid spec, a uv probe that could not run) never blocks a
setup that might have worked.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Callable

import tomli
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from databricks_agentbricks.errors import AgentCliError

# Only the executable token decides whether this run uses uv; `uv` appearing later in the command
# does not trigger the checks. `uv.exe` covers the Windows spelling (`which` resolves it through
# PATHEXT there).
_UV_COMMAND_NAMES = frozenset({"uv", "uv.exe"})

# uv options that make the app command use some other project than source_dir's (or none), or pick
# the interpreter explicitly. The preflight would then check the wrong requires-python or miss an
# interpreter outside uv's discovery, so it skips the Python check instead of reproducing uv's
# resolution.
_PROJECT_OVERRIDE_OPTIONS = (
    "--no-project",
    "--project",
    "--directory",
    "--script",
    "-s",
    "--python",
    "-p",
)

# uv options known to take no value, so the positional after one is the command uv runs. After any
# other option the next positional may be that option's value; scanning then continues, which can
# only skip the Python check (fail-open), never wrongly enforce it.
_UV_FLAGS_WITHOUT_VALUE = frozenset(
    {
        "--active",
        "--all-extras",
        "--all-groups",
        "--all-packages",
        "--exact",
        "--frozen",
        "--isolated",
        "--locked",
        "--native-tls",
        "--no-cache",
        "--no-dev",
        "--no-editable",
        "--no-env-file",
        "--no-sync",
        "--offline",
        "--quiet",
        "--verbose",
        "-q",
        "-v",
    }
)

# `uv python find` is a local probe (it never downloads). The install attempt only runs when the
# probe found nothing and may download an interpreter, so it gets the long leash.
_FIND_TIMEOUT_SECONDS = 15
_INSTALL_TIMEOUT_SECONDS = 180

# The `uv said:` line quotes uv's last stderr line, but only while it stays short enough to read.
_MAX_UV_SAID_CHARS = 200


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
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    os_name: str = os.name,
) -> None:
    """Run both checks for one ``agentbricks dev`` invocation.

    The uv check applies only when this invocation will actually run uv: the app command runs
    under uv, or run-local is asked to build the environment (which also uses uv). A non-uv
    project reusing its ``.venv`` is never checked. The Python check runs once uv is known to be
    present, and only when the project's interpreter is this run's requirement: an environment build
    always resolves it, but an app command that selects another project or none
    (``--no-project``, ``--project``, ``--directory``, ``--script``/``-s``), or picks its
    interpreter (``--python``/``-p``), may not, so it is skipped.
    """
    if not (uses_uv(command_tokens) or prepare_environment):
        return
    uv = ensure_uv_available(
        source_dir,
        command_tokens,
        prepare_environment=prepare_environment,
        which=which,
        os_name=os_name,
    )
    if prepare_environment or not _overrides_project(command_tokens):
        ensure_python_available(source_dir, uv=uv, run=run)


def _overrides_project(command_tokens: list[str]) -> bool:
    """Whether the uv app command selects a project other than source_dir's (or none), or its
    interpreter.

    Only uv's own options count: tokens from the command uv runs onward (``uv run start-server
    --project demo``) belong to that command, so the scan stops there or at ``--``.
    """
    if not uses_uv(command_tokens):
        return False
    seen_subcommand = False
    next_positional_is_command = False
    for token in command_tokens[1:]:
        if token == "--":
            return False
        if token.startswith("-"):
            if any(_matches_option(token, option) for option in _PROJECT_OVERRIDE_OPTIONS):
                return True
            next_positional_is_command = "=" in token or token in _UV_FLAGS_WITHOUT_VALUE
            continue
        if not seen_subcommand and token == "run":
            seen_subcommand = True
            next_positional_is_command = True
            continue
        if seen_subcommand and next_positional_is_command:
            return False
        # Possibly the value of the preceding option; the positional after it is the command.
        next_positional_is_command = True
    return False


def _matches_option(token: str, option: str) -> bool:
    """Whether ``token`` is ``option``, with its value attached by ``=`` or, for a short option
    such as ``-p3.11``, directly."""
    if token == option or token.startswith(f"{option}="):
        return True
    return not option.startswith("--") and token.startswith(option)


def ensure_uv_available(
    source_dir: pathlib.Path,
    command_tokens: list[str],
    *,
    prepare_environment: bool = False,
    which: Callable[[str], str | None] = shutil.which,
    os_name: str = os.name,
) -> str:
    """Return the uv the app command runs, or fail with the fix.

    run-local starts the command with ``source_dir`` as its working directory, so a relative
    path-form executable (``./bin/uv``) is resolved against the project, not the caller's cwd.
    The environment build runs uv from PATH, so it also needs PATH uv when the command names
    another one.
    """
    command_uv = command_tokens[0] if uses_uv(command_tokens) else None
    if command_uv is None or not os.path.dirname(command_uv):
        return _require_path_uv(command_uv or "uv", which=which, os_name=os_name)

    resolved = command_uv if os.path.isabs(command_uv) else str(source_dir / command_uv)
    found = which(resolved)
    if not found:
        raise AgentCliError(
            f"app.yaml runs uv from {command_uv}, but no executable uv is there.",
            hint=(
                f"Restore {command_uv}, or change the app.yaml command to run `uv` from PATH, "
                "then re-run `agentbricks dev`."
            ),
        )
    if prepare_environment:
        _require_path_uv("uv", which=which, os_name=os_name)
    return os.path.abspath(found)


def _require_path_uv(name: str, *, which: Callable[[str], str | None], os_name: str) -> str:
    """Return uv found on PATH as ``name``, or fail with the install command for the OS."""
    found = which(name)
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


def ensure_python_available(
    source_dir: pathlib.Path,
    *,
    uv: str = "uv",
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Check uv can provide an interpreter matching the project's ``requires-python``.

    Reads the spec from ``<source_dir>/pyproject.toml``. uv owns interpreter resolution, so a
    missing file, a missing or invalid spec, or a probe that could not complete all skip the check.
    When the probe finds nothing, try the install on the user's behalf (uv downloads Python on its
    own when allowed) and fail only when that install fails: run-local's uv would fail on the same
    resolution, so there is a real fix to name.
    """
    spec = _requires_python(source_dir)
    if spec is None:
        return
    try:
        SpecifierSet(spec)
    except InvalidSpecifier:
        return

    find = _run_uv(source_dir, uv, ["python", "find", spec], run=run, timeout=_FIND_TIMEOUT_SECONDS)
    if find is None or find.returncode == 0:
        return
    _notify(f'Installing Python "{spec}" with uv (this can take a minute)...')
    install = _run_uv(
        source_dir, uv, ["python", "install", spec], run=run, timeout=_INSTALL_TIMEOUT_SECONDS
    )
    if install is None or install.returncode == 0:
        return
    said = _last_short_line(install.stderr)
    uv_said = f"\nuv said: {said}" if said else ""
    raise AgentCliError(
        f'uv couldn\'t get Python "{spec}" (required by pyproject.toml).',
        hint=f'Run `uv python install "{spec}"`, then re-run `agentbricks dev`.{uv_said}',
    )


def _requires_python(source_dir: pathlib.Path) -> str | None:
    """The project's ``requires-python``, or None when absent, unreadable, or not a string."""
    try:
        data = tomli.loads((source_dir / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomli.TOMLDecodeError):
        return None
    project = data.get("project")
    spec = project.get("requires-python") if isinstance(project, dict) else None
    if not isinstance(spec, str) or not spec.strip():
        return None
    return spec.strip()


def _run_uv(
    source_dir: pathlib.Path,
    uv: str,
    args: list[str],
    *,
    run: Callable[..., subprocess.CompletedProcess],
    timeout: float,
) -> subprocess.CompletedProcess | None:
    """Run one uv subcommand in the project dir; None means the probe could not complete."""
    try:
        return run(
            [uv, *args],
            cwd=source_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # uv writes UTF-8; the locale's codec (a legacy Windows code page) could fail to
            # decode a non-ASCII path and abort dev instead of failing open.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _last_short_line(output: str | None) -> str | None:
    """The last non-empty line of uv's output, or None when it is empty or too long to quote."""
    if not output:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1] if len(lines[-1]) <= _MAX_UV_SAID_CHARS else None


def _notify(message: str) -> None:
    """One progress line on stderr, so a slow uv download never looks like a hang."""
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()
