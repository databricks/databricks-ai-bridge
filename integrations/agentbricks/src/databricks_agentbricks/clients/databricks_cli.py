"""Thin wrapper around the ``databricks`` CLI.

Agent Bricks shells out to the ``databricks`` CLI for the operations it doesn't call over the REST client
(``apps`` lifecycle, ``sync``, ``postgres`` endpoint/credential lookups, ``auth login``). This is the single place
that builds the argv, threads the ``--profile`` through, and normalizes a non-zero exit into an
``AgentCliError``. Kept separate from any one caller so ``deploy``, ``dev``, and ``app_resources``
share it without importing each other.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from typing import Optional

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.profiles import auth_login_command


def _databricks(
    args: list[str],
    profile: Optional[str],
    *,
    capture: bool = False,
    to_stderr: bool = False,
    check: bool = True,
    cwd: Optional[str] = None,
    action: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    cmd = ["databricks", *args]
    if profile:
        cmd += ["--profile", profile]
    # `to_stderr` shows an uncaptured command's output as progress while keeping stdout for Agent
    # Bricks' own `--output json` payload. Fd 2, not sys.stderr: a replaced sys.stderr has no fileno.
    stdout = 2 if to_stderr and not capture else None
    result = subprocess.run(cmd, text=True, capture_output=capture, stdout=stdout, cwd=cwd, env=env)
    if check and result.returncode != 0:
        # Agent Bricks drives the `databricks apps` CLI as an implementation detail; surface a failure in
        # Agent Bricks' own terms (`action`) rather than echoing the raw subcommand and --profile, which
        # leaks the underlying tool at the customer. The captured stderr still rides along as the
        # hint so debugging isn't lost.
        detail = (result.stderr or result.stdout or "").strip() if capture else None
        raise AgentCliError(action or "A Databricks CLI command failed.", hint=detail)
    return result


def run_databricks_login(profile: Optional[str]) -> None:
    """Run `databricks auth login`; with no profile it shows its own interactive profile picker."""
    command = ["databricks", "auth", "login", *(["--profile", profile] if profile else [])]
    retry = auth_login_command(profile) if profile else "agentbricks auth login"
    try:
        # Keep the child process interactive while preserving stdout for Agent Bricks JSON output.
        result = subprocess.run(command, text=True, check=False, stdout=sys.stderr)
    except FileNotFoundError as exc:
        raise AgentCliError(
            "Could not configure Databricks authentication: the `databricks` CLI was not found.",
            hint=f"Install the Databricks CLI, then retry `{retry}`.",
        ) from exc
    if result.returncode != 0:
        raise AgentCliError(f"`{' '.join(command)}` failed (exit {result.returncode}).")
