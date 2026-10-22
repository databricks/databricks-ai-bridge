"""Thin wrapper around the ``databricks`` CLI.

Agent Bricks shells out to the ``databricks`` CLI for the operations it doesn't call over the REST client
(``apps`` lifecycle, ``sync``, ``postgres`` endpoint/credential lookups, ``auth login``). This is the single place
that builds the argv, threads the ``--profile`` through, and normalizes a non-zero exit into an
``AgentCliError``. Kept separate from any one caller so ``deploy``, ``dev``, and ``app_resources``
share it without importing each other.
"""

from __future__ import annotations

import collections
import subprocess
import sys
from collections.abc import Mapping
from typing import Optional

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.profiles import auth_login_command

# Lines of relayed output a `tee` run keeps for callers that inspect a failure.
_TEE_TAIL_LINES = 200


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
    tee: bool = False,
) -> subprocess.CompletedProcess:
    """Run one ``databricks`` command.

    ``capture`` collects stdout/stderr instead of showing them. ``tee`` relays the command's merged
    output live to *our* stderr, line by line, and also returns its tail as ``stdout``: the user
    still sees a long ``sync``/``apps deploy`` as it runs, stdout stays clean for ``-o json``, and the
    caller can still match the failure text (for example, a busy App).
    """
    cmd = ["databricks", *args]
    if profile:
        cmd += ["--profile", profile]
    if tee:
        result = _run_tee(cmd, cwd=cwd, env=env)
    else:
        # `to_stderr` shows an uncaptured command's output as progress while keeping stdout for
        # Agent Bricks' own `--output json` payload. Fd 2, not sys.stderr: a replaced sys.stderr has
        # no fileno.
        stdout = 2 if to_stderr and not capture else None
        result = subprocess.run(
            cmd, text=True, capture_output=capture, stdout=stdout, cwd=cwd, env=env
        )
    if check and result.returncode != 0:
        # Agent Bricks drives the `databricks apps` CLI as an implementation detail; surface a failure in
        # Agent Bricks' own terms (`action`) rather than echoing the raw subcommand and --profile, which
        # leaks the underlying tool at the customer. The captured stderr still rides along as the
        # hint so debugging isn't lost.
        detail = (result.stderr or result.stdout or "").strip() if capture else None
        raise AgentCliError(action or "A Databricks CLI command failed.", hint=detail)
    return result


def _run_tee(
    cmd: list[str], *, cwd: Optional[str], env: Optional[Mapping[str, str]]
) -> subprocess.CompletedProcess:
    """Run ``cmd`` relaying its merged output to stderr; return its tail as ``stdout``."""
    tail: collections.deque[str] = collections.deque(maxlen=_TEE_TAIL_LINES)
    with subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=env,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stderr.write(line)
            sys.stderr.flush()
            tail.append(line)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout="".join(tail), stderr="")


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
