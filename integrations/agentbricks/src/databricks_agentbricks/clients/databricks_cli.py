"""Thin wrapper around the ``databricks`` CLI.

Agent Bricks shells out to the ``databricks`` CLI for the operations it doesn't call over the REST client
(``apps`` lifecycle, ``sync``, ``postgres`` endpoint/credential lookups). This is the single place
that builds the argv, threads the ``--profile`` through, and normalizes a non-zero exit into an
``AgentCliError``. Kept separate from any one caller so ``deploy``, ``dev``, and ``app_resources``
share it without importing each other.
"""

from __future__ import annotations

import collections
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from typing import Optional

from databricks_agentbricks.clients import terminal_relay
from databricks_agentbricks.errors import AgentCliError

# Lines of relayed output a `tee` run keeps for callers that inspect a failure.
_TEE_TAIL_LINES = 200
# Seconds between readiness probes while a long-running command starts up.
_READY_POLL_S = 0.25
# Seconds an interrupted command gets to exit on its own before it is killed (as `subprocess.run`).
_SIGINT_WAIT_S = 0.25


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
    ready_probe: Optional[Callable[[], bool]] = None,
    on_ready: Optional[Callable[[], None]] = None,
    quiet_status: Optional[Callable[[float], str]] = None,
) -> subprocess.CompletedProcess:
    """Run one ``databricks`` command.

    ``capture`` collects stdout/stderr instead of showing them. ``tee`` relays the command's merged
    output live to *our* stderr, line by line, and also returns its tail as ``stdout``: the user
    still sees a long ``sync``/``apps deploy`` as it runs, stdout stays clean for ``-o json``, and the
    caller can still match the failure text (for example, a busy App).

    ``ready_probe`` is for a long-running server command such as ``apps run-local``: its output
    streams as usual, the probe is polled while the command runs, and ``on_ready`` is called once
    the probe first succeeds. A command that exits before it is ready never triggers ``on_ready``,
    so the caller can hold back a success message until the server actually answers. With
    ``quiet_status`` too, on a terminal that supports it, the output is relayed through a
    pseudo-terminal and that status line is drawn whenever the output goes quiet before then (see
    ``terminal_relay``); otherwise the output streams as usual.
    """
    cmd = ["databricks", *args]
    if profile:
        cmd += ["--profile", profile]
    if ready_probe is not None and quiet_status is not None and terminal_relay.supported():
        result = terminal_relay.run(
            cmd, cwd=cwd, env=env, probe=ready_probe, on_ready=on_ready, quiet_status=quiet_status
        )
    elif ready_probe is not None:
        result = _run_until_ready(cmd, cwd=cwd, env=env, probe=ready_probe, on_ready=on_ready)
    elif tee:
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


def _run_until_ready(
    cmd: list[str],
    *,
    cwd: Optional[str],
    env: Optional[Mapping[str, str]],
    probe: Callable[[], bool],
    on_ready: Optional[Callable[[], None]],
) -> subprocess.CompletedProcess:
    """Run ``cmd`` with its output streamed, calling ``on_ready`` once ``probe`` first succeeds."""
    with subprocess.Popen(cmd, text=True, cwd=cwd, env=env) as proc:
        try:
            while proc.poll() is None:
                if probe():
                    if on_ready is not None:
                        on_ready()
                    proc.wait()
                    break
                time.sleep(_READY_POLL_S)
        except KeyboardInterrupt:
            # Ctrl-C reaches the child too; give it a moment to exit cleanly, as `subprocess.run`
            # does, then make sure it is gone.
            try:
                proc.wait(timeout=_SIGINT_WAIT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        except BaseException:
            proc.kill()
            raise
    return subprocess.CompletedProcess(cmd, proc.returncode)
