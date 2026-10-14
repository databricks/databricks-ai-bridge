"""The ``agentbricks`` CLI and the agents it produces: the system under test.

Every CLI step is a real ``agentbricks`` subprocess, written as the command a developer would type
(``cli.run("tools add ...")``), so a regression in the CLI fails the test that makes it. Nothing in
this module knows about any particular suite or deletes anything: the ``project`` fixture does.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping

from harness.logs import last_nonempty_line, log, log_command
from harness.run import E2EError, RunConfig, now
from harness.target import TargetWorkspace

# The Apps gateway's answers while an App is not up, e.g. restarting after a redeploy.
_GATEWAY_UNAVAILABLE = re.compile(r"HTTP 50[234]\b")


@dataclasses.dataclass
class BackgroundCommand:
    """A CLI command left running by ``AgentbricksCli.start``, e.g. ``agentbricks dev``."""

    label: str
    process: subprocess.Popen[str]
    log_path: pathlib.Path


class AgentbricksCli:
    """Runs ``agentbricks`` commands against a target workspace, logging each to its own files."""

    def __init__(self, target_workspace: TargetWorkspace, run_config: RunConfig, test_name: str):
        self.target_workspace = target_workspace
        self.run_config = run_config
        # One directory per test: tests run concurrently and each numbers its commands from 1.
        self.logs_dir = (
            run_config.output / "logs" / re.sub(r"[^A-Za-z0-9_.-]+", "-", test_name).strip("-")
        )
        # The CLI prints a hint when it detects a coding agent; the suite must behave the same
        # whoever launches it, and the hint would corrupt --output json errors on stderr.
        self._env = {**os.environ, **target_workspace.env, "AGENTBRICKS_DISABLE_AGENT_HINT": "1"}
        self._sequence = 0

    def run(
        self,
        command: str,
        *,
        timeout: float = 300,
        check: bool = True,
        retry_unavailable: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``agentbricks <command>`` to completion.

        ``retry_unavailable`` retries only the Apps gateway's "App not up" answers (a staging App
        can restart under the test); any other failure is the CLI's.
        """
        argv = self._argv(command)
        for attempt in range(1, 4):
            result = self._run_once(argv, self._label(command), timeout)
            output = result.stderr or result.stdout
            if (
                result.returncode == 0
                or not retry_unavailable
                or attempt == 3
                or _GATEWAY_UNAVAILABLE.search(output) is None
            ):
                break
            log(f"attempt {attempt}/3 | {command.split()[0]} | App unavailable; retrying")
            time.sleep(15)
        if check and result.returncode != 0:
            raise E2EError(f"Command failed ({result.returncode}): {shlex.join(argv)}\n{output}")
        return result

    @contextlib.contextmanager
    def start(
        self, command: str, *, env: Mapping[str, str] | None = None
    ) -> Iterator[BackgroundCommand]:
        """Run ``agentbricks <command>`` in the background for the block, then stop it."""
        argv = self._argv(command)
        label = self._label(command)
        log_path = self.logs_dir / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_command(argv)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                argv,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**self._env, **(env or {})},
            )
        try:
            yield BackgroundCommand(label, process, log_path)
        finally:
            if process.poll() is None:
                terminate_process_group(process)
            log(f"{label} | stopped")

    def _argv(self, command: str) -> list[str]:
        return [_agentbricks_bin(), *self.target_workspace.cli_args, *shlex.split(command)]

    def _label(self, command: str) -> str:
        """A log name from the command's first two words, e.g. ``007-tools-add``."""
        self._sequence += 1
        words = shlex.split(command)
        while words and words[0].startswith("-"):
            # A leading global option such as ``--output json``, and its value.
            words = words[1:] if "=" in words[0] else words[2:]
        # The second word only when it is a subcommand or name, never a flag or a path.
        names = words[:1] + [w for w in words[1:2] if re.fullmatch(r"[a-z][a-z0-9_-]*", w)]
        return "-".join([f"{self._sequence:03d}", *names])

    def _run_once(
        self, argv: list[str], label: str, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        """Run with stdout and stderr in log files, printing progress while a command is slow."""
        stdout_path = self.logs_dir / f"{label}.out"
        stderr_path = self.logs_dir / f"{label}.err"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        log_command(argv)
        started = time.monotonic()
        with (
            stdout_path.open("w", encoding="utf-8") as out,
            stderr_path.open("w", encoding="utf-8") as err,
        ):
            process = subprocess.Popen(
                argv,
                env=self._env,
                text=True,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            next_tick = 60.0
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    terminate_process_group(process, grace=0)
                    raise E2EError(f"{label} timed out after {timeout:.0f}s; log: {stderr_path}")
                if elapsed >= next_tick:
                    progress = last_nonempty_line(stderr_path) or last_nonempty_line(stdout_path)
                    log(f"tick {now():%H:%M} | {label} | running | {progress}")
                    next_tick += 60.0
                time.sleep(0.2 if elapsed < 5 else 2)
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        for text in (stdout, stderr):
            if text.strip():
                log(text)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _agentbricks_bin() -> str:
    """The CLI of the interpreter running the tests, i.e. the source checkout's own install."""
    sibling = pathlib.Path(sys.executable).parent / "agentbricks"
    found = str(sibling) if sibling.exists() else shutil.which("agentbricks")
    if found is None:
        raise E2EError("The agentbricks CLI is not installed in the test environment.")
    return found


def terminate_process_group(process: subprocess.Popen[str], *, grace: float = 20) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if grace <= 0:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
