"""Run local commands with bounded logs, readiness reporting and process cleanup."""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager

from databricks_agentbricks.clients.output_tail import StreamRelay


def stop_process(process, *, isolated=False, signum=signal.SIGTERM, grace=3.0):
    """Reap the command and terminate descendants in its dedicated POSIX process group."""

    def send(sig):
        try:
            if isolated and os.name == "posix":
                os.killpg(process.pid, sig)
            elif process.poll() is None:
                process.send_signal(sig)
        except ProcessLookupError:
            pass

    send(signum)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
    finally:
        # A command can exit while its server or tool children still hold the port.
        if isolated and os.name == "posix":
            send(signal.SIGKILL)
        process.wait()


def run_until_ready(
    cmd: list[str],
    *,
    cwd: str | None,
    env: Mapping[str, str] | None,
    probe: Callable[[], bool],
    on_ready: Callable[[], None] | None,
    on_exit: Callable[[int], None] | None = None,
    poll_seconds: float = 0.25,
    isolated: bool = False,
) -> subprocess.CompletedProcess:
    """Stream output and announce readiness only while the child is running."""
    stdout_target = None if os.isatty(1) else subprocess.PIPE
    stderr_target = None if os.isatty(2) else subprocess.PIPE
    if stdout_target is not None and stderr_target is not None:
        try:
            out, err = os.fstat(1), os.fstat(2)
            if out.st_ino and os.path.samestat(out, err):
                if stat.S_ISREG(out.st_mode):
                    stdout_target = stderr_target = None
                else:
                    stderr_target = subprocess.STDOUT
        except OSError:
            pass
    with subprocess.Popen(
        cmd,
        stdout=stdout_target,
        stderr=stderr_target,
        bufsize=0,
        cwd=cwd,
        env=env,
        start_new_session=isolated and os.name == "posix",
    ) as proc:
        stdout = StreamRelay(proc.stdout, 1) if proc.stdout is not None else None
        stderr = StreamRelay(proc.stderr, 2) if proc.stderr is not None else None
        drain_output = False
        try:
            while proc.poll() is None:
                if probe():
                    if on_ready is not None:
                        on_ready()
                    proc.wait()
                    break
                time.sleep(poll_seconds)
            drain_output = True
        except KeyboardInterrupt:
            stop_process(
                proc, isolated=isolated, signum=signal.SIGINT, grace=3.0 if isolated else 0.25
            )
            raise
        except BaseException:
            if not isolated:
                proc.kill()
            raise
        finally:
            try:
                if isolated:
                    stop_process(proc, isolated=True)
                proc.wait()
                if on_exit is not None:
                    on_exit(proc.returncode)
                if drain_output:
                    for relay in (stdout, stderr):
                        if relay is not None:
                            relay.finish()
            finally:
                for relay in (stdout, stderr):
                    if relay is not None:
                        relay.cancel()
    return subprocess.CompletedProcess(
        cmd,
        proc.returncode,
        stdout=stdout.tail.text() if stdout is not None else "",
        stderr=stderr.tail.text() if stderr is not None else "",
    )


@contextmanager
def _terminate_on_signal():
    previous = None
    if threading.current_thread() is threading.main_thread():

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def run_process(
    cmd, *, cwd=None, env=None, ready_probe=None, on_ready=None, on_exit=None, quiet_status=None
):
    """Run a local build or server without invoking the Databricks CLI."""
    from databricks_agentbricks.clients import terminal_relay

    with _terminate_on_signal():
        if ready_probe is not None and quiet_status is not None and terminal_relay.supported():
            return terminal_relay.run(
                cmd,
                cwd=cwd,
                env=env,
                probe=ready_probe,
                on_ready=on_ready,
                on_exit=on_exit,
                quiet_status=quiet_status,
                isolated=True,
            )
        return run_until_ready(
            cmd,
            cwd=cwd,
            env=env,
            probe=ready_probe or (lambda: False),
            on_ready=on_ready,
            on_exit=on_exit,
            isolated=True,
        )
