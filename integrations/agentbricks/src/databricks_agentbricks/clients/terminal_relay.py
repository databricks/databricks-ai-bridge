"""Relay a command's output through a pseudo-terminal, with a status line in the quiet gaps.

``agentbricks dev`` runs ``databricks apps run-local``, which builds the environment and then starts
the agent's server. Between the last install line and the server answering there can be a long,
silent stretch (the agent importing its libraries) where the terminal shows only a static cursor.
Drawing a spinner there without it overwriting the logs means knowing when the output goes quiet, so
the output has to pass through us. A pseudo-terminal keeps that invisible to the tools: they still
see a terminal, so colors, uv's own progress display, and interactive prompts (a ``breakpoint()``)
behave exactly as when they write to the terminal directly. Every byte is passed through unchanged;
the status line is only ever drawn on an empty line and erased before more output is written.

POSIX only: callers check :func:`supported` and fall back to plain streaming elsewhere.
"""

from __future__ import annotations

import errno
import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from typing import Optional

# Seconds without output before the status line appears.
_QUIET_AFTER_S = 1.0
# Seconds between select() wakeups: how often the status line animates and the probe can run.
_TICK_S = 0.1
# Seconds between readiness probes.
_READY_POLL_S = 0.25
# Seconds an interrupted command gets to exit on its own before it is killed (as `subprocess.run`).
_SIGINT_WAIT_S = 0.25
# Seconds to keep relaying output after the command exits or is interrupted, so a server's shutdown
# logs still reach the terminal (and its writes don't fail), without waiting on it forever.
_OUTPUT_GRACE_S = 3.0
# Return to the start of the line and clear it: erases the status line.
_CLEAR_LINE = b"\r\x1b[2K"


def supported() -> bool:
    """Whether output can be relayed: a POSIX system with stdout and stderr on a terminal.

    Both streams must be the terminal, since the command's stdout and stderr share one
    pseudo-terminal; with either redirected, merging them would change where output lands.
    """
    if sys.platform == "win32":
        return False
    try:
        return os.isatty(1) and os.isatty(2)
    except OSError:
        return False


def run(
    cmd: list[str],
    *,
    cwd: Optional[str],
    env: Optional[Mapping[str, str]],
    probe: Callable[[], bool],
    on_ready: Optional[Callable[[], None]],
    quiet_status: Callable[[float], str],
) -> subprocess.CompletedProcess:
    """Run ``cmd`` on a pseudo-terminal, relaying its output to ours unchanged.

    ``probe`` is polled until it first succeeds, then ``on_ready`` is called. Until then, whenever
    the output has been quiet for a moment at the start of a line, the line ``quiet_status(elapsed)``
    returns is drawn in place (``elapsed`` is seconds since the command started) and erased as soon
    as more output arrives or the command is ready.
    """
    import pty  # POSIX only; imported here so the module still loads on Windows

    master, slave = pty.openpty()
    _copy_window_size(master)
    previous_winch = signal.signal(signal.SIGWINCH, lambda *_: _copy_window_size(master))
    try:
        # stdin stays the real terminal, so typed input (e.g. at a debugger prompt) still reaches
        # the app; the process group is unchanged, so Ctrl+C reaches it as before.
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=slave, stderr=slave)
    except BaseException:
        os.close(master)
        os.close(slave)
        signal.signal(signal.SIGWINCH, previous_winch)
        raise
    os.close(slave)
    relay = _Relay(master)
    try:
        start = time.monotonic()
        ready = False
        last_probe = 0.0
        exited_at: Optional[float] = None
        while True:
            relay.pump(_TICK_S)
            now = time.monotonic()
            if proc.poll() is not None:
                exited_at = exited_at or now
                if relay.closed or now - exited_at > _OUTPUT_GRACE_S:
                    break
                continue
            if not ready and now - last_probe >= _READY_POLL_S:
                last_probe = now
                if probe():
                    ready = True
                    relay.clear_status()
                    if on_ready is not None:
                        on_ready()
            if not ready and relay.quiet_for(now) >= _QUIET_AFTER_S:
                relay.show_status(quiet_status(now - start))
        relay.clear_status()
    except KeyboardInterrupt:
        relay.clear_status()
        # Ctrl+C reaches the child too. Keep relaying its shutdown output, give it a moment to exit
        # cleanly as `subprocess.run` does, then make sure it is gone.
        deadline = time.monotonic() + _SIGINT_WAIT_S
        while proc.poll() is None and time.monotonic() < deadline:
            relay.pump(_TICK_S)
        if proc.poll() is None:
            proc.kill()
        relay.drain(_OUTPUT_GRACE_S)
        raise
    except BaseException:
        relay.clear_status()
        proc.kill()
        raise
    finally:
        signal.signal(signal.SIGWINCH, previous_winch)
        os.close(master)
        proc.wait()
    return subprocess.CompletedProcess(cmd, proc.returncode)


class _Relay:
    """Copies the pseudo-terminal's output to our stdout and owns the in-place status line."""

    def __init__(self, master: int) -> None:
        self._master = master
        self.closed = False
        self._status_shown = False
        self._at_line_start = True
        self._last_output = time.monotonic()

    def pump(self, timeout: float) -> None:
        """Relay whatever output arrives within ``timeout`` seconds."""
        if self.closed:
            time.sleep(timeout)
            return
        readable, _, _ = select.select([self._master], [], [], timeout)
        if not readable:
            return
        try:
            data = os.read(self._master, 65536)
        except OSError as exc:
            # Every writer has closed its end: EIO on Linux, EOF (b"") on macOS.
            if exc.errno != errno.EIO:
                raise
            data = b""
        if not data:
            self.closed = True
            return
        self.clear_status()
        _write(data)
        self._last_output = time.monotonic()
        self._at_line_start = data.endswith(b"\n")

    def drain(self, timeout: float) -> None:
        """Relay output until the pseudo-terminal closes, for at most ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while not self.closed and time.monotonic() < deadline:
            self.pump(_TICK_S)

    def quiet_for(self, now: float) -> float:
        """Seconds since the last output, or 0 when the cursor is mid-line (a prompt, a progress bar)."""
        return now - self._last_output if self._at_line_start else 0.0

    def show_status(self, line: str) -> None:
        _write(_CLEAR_LINE + line.encode())
        self._status_shown = True

    def clear_status(self) -> None:
        if self._status_shown:
            _write(_CLEAR_LINE)
            self._status_shown = False


def _write(data: bytes) -> None:
    sys.stdout.flush()  # keep our own buffered output in order with the raw writes
    view = memoryview(data)
    while view:
        written = os.write(1, view)
        view = view[written:]


def _copy_window_size(master: int) -> None:
    """Give the pseudo-terminal our terminal's size, so the tools wrap lines where it does."""
    import fcntl
    import termios

    try:
        size = fcntl.ioctl(1, termios.TIOCGWINSZ, b"\0" * 8)
        fcntl.ioctl(master, termios.TIOCSWINSZ, size)
    except OSError:
        pass
