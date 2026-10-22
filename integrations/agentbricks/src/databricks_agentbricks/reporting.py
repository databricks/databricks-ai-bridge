"""Framework-neutral progress and output port shared by CLI services.

Defines the CLI-framework-agnostic :class:`Reporter` protocol. Progress, notices, and primary
output go through its adapter; confirmation stays in the CLI command, so the service does not
import click or any render layer.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Optional, Protocol


class Reporter(Protocol):
    """How a service surfaces progress and notices. CLI-framework-agnostic: implemented by an
    adapter (e.g. a Click/rich reporter) so the core business logic never imports click or render."""

    def status(self, message: str) -> AbstractContextManager[None]:
        """A transient spinner around a short reconcile step."""
        ...

    def progress(
        self, message: str, *, done_message: Optional[str] = None
    ) -> AbstractContextManager[None]:
        """A persistent progress line around a long-running provision step.

        With ``done_message`` a successful exit also prints ``done_message`` with the total wait.
        """
        ...

    def timing(self, phase: str, seconds: float) -> None:
        """A per-phase duration, shown only when the adapter's debug-timings switch is on."""
        ...

    def note(self, message: str) -> None:
        """A one-off user notice (stderr)."""
        ...

    def echo(self, message: str, *, add_newline: bool = True) -> None:
        """Write primary output (stdout), terminating it with a newline unless ``add_newline`` is False.

        Pass ``add_newline=False`` for text that already ends in one, so relayed command output isn't
        double-spaced.
        """
        ...


# How often a long phase reports its elapsed time. Shared by the live spinner/progress renderers and
# by `agentbricks deployments wait`, so every surface prints the same `• <phase>… M:SS` line.
HEARTBEAT_INTERVAL_S = 15.0


def phase_name(message: str) -> str:
    """The stable phase name behind a progress message: ``"Deploying…"`` becomes ``"Deploying"``."""
    return message.strip().rstrip("…").rstrip(".").rstrip()


def format_elapsed(seconds: float) -> str:
    """Render an elapsed duration as ``M:SS`` (``0:45``, ``12:03``)."""
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


def heartbeat_line(phase: str, elapsed_s: float) -> str:
    """The plain heartbeat line for a phase that is still running."""
    return f"• {phase}… {format_elapsed(elapsed_s)}"


class Heartbeat:
    """The elapsed-time schedule for one long phase, independent of any clock or thread.

    ``tick(now)`` returns a heartbeat line each time the phase crosses another ``interval`` of
    elapsed time, and ``None`` otherwise. ``start`` may lie before ``now`` (a waiter that picks up a
    phase already in progress); the first line is then due at the next whole interval.
    """

    def __init__(
        self,
        message: str,
        *,
        start: float,
        now: Optional[float] = None,
        interval: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        self.phase = phase_name(message)
        self.start = start
        self._interval = interval
        already = max(0.0, (start if now is None else now) - start)
        self._next = (int(already // interval) + 1) * interval

    def tick(self, now: float) -> Optional[str]:
        """The heartbeat line due at ``now``, if any."""
        elapsed = now - self.start
        if elapsed < self._next:
            return None
        while self._next <= elapsed:
            self._next += self._interval
        return heartbeat_line(self.phase, elapsed)
