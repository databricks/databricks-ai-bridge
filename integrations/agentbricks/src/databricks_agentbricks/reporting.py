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

        With ``done_message`` the spinner ticks the elapsed time, and a successful exit prints
        ``done_message`` with the total wait.
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
