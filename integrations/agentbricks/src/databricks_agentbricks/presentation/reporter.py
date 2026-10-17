"""Click-backed terminal reporter for Agent Bricks services."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import Optional

import click

from databricks_agentbricks.presentation import render

# Debug switch for per-phase durations (documented in CONTRIBUTING.md, deliberately not in --help).
DEBUG_TIMINGS_ENV = "AGENTBRICKS_DEBUG_TIMINGS"


class ClickReporter:
    """A :class:`Reporter` backed by ``render`` + ``click``: spinners, progress lines, and notices."""

    def status(self, message: str) -> AbstractContextManager[None]:
        """The :class:`Reporter` spinner, as a ``render.status`` context manager (rich, on stderr)."""
        return render.status(message)

    def progress(
        self, message: str, *, done_message: Optional[str] = None
    ) -> AbstractContextManager[None]:
        """The :class:`Reporter` progress line, as a ``render.progress`` context manager (rich)."""
        return render.progress(message, done_message=done_message)

    def note(self, message: str) -> None:
        """The :class:`Reporter` notice, as a ``click.echo`` to stderr so it never pollutes stdout."""
        click.echo(message, err=True)

    def echo(self, message: str, *, add_newline: bool = True) -> None:
        """The :class:`Reporter` primary output, as a ``click.echo`` to stdout (``nl=add_newline``)."""
        click.echo(message, nl=add_newline)

    def timing(self, phase: str, seconds: float) -> None:
        """The :class:`Reporter` phase duration, on stderr only when ``AGENTBRICKS_DEBUG_TIMINGS=1``."""
        if os.environ.get(DEBUG_TIMINGS_ENV) == "1":
            click.echo(f"timing: {phase} {seconds:.1f}s", err=True)
