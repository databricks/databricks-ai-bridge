"""Click-backed terminal reporter for Agent Bricks services."""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager

import click

from databricks_agentbricks import errors
from databricks_agentbricks.presentation import render


class ClickReporter:
    """A :class:`Reporter` backed by ``render`` + ``click``: spinners, progress lines, and notices."""

    def status(self, message: str) -> AbstractContextManager[None]:
        """The :class:`Reporter` spinner, as a ``render.status`` context manager (rich, on stderr)."""
        return render.status(message)

    def progress(self, message: str) -> AbstractContextManager[None]:
        """The :class:`Reporter` progress line, as a ``render.progress`` context manager (rich)."""
        return render.progress(message)

    def note(self, message: str) -> None:
        """The :class:`Reporter` notice, as a ``click.echo`` to stderr so it never pollutes stdout.

        Written to the current ``sys.stderr``, not ``err=True``: while a TTY spinner is live, rich
        swaps ``sys.stderr`` for a proxy that prints above the spinner, and ``err=True`` writes
        past that proxy to the raw stream, where the spinner's redraw can overwrite the notice.
        """
        click.echo(message, file=sys.stderr)

    def echo(self, message: str, *, add_newline: bool = True) -> None:
        """The :class:`Reporter` primary output, as a ``click.echo`` (``nl=add_newline``).

        Goes to stdout, except under ``-o json``, where stdout carries only the command's JSON
        document and relayed command output moves to stderr. Like :meth:`note`, it writes to the
        current stream so output survives a live spinner.
        """
        stream = sys.stderr if errors._OUTPUT_MODE == "json" else sys.stdout
        click.echo(message, nl=add_newline, file=stream)
