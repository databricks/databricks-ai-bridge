"""Click-backed terminal reporter for Agent Bricks services."""

from __future__ import annotations

from contextlib import AbstractContextManager

import click

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
        """The :class:`Reporter` notice, as a ``click.echo`` to stderr so it never pollutes stdout."""
        click.echo(message, err=True)

    def echo(self, message: str, *, add_newline: bool = True) -> None:
        """The :class:`Reporter` primary output, as a ``click.echo`` to stdout (``nl=add_newline``)."""
        click.echo(message, nl=add_newline)
