from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol


class Reporter(Protocol):
    """How a service surfaces progress and notices. CLI-framework-agnostic: implemented by an
    adapter (e.g. a Click/rich reporter) so the core business logic never imports click or render."""

    def status(self, message: str) -> AbstractContextManager[None]:
        """A transient spinner around a short reconcile step."""
        ...

    def progress(self, message: str) -> AbstractContextManager[None]:
        """A persistent progress line around a long-running provision step."""
        ...

    def note(self, message: str) -> None:
        """A one-off user notice (stderr)."""
        ...

    def echo(self, message: str, *, newline: bool = True) -> None:
        """Write a line of primary output (stdout)."""
        ...
