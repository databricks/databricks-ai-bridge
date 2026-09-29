"""The user-interaction boundary a service talks through.

Defines the :class:`Reporter` and :class:`Prompter` protocols - the only way ``DeployService``
touches the terminal. Progress, notices, and primary output go out through the reporter; the one
destructive-verb confirmation comes back through the prompter. Both are CLI-framework-agnostic
protocols, implemented by an adapter (e.g. a Click/rich presenter), so the core business logic never
imports click or any render layer.
"""

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


class Prompter(Protocol):
    """How a service asks the operator to confirm a destructive step. Separate from
    :class:`Reporter` because it reads from the terminal rather than writing to it, and only the
    destructive verbs need it; the same adapter reasoning applies - the service never imports click."""

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        """Ask a yes/no question, returning the operator's answer."""
        ...
