"""Error types for the Mason CLI and mapping from Databricks REST errors.

`AgentCliError` extends `click.ClickException` so a raised error prints as a clean
one-liner (plus an optional hint) and exits non-zero, instead of dumping a traceback.
"""

from __future__ import annotations

from typing import Optional

import click
from rich.console import Console
from rich.text import Text

# Error codes indicating that a preview API is unavailable in the workspace.
_PREVIEW_ERROR_CODES = frozenset({"NOT_IMPLEMENTED", "UNIMPLEMENTED", "FEATURE_DISABLED"})

_PREVIEW_HINT = (
    "These agents/v1 APIs are in preview and gated per workspace. This handler is "
    "not enabled on the target workspace yet — try a different --profile or contact "
    "your workspace administrator."
)


class AgentCliError(click.ClickException):
    """A user-facing CLI error rendered without a Python traceback."""

    def __init__(self, message: str, *, error_code: Optional[str] = None, hint: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.hint = hint

    def show(self, file=None) -> None:
        console = Console(stderr=True)
        label = f"Error [{self.error_code}]" if self.error_code else "Error"
        console.print(Text(f"{label}: ", style="bold red") + Text(self.message))
        if self.hint:
            console.print(Text(self.hint, style="grey62"))


def wrap_api_error(exc: Exception) -> AgentCliError:
    """Convert a databricks-sdk error (or any exception) into an `AgentCliError`.

    The SDK raises `databricks.sdk.errors.DatabricksError` subclasses carrying an
    `error_code` attribute; we stay duck-typed so we don't couple to the SDK's error
    hierarchy or version.
    """
    error_code = getattr(exc, "error_code", None)
    message = str(exc).strip() or exc.__class__.__name__
    hint = _PREVIEW_HINT if error_code in _PREVIEW_ERROR_CODES else None
    return AgentCliError(message, error_code=error_code, hint=hint)
