"""Error types for the Mason CLI and mapping from Databricks REST errors.

`AgentCliError` extends `click.ClickException` so a raised error prints as a clean
one-liner (plus an optional hint) and exits non-zero, instead of dumping a traceback.
"""

from __future__ import annotations

import json
from typing import Optional

import click
from rich.console import Console
from rich.text import Text

# Error codes indicating that a preview API is unavailable in the workspace.
_PREVIEW_ERROR_CODES = frozenset({"NOT_IMPLEMENTED", "UNIMPLEMENTED", "FEATURE_DISABLED"})

# gRPC-style status codes the backend returns for transient failures (a cancelled or
# deadline-exceeded RPC, a briefly unavailable service). Shared with the client, which
# retries these before surfacing them. `client._do` retries; the hint here covers the
# case where retries are still exhausted.
TRANSIENT_ERROR_CODES = frozenset({"CANCELLED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "ABORTED"})

# Process-global output mode, set once by the root CLI group. When "json", errors are
# emitted as a machine-readable JSON object instead of the styled text one-liner, so a
# script driving `mason -o json` can parse failures instead of scraping human text.
_OUTPUT_MODE = "text"


def set_output_mode(mode: str) -> None:
    """Record the CLI's --output mode so errors can render to match it."""
    global _OUTPUT_MODE
    _OUTPUT_MODE = mode


_PREVIEW_HINT = (
    "These agents/v1 APIs are in preview and gated per workspace. This handler is "
    "not enabled on the target workspace yet — try a different --profile or contact "
    "your workspace administrator."
)

_TRANSIENT_HINT = (
    "This is usually a transient backend issue — re-running the command often succeeds."
)


class AgentCliError(click.ClickException):
    """A user-facing CLI error rendered without a Python traceback."""

    def __init__(
        self, message: str, *, error_code: Optional[str] = None, hint: Optional[str] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.hint = hint

    def show(self, file=None) -> None:
        if _OUTPUT_MODE == "json":
            payload: dict = {"message": self.message}
            if self.error_code:
                payload["code"] = self.error_code
            if self.hint:
                payload["hint"] = self.hint
            click.echo(json.dumps({"error": payload}, indent=2), err=True)
            return
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
    # A message-less DatabricksError wraps `IOError(None)`, so `str(exc)` is the literal
    # "None". Treat that (and an empty string) as "no detail" so we surface the error code
    # instead of a bare `Error [CANCELLED]: None`.
    detail = str(exc).strip()
    if detail == "None":
        detail = ""
    message = detail or _no_detail_message(error_code) or exc.__class__.__name__
    if error_code in _PREVIEW_ERROR_CODES:
        hint = _PREVIEW_HINT
    elif error_code in TRANSIENT_ERROR_CODES:
        hint = _TRANSIENT_HINT
    else:
        hint = None
    return AgentCliError(message, error_code=error_code, hint=hint)


def _no_detail_message(error_code: Optional[str]) -> Optional[str]:
    """Fallback message when the server returns an error code but no human-readable detail."""
    if not error_code:
        return None
    return f"The server returned {error_code} with no further detail."
