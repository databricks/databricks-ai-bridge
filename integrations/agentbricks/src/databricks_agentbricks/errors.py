"""CLI compatibility exports and presentation for errors shared with AgentKit."""

from databricks_agentkit._errors import (
    FEATURE_DISABLED,
    TRANSIENT_ERROR_CODES,
    AgentCliError,
    _set_renderer,
    error_output_mode,
    get_output_mode,
    set_output_mode,
    set_usage_output_mode,
    wrap_api_error,
)

__all__ = [
    "AgentCliError",
    "FEATURE_DISABLED",
    "TRANSIENT_ERROR_CODES",
    "error_output_mode",
    "get_output_mode",
    "set_output_mode",
    "set_usage_output_mode",
    "wrap_api_error",
]


def _render_error(error):
    from rich.console import Console

    from databricks_agentbricks.presentation import render

    render.diagnostic(
        "error", error.message, code=error.error_code, help=error.hint, con=Console(stderr=True)
    )


_set_renderer(_render_error)
