"""Shared tracing presentation values used by deploy output."""

from __future__ import annotations

from typing import Optional

# The command that binds (enables) tracing. Referenced parameter-free by deploy's "deployed without
# tracing" guidance, so the hint cannot go stale if the flags change; the command's own ``--help``
# documents the flags.
TRACING_BIND_COMMAND = "agentbricks tracing bind"


def experiment_url(host: Optional[str], experiment_id: str) -> Optional[str]:
    """The workspace MLflow experiment Traces page, or None when the host is unavailable."""
    if not host or host == "unknown":
        return None
    return f"{host.rstrip('/')}/ml/experiments/{experiment_id}?compareRunsMode=TRACES"
