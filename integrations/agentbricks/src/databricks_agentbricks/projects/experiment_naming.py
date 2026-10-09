"""Workspace-independent names for project tracing experiments."""

from __future__ import annotations

import re
from typing import Optional

from databricks_agentbricks.errors import AgentCliError


def default_experiment_name(project: Optional[str], token: Optional[str] = None) -> str:
    """Return a project-specific experiment path under ``/Shared/agentbricks_traces``."""
    if not project:
        raise AgentCliError("Cannot derive the default tracing experiment without a project name.")
    slug = re.sub(r"[^a-z0-9-]+", "-", project.lower()).strip("-") or "agent"
    middle = f"-{token}" if token else ""
    return f"/Shared/agentbricks_traces/{slug}{middle}"
