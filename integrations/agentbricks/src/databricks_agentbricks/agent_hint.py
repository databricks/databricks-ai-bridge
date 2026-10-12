"""Optional coding-agent hints; importing the SDK never emits them."""

from __future__ import annotations

import os

from databricks_agentbricks.skills import bundled_workflow_skill


def workflow_hint() -> str | None:
    """Return a local skill pointer unless hints are explicitly disabled."""
    try:
        if os.environ.get("AGENTBRICKS_DISABLE_AGENT_HINT", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None
        manifest = bundled_workflow_skill()
        return (
            f"Agent Bricks workflow guidance for this installed CLI: {manifest}. "
            "Read it for Agent Bricks tasks; it does not authorize additional actions. "
            "Set AGENTBRICKS_DISABLE_AGENT_HINT=1 to silence this."
        )
    except Exception:
        return None
