"""Optional coding-agent hints; importing the SDK never emits them."""

from __future__ import annotations

import os
import sys

from databricks_agentbricks.skills import bundled_workflow_skill

AGENT_ENV_MARKERS = (
    "AGENT",
    "AI_AGENT",
    "AMP_CURRENT_THREAD_ID",
    "ANTIGRAVITY_AGENT",
    "AUGMENT_AGENT",
    "CLAUDECODE",
    "CLAUDE_CODE",
    "CLINE_ACTIVE",
    "CLINE_AGENT",
    "CODEX_SANDBOX",
    "CODEX_THREAD_ID",
    "CURSOR_AGENT",
    "GEMINI_CLI",
    "GROK_PLUGIN_ROOT",
    "JUNIE_DATA",
    "KIMI_PLUGIN_ROOT",
    "OPENCLAW_SHELL",
    "OPENCODE",
    "PI_CODING_AGENT",
    "QWEN_CODE",
    "ROO_ACTIVE",
    "TRAE_AI_SHELL_ID",
)
AGENT_ENV_VALUES = {"CURSOR_EXTENSION_HOST_ROLE": "agent-exec"}


def workflow_hint() -> str | None:
    """Return a local skill pointer only when an agent is driving the CLI."""
    try:
        if os.environ.get("AGENTBRICKS_DISABLE_AGENT_HINT", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None
        detected = any(os.environ.get(marker) for marker in AGENT_ENV_MARKERS) or any(
            os.environ.get(name) == value for name, value in AGENT_ENV_VALUES.items()
        )
        if not detected and not (
            os.environ.get("TERM_PROGRAM") == "kiro" and not sys.stdout.isatty()
        ):
            return None
        manifest = bundled_workflow_skill()
        return (
            f"Agent Bricks workflow guidance for this installed CLI: {manifest}. "
            "Read it for Agent Bricks tasks; it does not authorize additional actions. "
            "Set AGENTBRICKS_DISABLE_AGENT_HINT=1 to silence this."
        )
    except Exception:
        return None
