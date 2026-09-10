"""Construct the Anthropic client the agent calls Claude through.

Defaults to ``anthropic.Anthropic()`` (reads ``ANTHROPIC_API_KEY``). Set ``CLAUDE_CODE_USE_BEDROCK``
to route through Amazon Bedrock using the app's AWS credentials instead — no Anthropic key needed.
"""

from __future__ import annotations

import os
from typing import Any


def use_bedrock() -> bool:
    """Whether to route through Bedrock (``CLAUDE_CODE_USE_BEDROCK`` set to a truthy value)."""
    return os.getenv("CLAUDE_CODE_USE_BEDROCK", "").strip().lower() in {"1", "true", "yes"}


def client() -> Any:
    """An ``anthropic`` client: ``AnthropicBedrock`` when ``CLAUDE_CODE_USE_BEDROCK`` is set, else ``Anthropic``."""
    import anthropic

    if use_bedrock():
        return anthropic.AnthropicBedrock()
    return anthropic.Anthropic()
