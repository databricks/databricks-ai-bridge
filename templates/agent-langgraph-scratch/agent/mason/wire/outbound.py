"""Serialize LangGraph astream events to JSON dicts — the SDK's native shape, no imposed contract.

``astream(stream_mode=["updates", "messages"])`` yields two event shapes: ``updates`` (completed
node outputs — full LangChain messages, incl. tool calls/results) and ``messages`` (token-level
chunks for streaming text). We relay each as-is, made JSON: completed messages under
``{"type": "message", "message": <LangChain message dict>}`` and text chunks under
``{"type": "delta", "content": ..., "id": ...}``. Nothing is reshaped into the Responses contract —
the client receives LangGraph's native output. (The AgentServer-backed templates emit Responses-shaped
events; this from-scratch one shows the raw SDK shape instead.)

When a human-approval gate fires (see ``REQUIRE_APPROVAL`` in ``agent/agent.py``), ``updates`` carries
an ``__interrupt__`` key instead of a node output; we relay it as ``{"type": "interrupt", "id": ...,
"value": <HITLRequest>}``. The run is now paused on the session's thread — the client approves/edits/
rejects by POSTing ``resume`` back with the same ``session_id`` (see ``wire/inbound.py``).
"""

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from langchain.messages import AIMessageChunk

logger = logging.getLogger(__name__)


async def process_agent_astream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[dict, None]:
    """Yield each LangGraph stream event as a JSON-able dict in LangChain's native shape."""
    async for event in async_stream:
        mode, payload = event[0], event[1]
        if mode == "updates":
            # A gated tool call pauses the run; `__interrupt__` carries the approval request(s).
            if interrupts := payload.get("__interrupt__"):
                for it in interrupts:
                    yield {"type": "interrupt", "id": it.id, "value": it.value}
                continue
            for node_data in payload.values():
                messages = node_data.get("messages", []) if isinstance(node_data, dict) else []
                for msg in messages:
                    yield {"type": "message", "message": msg.model_dump()}
        elif mode == "messages":
            try:
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    yield {"type": "delta", "content": content, "id": chunk.id}
            except Exception:
                logger.exception("Error processing agent stream chunk")
