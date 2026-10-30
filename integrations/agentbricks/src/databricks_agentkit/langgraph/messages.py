"""Serialize LangGraph messages for invocation output, stream deltas, and session history.

Claude 5.x replies carry reasoning blocks (summary text and a signature) before the answer text.
Callers read the answer from a message's ``content``, so reasoning becomes a separate item shaped
like the OpenAI template's: ``{"role": "assistant", "content": [<reasoning block>, ...]}``.
Checkpoints keep the original blocks so the model receives each signature on the next turn.
"""

import json
from typing import Any

_LEGACY_REASONING_PREFIX = '[{"type": "reasoning"'


def message_items(message: Any) -> list[dict[str, Any]]:
    """Return a message as output items: its reasoning item, if any, then the message itself.

    The message keeps every field of ``model_dump()``; only an AI message whose content holds
    reasoning blocks changes, to a separate reasoning item and the answer as a string.
    """
    data = message.model_dump() if hasattr(message, "model_dump") else message
    if not isinstance(data, dict) or data.get("type") not in ("ai", "AIMessageChunk"):
        return [data]
    reasoning, text = _split(data.get("content"))
    if not reasoning:
        return [data]
    return [{"role": "assistant", "content": reasoning}, {**data, "content": text}]


def delta_contents(content: Any) -> list[str | list[dict[str, Any]]]:
    """Return a streamed chunk's delta payloads: reasoning blocks, then the answer text.

    Signature-only reasoning chunks carry no readable text and are skipped; the final message
    item still has the signature.
    """
    reasoning, text = _split(content)
    if isinstance(text, list) and all(_is_text(block) for block in text):
        # A Claude 5.x answer chunk after its reasoning: a text block.
        text = "".join(
            block if isinstance(block, str) else block.get("text") or "" for block in text
        )
    readable = [block for block in reasoning if _reasoning_text(block)]
    return [part for part in (readable, text) if part]


def gateway_messages(messages: list[Any]) -> list[Any]:
    """Return ``messages`` with JSON-encoded Claude reasoning restored to gateway content blocks.

    databricks-langchain 0.21.0 checkpoints Claude 5.x replies as one string. Sending that back as
    text drops each reasoning signature, which Claude needs after a tool call. Call this on the
    messages a chat model sends; other messages, and list content, pass through unchanged.
    """
    restored = []
    for message in messages:
        content = getattr(message, "content", None)
        if getattr(message, "type", None) == "ai" and isinstance(content, str):
            blocks = _parse_legacy_reasoning(content)
            if isinstance(blocks, list):
                message = message.model_copy(update={"content": blocks})
        restored.append(message)
    return restored


def _is_text(block: Any) -> bool:
    return isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")


def _split(content: Any) -> tuple[list[dict[str, Any]], Any]:
    if isinstance(content, str):
        content = _parse_legacy_reasoning(content)
    if not isinstance(content, list):
        return [], content
    reasoning = [
        _without_index(block)
        for block in content
        if isinstance(block, dict) and block.get("type") == "reasoning"
    ]
    if not reasoning:
        return [], content
    text = "".join(
        block if isinstance(block, str) else block.get("text") or ""
        for block in content
        if _is_text(block)
    )
    return reasoning, text


def _parse_legacy_reasoning(content: str) -> str | list[Any]:
    """Recover blocks from databricks-langchain 0.21.0, which JSON-encodes each reasoning chunk.

    Its streamed reply reads ``[{"type": "reasoning", ...}][...]Hello``: one JSON list per chunk,
    concatenated before the answer text. Only Claude reasoning, whose summaries carry a
    ``signature``, is recovered; other models' JSON content (gpt-oss) stays as it is.
    """
    if not content.startswith(_LEGACY_REASONING_PREFIX):
        return content
    decoder = json.JSONDecoder()
    blocks: list[Any] = []
    position = 0
    while content.startswith(_LEGACY_REASONING_PREFIX, position):
        try:
            parsed, position = decoder.raw_decode(content, position)
        except json.JSONDecodeError:
            return content
        if not isinstance(parsed, list) or not all(_is_claude_block(block) for block in parsed):
            return content
        blocks.extend(parsed)
    blocks = _merge_reasoning(blocks)
    if position < len(content):
        blocks.append({"type": "text", "text": content[position:]})
    return blocks


def _merge_reasoning(blocks: list[Any]) -> list[Any]:
    """Join streamed reasoning fragments into one block per signature, as the gateway sent it."""
    merged: list[Any] = []
    text = ""
    for block in blocks:
        if block.get("type") != "reasoning":
            merged.append(block)
            continue
        for part in block["summary"]:
            text += part.get("text") or ""
            if part.get("signature"):
                merged.append(_reasoning_block(text, part["signature"]))
                text = ""
    if text:
        merged.append(_reasoning_block(text, ""))
    return merged


def _reasoning_block(text: str, signature: str) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text, "signature": signature}],
    }


def _is_claude_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("type") == "text":
        return True
    summary = block.get("summary")
    return (
        block.get("type") == "reasoning"
        and isinstance(summary, list)
        and all(isinstance(part, dict) and "signature" in part for part in summary)
    )


def _without_index(block: dict[str, Any]) -> dict[str, Any]:
    """Drop the stream-merge ``index`` keys LangChain chunks carry."""
    block = {key: value for key, value in block.items() if key != "index"}
    if isinstance(block.get("summary"), list):
        block["summary"] = [
            {key: value for key, value in part.items() if key != "index"}
            if isinstance(part, dict)
            else part
            for part in block["summary"]
        ]
    return block


def _reasoning_text(block: dict[str, Any]) -> str:
    return "".join(
        part.get("text") or "" for part in block.get("summary") or [] if isinstance(part, dict)
    )
