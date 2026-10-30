"""Serialize LangGraph messages for invocation output, stream deltas, and session history.

Reasoning models (Claude 5.x, Qwen, Gemini, gpt-oss) reply with reasoning blocks before the answer
text; Claude's carry a signature. Callers read the answer from a message's ``content``, so reasoning
becomes a separate item shaped like the OpenAI template's:
``{"role": "assistant", "content": [<reasoning block>, ...]}``. Checkpoints keep the original
blocks so the model receives each signature on the next turn.
"""

import json
from typing import Any

_LEGACY_REASONING_PREFIX = '[{"type": "reasoning"'
_LEGACY_PREFIXES = (_LEGACY_REASONING_PREFIX, '[{"type": "text"')


def message_items(message: Any) -> list[dict[str, Any]]:
    """Return a message as output items: its reasoning item, if any, then the message itself.

    The message keeps every field of ``model_dump()``; only an AI message whose content is a list
    of reasoning or text blocks changes, to a separate reasoning item and the answer as a string.
    """
    data = message.model_dump() if hasattr(message, "model_dump") else message
    if not isinstance(data, dict) or data.get("type") not in ("ai", "AIMessageChunk"):
        return [data]
    content = data.get("content")
    reasoning, text = _split(content)
    if text is content:
        # Nothing to split: a plain string, or a list with other parts (e.g. images).
        return [data]
    answer = {**data, "content": text}
    return [{"role": "assistant", "content": reasoning}, answer] if reasoning else [answer]


def delta_contents(content: Any) -> list[str | list[dict[str, Any]]]:
    """Return a streamed chunk's delta payloads: reasoning blocks, then the answer text.

    Signature-only reasoning chunks carry no readable text and are skipped; the final message
    item still has the signature.
    """
    reasoning, text = _split(content)
    readable = [block for block in reasoning if _reasoning_text(block)]
    return [part for part in (readable, text) if part]


def gateway_messages(messages: list[Any]) -> list[Any]:
    """Return ``messages`` with JSON-encoded reasoning replies restored for the next request.

    databricks-langchain 0.21.0 checkpoints reasoning replies as one JSON string. Sending that back
    as text drops each Claude reasoning signature, which Claude needs after a tool call, and sends
    other models their own reasoning as answer text. Signed reasoning goes back as gateway content
    blocks; unsigned reasoning (Qwen, gpt-oss) is dropped, leaving the answer text. Call this on
    the messages a chat model sends; other messages, and list content, pass through unchanged.
    """
    restored = []
    for message in messages:
        content = getattr(message, "content", None)
        if getattr(message, "type", None) == "ai" and isinstance(content, str):
            blocks = _parse_legacy_reasoning(content)
            if isinstance(blocks, list):
                message = message.model_copy(update={"content": _signed_only(blocks)})
        restored.append(message)
    return restored


def _is_text(block: Any) -> bool:
    return isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")


def _is_reasoning(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "reasoning"


def _split(content: Any) -> tuple[list[dict[str, Any]], Any]:
    """Split content into reasoning blocks and answer text; other content comes back as is."""
    if isinstance(content, str):
        content = _parse_legacy_reasoning(content)
        if content == []:
            # Only empty reasoning fragments: no answer text yet.
            return [], ""
    if (
        not isinstance(content, list)
        or not content
        or not all(_is_text(block) or _is_reasoning(block) for block in content)
    ):
        return [], content
    reasoning = [_without_index(block) for block in content if _is_reasoning(block)]
    text = "".join(
        block if isinstance(block, str) else block.get("text") or ""
        for block in content
        if _is_text(block)
    )
    return reasoning, text


def _parse_legacy_reasoning(content: str) -> str | list[Any]:
    """Recover blocks from databricks-langchain 0.21.0, which JSON-encodes each list chunk.

    Its streamed reply reads ``[{"type": "reasoning", ...}][...]Hello``: one JSON list per chunk,
    concatenated before the answer text. Only content that starts with reasoning is recovered.
    """
    if not content.startswith(_LEGACY_REASONING_PREFIX):
        return content
    decoder = json.JSONDecoder()
    blocks: list[Any] = []
    position = 0
    while content.startswith(_LEGACY_PREFIXES, position):
        try:
            parsed, position = decoder.raw_decode(content, position)
        except json.JSONDecodeError:
            return content
        if not isinstance(parsed, list) or not all(_is_gateway_block(block) for block in parsed):
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
    # Claude parts always have a signature key (empty until the block ends); other models' don't.
    signature: str | None = None
    for block in blocks:
        if block.get("type") != "reasoning":
            if text:
                merged.append(_reasoning_block(text, signature))
                text, signature = "", None
            merged.append(block)
            continue
        for part in block["summary"]:
            text += part.get("text") or ""
            if "signature" in part:
                signature = part["signature"] or ""
            if part.get("signature"):
                merged.append(_reasoning_block(text, signature))
                text, signature = "", None
    if text:
        merged.append(_reasoning_block(text, signature))
    return merged


def _reasoning_block(text: str, signature: str | None) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": "summary_text", "text": text}
    if signature is not None:
        summary["signature"] = signature
    return {"type": "reasoning", "summary": [summary]}


def _signed_only(blocks: list[Any]) -> str | list[Any]:
    """Drop unsigned reasoning; with no signed reasoning left, return the answer as a string."""
    signed = [block for block in blocks if not _is_reasoning(block) or _is_signed(block)]
    if any(_is_reasoning(block) for block in signed):
        return signed
    return "".join(block.get("text") or "" for block in signed if _is_text(block))


def _is_signed(block: dict[str, Any]) -> bool:
    return any(isinstance(part, dict) and part.get("signature") for part in block["summary"])


def _is_gateway_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("type") == "text":
        return True
    summary = block.get("summary")
    return (
        block.get("type") == "reasoning"
        and isinstance(summary, list)
        and all(isinstance(part, dict) for part in summary)
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
