"""Normalize native Agent Framework message content for Agent Bricks clients."""

import json
from typing import Any

from agent_framework import Message


def serialize_message(message: Message) -> dict[str, Any]:
    """Retain text, tool calls, and tool results in the common browser/wire shape."""
    result: dict[str, Any] = {"role": message.role, "content": message.text}
    calls = []
    tool_results = []
    for content in message.contents:
        if content.type == "function_call":
            arguments = content.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append({"id": content.call_id, "name": content.name, "args": arguments})
        elif content.type == "function_result":
            tool_results.append(str(content.result))
    if calls:
        result["tool_calls"] = calls
    if tool_results:
        result["role"] = "tool"
        result["content"] = "\n".join(filter(None, [message.text, *tool_results]))
        if message.author_name:
            result["name"] = message.author_name
    return result
