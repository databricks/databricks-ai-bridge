"""Ask Genie One a question through a generated project's own adapter, and print what came back.

``test_genie_one.py`` runs this with the project's interpreter from the project's directory, so the
adapter reads that project's ``agent.toml`` and authenticates as the agent would. It prints one JSON
line, ``{"answer": ..., "rows": ...}``; the test asserts on it.

Usage: ``python genie_one_conversation.py langgraph|openai``
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

QUESTION = (
    "How many rows are in samples.nyctaxi.trips? "
    "Use only this sample table and include a SQL result."
)
# genie_ask returns while Genie is still answering; a SQL answer can take minutes.
ANSWER_TIMEOUT_SECONDS = 600

Call = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def _payload(value: Any) -> dict[str, Any]:
    """A Genie One tool's JSON result: its structured content, else its first JSON text block."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True)
    structured = value.get("structuredContent") or value.get("structured_content")
    if structured:
        return structured
    for block in value.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"No JSON result in the Genie One tool response: {value}")


async def _conversation(call: Call) -> dict[str, Any]:
    answer = await call("genie_ask", {"question": QUESTION})
    deadline = time.monotonic() + ANSWER_TIMEOUT_SECONDS
    while answer.get("status") == "in_progress" and time.monotonic() < deadline:
        await asyncio.sleep(5)
        answer = await call(
            "genie_poll_response",
            {"conversation_id": answer["conversation_id"], "response_id": answer["response_id"]},
        )
    rows = None
    if answer.get("status") == "completed" and answer.get("query_items"):
        rows = await call(
            "genie_get_query_result",
            {
                "conversation_id": answer["conversation_id"],
                "response_id": answer["response_id"],
                "item_id": answer["query_items"][0]["item_id"],
            },
        )
    return {"answer": answer, "rows": rows}


async def _langgraph() -> dict[str, Any]:
    from databricks_agentkit.langgraph.mcp import mcp_tools

    tools = {tool.name: tool for tool in await mcp_tools()}

    async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        message = await tools[name].ainvoke(
            {"type": "tool_call", "id": uuid.uuid4().hex, "name": name, "args": arguments}
        )
        return _payload({**(message.artifact or {}), "content": message.content})

    return await _conversation(call)


async def _openai() -> dict[str, Any]:
    from databricks_agentkit.openai.mcp import mcp_servers

    (server,) = await mcp_servers()
    async with server:

        async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            result = await server.call_tool(name, arguments)
            if result.isError:
                raise RuntimeError(f"{name} failed: {result}")
            return _payload(result)

        return await _conversation(call)


if __name__ == "__main__":
    conversation = {"langgraph": _langgraph, "openai": _openai}[sys.argv[1]]
    sys.stdout.write(json.dumps(asyncio.run(conversation()), default=str) + "\n")
