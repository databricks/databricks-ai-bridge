"""Small checkpointed LangGraph agent shared by both recovery examples."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator, Iterator
from typing import Any
from uuid import uuid4

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentStreamEvent,
    create_function_call_item,
    create_function_call_output_item,
    create_text_output_item,
    to_chat_completions_input,
)

MAX_WAIT_SECONDS = 300
RESUME_FROM_CHECKPOINT_KEY = "langgraph_resume_from_checkpoint"


@tool
async def wait_for_completion(seconds: int) -> str:
    """Wait for a bounded number of seconds before completing the request."""
    if seconds < 0 or seconds > MAX_WAIT_SECONDS:
        raise ValueError(f"seconds must be between 0 and {MAX_WAIT_SECONDS}")
    await asyncio.sleep(seconds)
    return f"Waited for {seconds} seconds."


def _create_agent(checkpointer: Any):
    return create_agent(
        model=ChatDatabricks(
            endpoint=os.getenv("DATABRICKS_MODEL", "databricks-gpt-5-2")
        ),
        tools=[wait_for_completion],
        system_prompt=(
            "Answer concisely. For a message beginning with PROPOSAL:, describe the "
            "planned action, end with APPROVAL_REQUIRED, and do not call tools. For a "
            "message beginning with APPROVED:, or any other message asking you to wait, "
            "always call wait_for_completion with the requested duration before "
            "confirming completion."
        ),
        checkpointer=checkpointer,
    )


def _thread_id(request: ResponsesAgentRequest) -> str:
    custom_inputs = dict(request.custom_inputs or {})
    session_key = custom_inputs.get("thread_id") or custom_inputs.get("session_id")
    if session_key:
        return str(session_key)
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    return str(uuid4())


def _request_input(request: ResponsesAgentRequest) -> dict[str, Any]:
    return {
        "messages": to_chat_completions_input(
            [item.model_dump(exclude_none=True) for item in request.input]
        )
    }


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return "".join(text_parts)
        return ""
    return ""


def _tool_output(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _response_items(message: BaseMessage) -> Iterator[dict[str, Any]]:
    if isinstance(message, ToolMessage):
        yield create_function_call_output_item(
            call_id=message.tool_call_id,
            output=_tool_output(message.content),
        )
        return

    if not isinstance(message, AIMessage):
        return

    for tool_call in message.tool_calls:
        arguments = tool_call.get("args", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, default=str)
        yield create_function_call_item(
            id=str(uuid4()),
            call_id=str(tool_call.get("id") or uuid4()),
            name=str(tool_call.get("name") or "unknown_tool"),
            arguments=arguments,
        )

    text = _message_text(message.content)
    if text:
        item = create_text_output_item(text=text, id=str(uuid4()))
        item["status"] = "completed"
        yield item


def _updated_messages(update: dict[str, Any]) -> Iterator[BaseMessage]:
    for node_update in update.values():
        if not isinstance(node_update, dict):
            continue
        messages = node_update.get("messages")
        if isinstance(messages, BaseMessage):
            yield messages
        elif isinstance(messages, (list, tuple)):
            yield from (
                message for message in messages if isinstance(message, BaseMessage)
            )


async def stream_agent(
    request: ResponsesAgentRequest,
    checkpointer: Any,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    agent = _create_agent(checkpointer)
    config = {"configurable": {"thread_id": _thread_id(request)}}
    graph_input: dict[str, Any] | None = _request_input(request)
    custom_inputs = dict(request.custom_inputs or {})
    if custom_inputs.get(RESUME_FROM_CHECKPOINT_KEY):
        checkpoint = await agent.aget_state(config)
        if checkpoint.next:
            graph_input = None

    async for update in agent.astream(
        graph_input,
        config,
        stream_mode="updates",
        durability="sync",
    ):
        for message in _updated_messages(update):
            for item in _response_items(message):
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=item,
                )
