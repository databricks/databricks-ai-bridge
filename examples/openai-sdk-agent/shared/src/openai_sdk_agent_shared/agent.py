"""Small OpenAI Agents SDK agent shared by both recovery examples."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from uuid import uuid4

from agents import Agent, Runner, function_tool
from agents.result import StreamEvent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentStreamEvent

MAX_WAIT_SECONDS = 300


@function_tool
async def wait_for_completion(seconds: int) -> str:
    """Wait for a bounded number of seconds before completing the request."""
    if seconds < 0 or seconds > MAX_WAIT_SECONDS:
        raise ValueError(f"seconds must be between 0 and {MAX_WAIT_SECONDS}")
    await asyncio.sleep(seconds)
    return f"Waited for {seconds} seconds."


def _create_agent() -> Agent:
    return Agent(
        name="Durable assistant",
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        instructions=(
            "Answer concisely. When the user asks you to wait, always call "
            "wait_for_completion with the requested duration before answering."
        ),
        tools=[wait_for_completion],
    )


def _request_input(request: ResponsesAgentRequest) -> list[dict[str, Any]]:
    return [item.model_dump(exclude_none=True) for item in request.input]


async def invoke_agent(request: ResponsesAgentRequest, session) -> list[dict[str, Any]]:
    result = await Runner.run(
        _create_agent(),
        _request_input(request),
        session=session,
    )
    return [item.to_input_item() for item in result.new_items]


async def _responses_events(
    events: AsyncIterator[StreamEvent],
) -> AsyncGenerator[ResponsesAgentStreamEvent | dict[str, Any], None]:
    current_item_id = str(uuid4())
    async for event in events:
        if event.type == "raw_response_event":
            event_data = event.data.model_dump()
            if event_data["type"] == "response.output_item.added":
                current_item_id = str(uuid4())
                event_data["item"]["id"] = current_item_id
            elif isinstance(event_data.get("item"), dict) and event_data["item"].get(
                "id"
            ):
                event_data["item"]["id"] = current_item_id
            elif event_data.get("item_id") is not None:
                event_data["item_id"] = current_item_id
            yield event_data
        elif (
            event.type == "run_item_stream_event"
            and event.item.type == "tool_call_output_item"
        ):
            output = event.item.to_input_item()
            if not isinstance(output.get("output"), str):
                try:
                    output["output"] = json.dumps(output.get("output"))
                except (TypeError, ValueError):
                    output["output"] = str(output.get("output"))
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done", item=output
            )


async def stream_agent(
    request: ResponsesAgentRequest,
    session,
) -> AsyncGenerator[ResponsesAgentStreamEvent | dict[str, Any], None]:
    result = Runner.run_streamed(
        _create_agent(),
        _request_input(request),
        session=session,
    )
    async for event in _responses_events(result.stream_events()):
        yield event
