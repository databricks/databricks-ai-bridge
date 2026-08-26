"""Minimal OpenAI Agents SDK loop with persistent session state."""

import os
from collections.abc import Awaitable, Callable
from typing import Any

from agents import Agent, Runner
from databricks_openai.agents import AsyncDatabricksSession

EventEmitter = Callable[[dict[str, Any]], Awaitable[int]]

assistant = Agent(
    name="Durable assistant",
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    instructions="Answer concisely.",
)


async def run_openai_agent(prompt: str, session_id: str, emit: EventEmitter) -> str:
    session = AsyncDatabricksSession(
        session_id=session_id,
        autoscaling_endpoint=os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"],
        schema=os.getenv("OPENAI_AGENT_SESSION_SCHEMA", "durable_openai_agent_sessions"),
    )
    result = Runner.run_streamed(assistant, input=prompt, session=session)

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            await emit(event.data.model_dump(mode="json"))

    return str(result.final_output or "")
