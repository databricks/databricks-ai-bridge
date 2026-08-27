"""LangGraph agent hosted by the Databricks durable runtime."""

import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any

import uvicorn
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.durable_app import DatabricksDurableApp, DurableAgentContext
from databricks_langchain import ChatDatabricks
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessageChunk

from agent.session_store import checkpointer, thread_config
from agent.tools.sample_tool import get_current_time

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)
MODEL = "databricks-gpt-5-2"

app = DatabricksDurableApp()


def configure() -> None:
    """Validate Databricks auth before accepting traffic."""
    try:
        WorkspaceClient()
    except Exception as exc:
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
        target = f"profile {profile!r}" if profile else "default Databricks authentication"
        raise RuntimeError(
            f"Databricks auth is not configured for {target}. "
            "Run `databricks auth login --profile <name>` or update .env."
        ) from exc


async def create_agent_graph():
    """Build the agent with tools and the configured conversation checkpointer."""
    return create_agent(
        model=ChatDatabricks(endpoint=MODEL),
        tools=[get_current_time],
        checkpointer=await checkpointer(),
    )


@app.entrypoint
async def agent(request: dict, context: DurableAgentContext) -> dict:
    """Run one turn while the SDK owns HTTP, background work, and replay."""
    graph = await create_agent_graph()
    output = []
    stream = graph.astream(
        input={"messages": request.get("input") or []},
        config=thread_config(context.session_id),
        stream_mode=["updates", "messages"],
    )
    async for event in _serialize_events(stream):
        await context.emit(event)
        if event["type"] == "message":
            output.append(event["message"])
    return {"output": output, "session_id": context.session_id, "status": "completed"}


async def _serialize_events(async_stream: AsyncIterator[Any]) -> AsyncGenerator[dict, None]:
    """Convert LangGraph events to JSON messages and token deltas."""
    async for mode, payload in async_stream:
        if mode == "updates":
            for node_data in payload.values():
                messages = node_data.get("messages", []) if isinstance(node_data, dict) else []
                for message in messages:
                    yield {"type": "message", "message": message.model_dump()}
        elif mode == "messages":
            try:
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    yield {"type": "delta", "content": content, "id": chunk.id}
            except Exception:
                logger.exception("Error processing agent stream chunk")


def main() -> None:
    configure()
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", "8000")))
    uvicorn.run(app, host="0.0.0.0", port=port)
