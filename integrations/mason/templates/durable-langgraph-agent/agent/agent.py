import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.messages import AIMessageChunk

from databricks_mason import (
    DurableAgentContext,
    configure_tracing,
    tag_session,
    workspace_client,
    workspace_headers,
)

from agent.tools import all_tools

logger = logging.getLogger(__name__)

MODEL = "databricks-gpt-5-2"


class _RoutedChatDatabricks(ChatDatabricks):
    """Forward account-host workspace routing to the underlying OpenAI clients."""

    def _get_client_kwargs(self) -> dict[str, Any]:
        kwargs = super()._get_client_kwargs()
        if headers := workspace_headers():
            kwargs["default_headers"] = headers
        return kwargs


def configure() -> None:
    """Configure optional tracing after the project environment is loaded."""
    configure_tracing()


async def create_agent_graph():
    """Build the same model-and-tools LangGraph agent used by the standard template."""
    return create_agent(
        model=_RoutedChatDatabricks(endpoint=MODEL, workspace_client=workspace_client()),
        tools=all_tools(),
    )


async def run_agent(agent_input: object, context: DurableAgentContext) -> dict:
    """Run one durable agent attempt and persist its streaming events."""
    if not isinstance(agent_input, list):
        raise ValueError("input must be a list of message objects")

    tag_session(context.session_id)
    agent = await create_agent_graph()
    output = []
    async for event in _serialize_events(
        agent.astream(
            input={"messages": agent_input},
            stream_mode=["updates", "messages"],
        )
    ):
        await context.emit(event)
        if event.get("type") == "message":
            output.append(event["message"])

    return {
        "output": output,
        "session_id": context.session_id,
        "recovered": context.is_recovery,
    }


async def _serialize_events(async_stream: AsyncIterator[Any]) -> AsyncGenerator[dict, None]:
    """Convert LangGraph updates and message chunks into persisted JSON events."""
    async for mode, payload in async_stream:
        if mode == "updates":
            for node_data in payload.values():
                messages = node_data.get("messages", []) if isinstance(node_data, dict) else []
                for message in messages:
                    yield {"type": "message", "message": message.model_dump(mode="json")}
        elif mode == "messages":
            try:
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield {
                        "type": "delta",
                        "content": chunk.content,
                        "id": chunk.id,
                    }
            except Exception:
                logger.exception("Error processing agent stream chunk")
