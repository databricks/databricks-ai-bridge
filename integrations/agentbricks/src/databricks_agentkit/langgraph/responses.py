"""Collect per-invocation LangGraph responses without replaying saved messages to clients."""

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import get_checkpoint_id

from databricks_agentkit.langgraph.session_store import invocation_id_from_metadata


async def checkpointed_messages(
    graph: Any, config: RunnableConfig, invocation_id: str
) -> list[BaseMessage]:
    """Restore this invocation's committed message updates, excluding earlier turns.

    Final graph values may contain earlier turns or reducer-replaced messages. Read the original
    task outputs instead. Checkpoint history and task writes must be retained during recovery.
    """
    child = None
    steps = []
    async for state in graph.aget_state_history(config):
        if child is not None:
            if get_checkpoint_id(state.config) != get_checkpoint_id(child.parent_config):
                continue
            # Parent task results are committed by the child. The latest pending work remains
            # LangGraph's responsibility when the caller resumes execution.
            if child.metadata and child.metadata.get("source") == "loop":
                steps.append(
                    [
                        message
                        for task in state.tasks
                        if task.name != "__start__" and isinstance(task.result, dict)
                        for message in task.result.get("messages", [])
                    ]
                )
        if invocation_id_from_metadata(state.metadata) != invocation_id or not state.parent_config:
            break
        child = state
    return [message for step in reversed(steps) for message in step]


async def collect_response(
    events: AsyncIterator[Any],
    emit: Callable[[dict[str, Any]], Awaitable[object]],
    restored_messages: Sequence[BaseMessage] = (),
) -> dict[str, Any]:
    """Return saved and new output while emitting only the new stream events.

    This preserves the message-oriented template's response format. It does not reconcile gaps
    between graph checkpoints and Runtime Store events or promise exactly-once event delivery.
    """
    outputs = [
        {"type": "message", "message": message.model_dump()} for message in restored_messages
    ]
    async for event in _serialize_events(events):
        await emit(event)
        if event.get("type") in ("message", "interrupt"):
            outputs.append(event)

    interrupted = bool(outputs and outputs[-1].get("type") == "interrupt")
    return {
        "output": [event["message"] if event["type"] == "message" else event for event in outputs],
        "status": "interrupted" if interrupted else "completed",
    }


async def _serialize_events(events: AsyncIterator[Any]) -> AsyncIterator[dict[str, Any]]:
    async for mode, payload in events:
        if mode == "updates":
            if interrupts := payload.get("__interrupt__"):
                for item in interrupts:
                    yield {"type": "interrupt", "id": item.id, "value": item.value}
                continue
            for node_data in payload.values():
                messages = node_data.get("messages", []) if isinstance(node_data, dict) else []
                for message in messages:
                    yield {"type": "message", "message": message.model_dump()}
        elif mode == "messages":
            try:
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    yield {"type": "delta", "content": content, "id": chunk.id}
            except (KeyError, IndexError, TypeError):
                continue
