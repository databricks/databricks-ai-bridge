"""Restore per-invocation LangGraph responses from durable checkpoints."""

from typing import Any

from langchain_core.messages import BaseMessage
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
