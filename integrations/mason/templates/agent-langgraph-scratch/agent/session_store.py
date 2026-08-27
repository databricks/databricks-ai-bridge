"""LangGraph conversation checkpointer for local and deployed execution."""

import os

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

_SESSION_STORE_ENV = "AGENT_SESSION_STORE"
_SHARED_PROJECT = "databricks-internal-lakebase-agent-session-store"
_DEFAULT_BRANCH = "production"
_CHECKPOINT_SCHEMA = "langgraph_checkpoints"

_saver: BaseCheckpointSaver | None = None


async def checkpointer() -> BaseCheckpointSaver:
    """Use memory locally and Lakebase when Mason wires a managed Session Store."""
    global _saver
    if _saver is None:
        _saver = await _durable_checkpointer() if os.getenv(_SESSION_STORE_ENV) else InMemorySaver()
    return _saver


async def _durable_checkpointer() -> BaseCheckpointSaver:
    from databricks_langchain.checkpoint import AsyncCheckpointSaver

    saver = AsyncCheckpointSaver(
        project=_SHARED_PROJECT,
        branch=_DEFAULT_BRANCH,
        schema=_CHECKPOINT_SCHEMA,
    )
    await saver.__aenter__()
    await saver.setup()
    return saver


def thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}
