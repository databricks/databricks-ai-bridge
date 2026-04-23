from __future__ import annotations

import copy
import logging
from typing import Any, Sequence

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.long_running.repair import DEFAULT_SYNTHETIC_INTERRUPTED_OUTPUT

logger = logging.getLogger(__name__)

try:
    from databricks_ai_bridge.lakebase import AsyncLakebasePool, LakebasePool
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _checkpoint_imports_available = True
except ImportError:
    PostgresSaver = object  # type: ignore
    AsyncPostgresSaver = object  # type: ignore

    _checkpoint_imports_available = False

try:
    from langchain_core.messages import AIMessage, ToolMessage

    _message_imports_available = True
except ImportError:
    AIMessage = object  # type: ignore
    ToolMessage = object  # type: ignore
    _message_imports_available = False


def _build_tool_resume_repair(messages: Sequence[Any]) -> list[Any]:
    """Build synthetic ``ToolMessage`` responses for orphan tool calls.

    Internal helper used by ``_repair_loaded_checkpoint_tuple``. When a
    LangGraph run is killed mid-tool, the checkpointer preserves the
    trailing ``AIMessage.tool_calls`` but the paired ``ToolMessage``s
    never land. Replaying that state to the LLM fails because the API
    (Anthropic in particular) requires every ``tool_use`` to be
    immediately followed by a matching ``tool_result``.

    Walks the trailing assistant turn (the last contiguous block of
    ``AIMessage`` / ``ToolMessage``) and returns a synthetic
    ``ToolMessage`` for each ``tool_call`` id that lacks a matching
    ``ToolMessage.tool_call_id``. The caller appends these to the
    ``messages`` channel before the next model call.
    """
    if not _message_imports_available or not messages:
        return []

    # Trailing assistant turn: walk backwards until we hit a non-assistant/
    # non-tool message. That block is the "pending" turn whose tool_use ↔
    # tool_result pairing we need to enforce.
    trailing_start = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], (AIMessage, ToolMessage)):
            trailing_start = i
        else:
            break

    tool_call_ids: list[str] = []
    answered: set[str] = set()
    for msg in messages[trailing_start:]:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id and tc_id not in tool_call_ids:
                    tool_call_ids.append(tc_id)
        elif isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid:
                answered.add(tcid)

    orphans = [tc_id for tc_id in tool_call_ids if tc_id not in answered]
    return [
        ToolMessage(tool_call_id=tc_id, content=DEFAULT_SYNTHETIC_INTERRUPTED_OUTPUT)
        for tc_id in orphans
    ]


def _repair_loaded_checkpoint_tuple(tup: Any) -> Any:
    """Return a copy of ``tup`` with orphan tool_calls in its ``messages``
    channel closed by synthetic ``ToolMessage`` s.

    Called on every ``(a)get_tuple`` to make the served checkpoint
    protocol-valid (every ``tool_use`` paired with a ``tool_result``)
    transparently. A kill between the ``model`` and ``tools`` nodes leaves
    the trailing ``AIMessage.tool_calls`` unpaired; on the NEXT turn that
    state would otherwise leak into the LLM and be rejected by the
    provider's pairing check.

    Idempotent — ``_build_tool_resume_repair`` is a no-op when state is
    already clean. Cheap — the walk is O(trailing-turn).

    Side effect: the synthetic ``ToolMessage`` s added here become part of
    the state LangGraph writes on the NEXT node boundary, so the repair
    self-heals the DB row over time rather than re-computing on every read.
    """
    if tup is None or not _message_imports_available:
        return tup

    checkpoint = getattr(tup, "checkpoint", None)
    if not isinstance(checkpoint, dict):
        return tup
    channel_values = checkpoint.get("channel_values")
    if not isinstance(channel_values, dict):
        return tup
    messages = channel_values.get("messages")
    if not isinstance(messages, list) or not messages:
        return tup

    repair = _build_tool_resume_repair(messages)
    if not repair:
        return tup

    logger.info(
        "[durable] checkpoint read-time repair: injected %d synthetic ToolMessage(s)",
        len(repair),
    )
    new_checkpoint = copy.copy(checkpoint)
    new_checkpoint["channel_values"] = dict(channel_values)
    new_checkpoint["channel_values"]["messages"] = list(messages) + list(repair)
    return tup._replace(checkpoint=new_checkpoint)


class CheckpointSaver(PostgresSaver):
    """
    LangGraph PostgresSaver using a Lakebase connection pool.

    Supports two modes: Lakebase Provisioned VS Autoscaling
    https://docs.databricks.com/aws/en/oltp/#feature-comparison
    """

    def __init__(
        self,
        *,
        instance_name: str | None = None,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        **pool_kwargs: Any,
    ) -> None:
        # Lazy imports
        if not _checkpoint_imports_available:
            raise ImportError(
                "CheckpointSaver requires databricks-langchain[memory]. "
                "Please install with: pip install databricks-langchain[memory]"
            )

        self._lakebase: LakebasePool = LakebasePool(
            instance_name=instance_name,
            autoscaling_endpoint=autoscaling_endpoint,
            project=project,
            branch=branch,
            workspace_client=workspace_client,
            **dict(pool_kwargs),
        )
        super().__init__(self._lakebase.pool)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close the connection pool."""
        self._lakebase.close()
        return False

    def get_tuple(self, config):
        """Return the checkpoint tuple, with trailing orphan tool_calls paired."""
        return _repair_loaded_checkpoint_tuple(super().get_tuple(config))


class AsyncCheckpointSaver(AsyncPostgresSaver):
    """
    Async LangGraph PostgresSaver using a Lakebase connection pool.

    Supports two modes: Lakebase Provisioned VS Autoscaling
    https://docs.databricks.com/aws/en/oltp/#feature-comparison

    Checkpoint tables are created automatically when entering the context manager.
    """

    def __init__(
        self,
        *,
        instance_name: str | None = None,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        **pool_kwargs: Any,
    ) -> None:
        # Lazy imports
        if not _checkpoint_imports_available:
            raise ImportError(
                "AsyncCheckpointSaver requires databricks-langchain[memory]. "
                "Please install with: pip install databricks-langchain[memory]"
            )

        self._lakebase: AsyncLakebasePool = AsyncLakebasePool(
            instance_name=instance_name,
            autoscaling_endpoint=autoscaling_endpoint,
            project=project,
            branch=branch,
            workspace_client=workspace_client,
            **dict(pool_kwargs),
        )
        super().__init__(self._lakebase.pool)

    async def __aenter__(self):
        """Enter async context manager and open the connection pool."""
        await self._lakebase.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager and close the connection pool."""
        await self._lakebase.close()
        return False

    async def aget_tuple(self, config):
        """Return the checkpoint tuple, with trailing orphan tool_calls paired."""
        return _repair_loaded_checkpoint_tuple(await super().aget_tuple(config))
