from __future__ import annotations

from typing import Any, Callable, Sequence

from databricks.sdk import WorkspaceClient

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


DEFAULT_TOOL_RESUME_REPAIR_OUTPUT = (
    "Tool call was interrupted by a durable resume and did not complete. "
    "Please retry if still needed."
)


def build_tool_resume_repair(
    messages: Sequence[Any],
    synthetic_output: str = DEFAULT_TOOL_RESUME_REPAIR_OUTPUT,
) -> list[Any]:
    """Build synthetic ``ToolMessage`` responses for orphan tool calls.

    When a LangGraph run is killed mid-tool, the checkpointer preserves the
    trailing ``AIMessage.tool_calls`` but the paired ``ToolMessage``s never
    land. Replaying that state to the LLM on resume fails because the API
    (Anthropic in particular) requires every ``tool_use`` to be immediately
    followed by a matching ``tool_result``.

    Walks the trailing assistant turn (the last contiguous block of
    ``AIMessage`` / ``ToolMessage``) and returns a synthetic ``ToolMessage``
    for each ``tool_call`` id that lacks a matching
    ``ToolMessage.tool_call_id``. Appending the returned list via the
    ``add_messages`` reducer restores a valid conversation.

    Example::

        from databricks_langchain import build_tool_resume_repair

        state = await graph.aget_state(config)
        repair = build_tool_resume_repair(state.values.get("messages", []))
        if repair:
            await graph.aupdate_state(config, {"messages": repair})

    Args:
        messages: The current ``messages`` list from graph state.
        synthetic_output: Text for each injected ``ToolMessage.content``.

    Returns:
        A list of ``ToolMessage`` instances (possibly empty). Empty means
        the state is already consistent — no repair needed.
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
    return [ToolMessage(tool_call_id=tc_id, content=synthetic_output) for tc_id in orphans]


def build_tool_resume_repair_pre_model_hook(
    synthetic_output: str = DEFAULT_TOOL_RESUME_REPAIR_OUTPUT,
) -> Callable[[dict], dict]:
    """Return a LangGraph ``pre_model_hook`` that repairs orphan tool calls.

    Wires ``build_tool_resume_repair`` into the graph as a pre-model hook so
    durable-resume recovery happens automatically before every LLM call. Keeps
    repair logic off the handler — callers only add one argument to
    ``create_agent``.

    Usage::

        from databricks_langchain import build_tool_resume_repair_pre_model_hook

        agent = create_agent(
            model=model,
            tools=tools,
            checkpointer=checkpointer,
            pre_model_hook=build_tool_resume_repair_pre_model_hook(),
        )

    The hook fires on every model turn and is a no-op when state is clean, so
    the happy path is free. On a mid-tool crash-resume, it injects synthetic
    ``ToolMessage``s for any ``AIMessage.tool_calls`` in the trailing turn
    whose paired ``ToolMessage`` never landed. Satisfies Anthropic's
    ``tool_use`` ⇄ ``tool_result`` contract without needing manual
    ``aupdate_state(..., as_node="tools")`` surgery.

    Args:
        synthetic_output: Text for each injected ``ToolMessage.content``.

    Returns:
        A callable suitable to pass as ``pre_model_hook`` to
        ``langchain.agents.create_agent`` (or ``create_react_agent``).
    """

    def _hook(state: dict) -> dict:
        repair = build_tool_resume_repair(
            state.get("messages", []), synthetic_output=synthetic_output
        )
        return {"messages": repair} if repair else {}

    return _hook


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
