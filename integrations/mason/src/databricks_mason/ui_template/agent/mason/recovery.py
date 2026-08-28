"""Checkpointed tool sequence for demonstrating crash recovery."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypedDict

from agent.mason.session_store import checkpointer, thread_config  # ty: ignore[unresolved-import]
from agent.tools.long_running import (
    process_id,
    step_seconds,
    tool_step_1,
    tool_step_2,
    tool_step_3,
    tool_step_4,
)
from langgraph.checkpoint.base import BaseCheckpointSaver  # ty: ignore[unresolved-import]
from langgraph.graph import END, START, StateGraph  # ty: ignore[unresolved-import]

StepRunner = Callable[[], Awaitable[dict[str, Any]]]
StepDefinition = tuple[str, StepRunner]

_STEP_TOOLS = (tool_step_1, tool_step_2, tool_step_3, tool_step_4)
_active_tasks: dict[str, asyncio.Task[Any]] = {}
_task_errors: dict[str, str] = {}
_compiled_graph: Any | None = None


class RecoveryState(TypedDict, total=False):
    session_id: str
    outputs: list[dict[str, Any]]


def step_names() -> list[str]:
    return [tool.name for tool in _STEP_TOOLS]


def recovery_config(session_id: str) -> dict[str, Any]:
    """Map the public session id onto a separate durable workflow thread."""
    actor_id = thread_config(session_id)["configurable"]["actor_id"]
    thread_id = hashlib.sha256(f"mason-demo-recovery\x1f{session_id}".encode()).hexdigest()
    return {"configurable": {"thread_id": thread_id, "actor_id": actor_id}}


def _tool_runner(tool: Any) -> StepRunner:
    async def run() -> dict[str, Any]:
        result = await tool.ainvoke({})
        if isinstance(result, dict):
            return result
        return {"tool": tool.name, "output": str(result), "instance_id": process_id()}

    return run


def _step_node(runner: StepRunner):
    async def run(state: RecoveryState) -> RecoveryState:
        result = await runner()
        return {"outputs": [*state.get("outputs", []), result]}

    return run


def build_recovery_graph(
    saver: BaseCheckpointSaver,
    steps: Sequence[StepDefinition] | None = None,
):
    """Build the sequential graph; exposed for the checkpoint-boundary recovery test."""
    definitions = list(steps or [(tool.name, _tool_runner(tool)) for tool in _STEP_TOOLS])
    graph = StateGraph(RecoveryState)
    for name, runner in definitions:
        graph.add_node(name, _step_node(runner))
    graph.add_edge(START, definitions[0][0])
    for current, following in zip(definitions, definitions[1:], strict=False):
        graph.add_edge(current[0], following[0])
    graph.add_edge(definitions[-1][0], END)
    return graph.compile(checkpointer=saver)


def _graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_recovery_graph(checkpointer())
    return _compiled_graph


def _task_active(session_id: str) -> bool:
    task = _active_tasks.get(session_id)
    return bool(task and not task.done())


def _record_task_result(session_id: str, task: asyncio.Task[Any]) -> None:
    if _active_tasks.get(session_id) is task:
        _active_tasks.pop(session_id, None)
    if task.cancelled():
        return
    try:
        task.result()
    except Exception as error:
        _task_errors[session_id] = str(error)


def _launch(session_id: str, graph_input: RecoveryState | None) -> None:
    _task_errors.pop(session_id, None)
    task = asyncio.create_task(_graph().ainvoke(graph_input, config=recovery_config(session_id)))
    _active_tasks[session_id] = task
    task.add_done_callback(lambda completed: _record_task_result(session_id, completed))


async def status(session_id: str) -> dict[str, Any]:
    snapshot = await _graph().aget_state(recovery_config(session_id))
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    outputs = list(values.get("outputs") or [])
    next_steps = list(snapshot.next)
    worker_active = _task_active(session_id)
    error = _task_errors.get(session_id)
    if error:
        state = "failed"
    elif worker_active:
        state = "running"
    elif next_steps:
        state = "waiting_for_resume"
    elif len(outputs) == len(_STEP_TOOLS):
        state = "completed"
    else:
        state = "not_started"
    current_step = next_steps[0] if next_steps else None
    if worker_active and current_step is None and len(outputs) < len(_STEP_TOOLS):
        current_step = step_names()[len(outputs)]
    return {
        "session_id": session_id,
        "status": state,
        "steps": step_names(),
        "outputs": outputs,
        "current_step": current_step,
        "worker_active": worker_active,
        "needs_resume": bool(next_steps and not worker_active and not error),
        "error": error,
        "instance_id": process_id(),
        "step_seconds": step_seconds(),
    }


async def start(session_id: str) -> dict[str, Any]:
    current = await status(session_id)
    if current["status"] != "not_started":
        return current
    _launch(session_id, {"session_id": session_id, "outputs": []})
    await asyncio.sleep(0)
    return await status(session_id)


async def resume(session_id: str) -> dict[str, Any]:
    current = await status(session_id)
    if current["worker_active"] or current["status"] == "completed":
        return current
    if not current["needs_resume"]:
        raise ValueError(f"No incomplete recovery sequence exists for session {session_id!r}.")
    _launch(session_id, None)
    await asyncio.sleep(0)
    return await status(session_id)
