"""Checkpointed tool sequence with manual heartbeat-based recovery."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from databricks_mason.runtime.durability import (
    DurabilityLease,
    SessionStoreDurabilityLog,
    execution_id,
    heartbeat_seconds,
    stale_seconds,
)
from databricks_mason.runtime.long_running import (
    process_id,
    step_seconds,
    tool_step_1,
    tool_step_2,
    tool_step_3,
    tool_step_4,
)
from databricks_mason.runtime.session_store import (
    checkpointer,
    thread_config,
)

StepRunner = Callable[[], Awaitable[dict[str, Any]]]
StepDefinition = tuple[str, StepRunner]

_STEP_TOOLS = (tool_step_1, tool_step_2, tool_step_3, tool_step_4)
_active_tasks: dict[str, asyncio.Task[Any]] = {}
_task_errors: dict[str, str] = {}
_compiled_graph: Any | None = None
_durability: SessionStoreDurabilityLog | None = None

logger = logging.getLogger(__name__)


class RecoveryState(TypedDict, total=False):
    session_id: str
    outputs: list[dict[str, Any]]


def step_names() -> list[str]:
    return [tool.name for tool in _STEP_TOOLS]


def recovery_config(session_id: str) -> dict[str, Any]:
    """Map the public session id onto a separate durable workflow thread."""
    configurable = thread_config(session_id)["configurable"]
    actor_id = configurable.get("actor_id") or os.getenv("AGENT_SESSION_ACTOR_ID") or session_id
    return {"configurable": {"thread_id": execution_id(session_id), "actor_id": actor_id}}


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
) -> Any:
    """Build the sequential graph; exposed for the checkpoint-boundary recovery test."""
    definitions = list(steps or [(tool.name, _tool_runner(tool)) for tool in _STEP_TOOLS])
    graph: StateGraph[RecoveryState] = StateGraph(RecoveryState)  # type: ignore[assignment]
    for name, runner in definitions:
        graph.add_node(name, _step_node(runner))
    graph.add_edge(START, definitions[0][0])
    for current, following in zip(definitions, definitions[1:], strict=False):
        graph.add_edge(current[0], following[0])
    graph.add_edge(definitions[-1][0], END)
    return graph.compile(checkpointer=saver)


async def _graph():
    global _compiled_graph
    if _compiled_graph is None:
        saver = checkpointer()
        if inspect.isawaitable(saver):
            saver = await saver
        _compiled_graph = build_recovery_graph(saver)  # type: ignore[arg-type]
    return _compiled_graph


def _durability_log() -> SessionStoreDurabilityLog:
    global _durability
    if _durability is None:
        _durability = SessionStoreDurabilityLog()
    return _durability


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


async def _heartbeat_loop(lease: DurabilityLease) -> None:
    while True:
        await asyncio.sleep(heartbeat_seconds())
        try:
            if not await _durability_log().heartbeat(lease):
                return
        except Exception:
            logger.exception("Failed to persist durability heartbeat for %s", lease.execution_id)


async def _run_attempt(
    session_id: str,
    graph_input: RecoveryState | None,
    lease: DurabilityLease,
) -> None:
    heartbeat = asyncio.create_task(_heartbeat_loop(lease))
    try:
        graph = await _graph()
        await graph.ainvoke(graph_input, config=recovery_config(session_id))
    except asyncio.CancelledError:
        raise
    except Exception as error:
        try:
            await _durability_log().fail(lease, str(error))
        except Exception:
            logger.exception("Failed to persist failed durability attempt %s", lease.execution_id)
        raise
    else:
        try:
            await _durability_log().complete(lease)
        except Exception:
            logger.exception(
                "Failed to persist completed durability attempt %s", lease.execution_id
            )
    finally:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)


async def _launch(session_id: str, graph_input: RecoveryState | None) -> None:
    _task_errors.pop(session_id, None)
    lease = await _durability_log().claim(session_id, process_id())
    task = asyncio.create_task(_run_attempt(session_id, graph_input, lease))
    _active_tasks[session_id] = task
    task.add_done_callback(lambda completed: _record_task_result(session_id, completed))


async def status(session_id: str) -> dict[str, Any]:
    graph = await _graph()
    snapshot = await graph.aget_state(recovery_config(session_id))
    durability = await _durability_log().state(session_id)
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    outputs = list(values.get("outputs") or [])
    next_steps = list(snapshot.next)
    worker_active = _task_active(session_id)
    owner_active = durability.status == "running"
    error = _task_errors.get(session_id)
    if error:
        state = "failed"
    elif len(outputs) == len(_STEP_TOOLS):
        state = "completed"
    elif worker_active or owner_active:
        state = "running"
    elif next_steps:
        state = "stopped" if durability.status == "stopped" else "waiting_for_start"
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
        "owner_active": owner_active,
        "needs_resume": bool(next_steps and not owner_active and not error),
        "error": error,
        "instance_id": process_id(),
        "step_seconds": step_seconds(),
        "execution_id": durability.execution_id,
        "attempt": durability.attempt,
        "owner_id": durability.owner_id,
        "heartbeat_at": durability.heartbeat_at,
        "heartbeat_age_seconds": durability.heartbeat_age_seconds,
        "heartbeat_fresh": durability.heartbeat_fresh,
        "heartbeat_interval_seconds": heartbeat_seconds(),
        "stale_after_seconds": stale_seconds(),
        "durability_event_count": durability.event_count,
        "recent_durability_events": durability.recent_events,
        "claim_mode": "session_store_last_writer_wins",
        "atomic_claim": False,
    }


async def start(session_id: str) -> dict[str, Any]:
    current = await status(session_id)
    if current["status"] in {"running", "completed"}:
        return current
    graph_input: RecoveryState | None = (
        {"session_id": session_id, "outputs": []} if current["status"] == "not_started" else None
    )
    await _launch(session_id, graph_input)
    await asyncio.sleep(0)
    return await status(session_id)


async def resume(session_id: str) -> dict[str, Any]:
    current = await status(session_id)
    if current["worker_active"] or current["status"] == "completed":
        return current
    if not current["needs_resume"]:
        raise ValueError(f"No incomplete recovery sequence exists for session {session_id!r}.")
    await _launch(session_id, None)
    await asyncio.sleep(0)
    return await status(session_id)
