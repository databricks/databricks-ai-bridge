"""LangGraph template adapter with SQLite checkpoints and incremental text."""

import importlib
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path

from databricks_agentkit.sandbox.adapters.common import (
    configure_project,
    measure,
    prepare,
    prepare_source,
    shutdown_session,
    text_content,
    trace_result,
    workspace_client_for,
)
from databricks_agentkit.sandbox.adapters.imports import warm_imports


async def invoke(request: dict, state_dir: Path, *, emit=None) -> dict:
    started = time.perf_counter()
    timings = {}
    with warm_imports(state_dir, timings):
        with measure(timings, "framework_import"):
            from langchain.messages import AIMessage, AIMessageChunk, ToolMessage
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            from databricks_agentkit.langgraph import checkpointer, start_trace, thread_config
            from databricks_agentkit.runtime.tool_manifest import resolve_session_store

        with measure(timings, "project_import"):
            project = importlib.import_module("agent.agent")
    with measure(timings, "bindings"):
        message, model = prepare(request, state_dir, project.MODEL)
    with measure(timings, "configuration"):
        configure_project(project)
    reply, interrupted, tool_calls, tool_events = "", False, [], []
    store = resolve_session_store()
    async with AsyncExitStack() as stack:
        with measure(timings, "session_open"):
            saver = (
                checkpointer(store)
                if store
                else await stack.enter_async_context(
                    AsyncSqliteSaver.from_conn_string(str(state_dir / "chat.sqlite"))
                )
            )
        with measure(timings, "agent_build"):
            graph = await project.create_agent_graph(
                request["session_id"],
                model=model,
                workspace_client_for=workspace_client_for,
                checkpointer_override=saver,
            )
        prepared = time.perf_counter()
        with start_trace(
            name="sandbox.invoke", inputs={"message": message}, session_id=request["session_id"]
        ) as span:
            async for mode, payload in graph.astream(
                {"messages": [{"role": "user", "content": message}]},
                config=thread_config(request["session_id"]),
                stream_mode=["updates", "messages"],
                durability="sync",
            ):
                if mode == "messages":
                    chunk = payload[0]
                    if isinstance(chunk, AIMessageChunk) and emit:
                        if text := text_content(chunk.content):
                            emit(text)
                elif mode == "updates":
                    if payload.get("__interrupt__"):
                        interrupted = True
                    for node in payload.values():
                        for item in node.get("messages", []) if isinstance(node, dict) else []:
                            if isinstance(item, AIMessage):
                                if text := text_content(item.content):
                                    reply = text
                                tool_calls.extend(call["name"] for call in item.tool_calls)
                            elif isinstance(item, ToolMessage):
                                tool_events.append(
                                    {
                                        "call_id": item.tool_call_id,
                                        "name": item.name or "tool",
                                        "status": "failed"
                                        if item.status == "error"
                                        else "completed",
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                    }
                                )
            if span is not None:
                span.set_outputs({"reply": reply, "tool_calls": tool_calls})
        trace = trace_result(span)
    if interrupted:
        reply = (
            "This action requires approval. The sandbox launcher demo does not yet "
            "support approval and resume; the action has not been executed."
        )
    return {
        "reply": reply,
        "model": model,
        "framework": "langgraph",
        "tool_auth": "sandbox-workspace-identity",
        "approval_required": interrupted,
        "tool_calls": tool_calls,
        "tool_events": tool_events,
        "trace": trace,
        "session_backend": {"mode": "managed", "store": store}
        if store
        else {"mode": "sandbox-sqlite"},
        "worker_timings_ms": {
            "prepare": round((prepared - started) * 1000, 2),
            "agent_execution": round((time.perf_counter() - prepared) * 1000, 2),
            **timings,
        },
    }


async def prepare_session(state_dir: Path) -> dict:
    """Import and configure the template without a model call or tool execution."""
    timings = {}
    with warm_imports(state_dir, timings):
        with measure(timings, "framework_import"):
            importlib.import_module("langchain.messages")
            importlib.import_module("langgraph.checkpoint.sqlite.aio")
            importlib.import_module("databricks_agentkit.langgraph")
            importlib.import_module("databricks_agentkit.runtime.tool_manifest")
            importlib.import_module("databricks_agentkit.langgraph.session_store")
            importlib.import_module("databricks_agentkit.runtime.tracing")
            importlib.import_module("agent.agent")
    with measure(timings, "bindings"):
        prepare_source(state_dir)
    with measure(timings, "configuration"):
        configure_project(importlib.import_module("agent.agent"))
    return {"timings_ms": timings}


# Function attributes survive re-export through runtime.sandbox_entry.
setattr(invoke, "prepare_session", prepare_session)  # noqa: B010
setattr(invoke, "shutdown_session", shutdown_session)  # noqa: B010
