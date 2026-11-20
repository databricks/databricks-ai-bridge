"""OpenAI Agents template adapter with SQLite history and incremental text."""

import asyncio
import importlib
import time
from pathlib import Path

from databricks_agentkit.sandbox.adapters.common import (
    configure_project,
    measure,
    prepare,
    prepare_source,
    shutdown_session,
    trace_result,
    workspace_client_for,
)
from databricks_agentkit.sandbox.adapters.imports import warm_imports


async def invoke(request: dict, state_dir: Path, *, emit=None) -> dict:
    started = time.perf_counter()
    timings = {}
    with warm_imports(state_dir, timings):
        with measure(timings, "framework_import"):
            from agents import Runner, SQLiteSession, set_tracing_disabled
            from agents.mcp import MCPServerManager
            from agents.stream_events import RawResponsesStreamEvent
            from openai.types.responses import ResponseTextDeltaEvent

            from databricks_agentkit.openai import mcp_servers, session_store
            from databricks_agentkit.runtime import start_trace
            from databricks_agentkit.runtime.tool_manifest import resolve_session_store

        with measure(timings, "project_import"):
            project = importlib.import_module("agent.agent")
            mcps = importlib.import_module("agent.mcps")
    with measure(timings, "bindings"):
        message, model = prepare(request, state_dir, project.MODEL)
    with measure(timings, "configuration"):
        configure_project(project)
    set_tracing_disabled(True)
    store = resolve_session_store()
    with measure(timings, "session_open"):
        session = (
            session_store(request["session_id"], store=store)
            if store
            else SQLiteSession(request["session_id"], db_path=str(state_dir / "chat.sqlite"))
        )

    discovery_started = time.perf_counter()
    servers = await mcp_servers(mcps.build_mcp_servers(), workspace_client_for=workspace_client_for)
    discovered = {}
    async with MCPServerManager(servers) as manager:
        if manager.errors:
            names = ", ".join(server.name for server in manager.errors)
            raise RuntimeError(f"Could not connect to configured MCP servers: {names}")
        for server in manager.active_servers:
            if hasattr(server, "cache_tools_list"):
                # The SDK's base MCPServer type omits this optional writable attribute.
                setattr(server, "cache_tools_list", True)  # noqa: B010
            tools = await asyncio.wait_for(server.list_tools(), manager.connect_timeout_seconds)
            discovered[server.name] = [tool.name for tool in tools]
        timings["mcp_discovery"] = round((time.perf_counter() - discovery_started) * 1000, 2)
        with measure(timings, "agent_build"):
            agent = project.create_agent(
                request["session_id"],
                manager.active_servers,
                model=model,
                workspace_client_for=workspace_client_for,
            )
        prepared = time.perf_counter()
        with start_trace(
            name="sandbox.invoke", inputs={"message": message}, session_id=request["session_id"]
        ) as span:
            result = Runner.run_streamed(agent, message, session=session, max_turns=10)
            async for event in result.stream_events():
                if (
                    isinstance(event, RawResponsesStreamEvent)
                    and isinstance(event.data, ResponseTextDeltaEvent)
                    and emit is not None
                ):
                    emit(event.data.delta)
            if span is not None:
                span.set_outputs({"reply": str(result.final_output)})
    trace = trace_result(span)
    if result.interruptions:
        reply = (
            "This action requires approval. The sandbox launcher demo does not yet "
            "support approval and resume; the action has not been executed."
        )
    else:
        reply = str(result.final_output)
    return {
        "reply": reply,
        "model": model,
        "framework": "openai-agents",
        "mcp_tools": discovered,
        "tool_auth": "sandbox-workspace-identity",
        "approval_required": bool(result.interruptions),
        "trace": trace,
        "session_backend": {"mode": "managed", "store": store}
        if store
        else {"mode": "sandbox-sqlite"},
        "tool_calls": [
            item.raw_item.name
            for item in result.new_items
            if item.type == "tool_call_item" and hasattr(item.raw_item, "name")
        ],
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
            importlib.import_module("agents")
            importlib.import_module("agents.mcp")
            importlib.import_module("agents.stream_events")
            importlib.import_module("openai.types.responses")
            importlib.import_module("databricks_agentkit.openai")
            importlib.import_module("databricks_agentkit.runtime")
            importlib.import_module("databricks_agentkit.runtime.tool_manifest")
            importlib.import_module("databricks_agentkit.openai.mcp")
            importlib.import_module("databricks_agentkit.openai.sessions")
            importlib.import_module("databricks_agentkit.runtime.tracing")
            importlib.import_module("agent.agent")
            importlib.import_module("agent.mcps")
    with measure(timings, "bindings"):
        prepare_source(state_dir)
    with measure(timings, "configuration"):
        configure_project(importlib.import_module("agent.agent"))
    return {"timings_ms": timings}


# Function attributes survive re-export through runtime.sandbox_entry.
setattr(invoke, "prepare_session", prepare_session)  # noqa: B010
setattr(invoke, "shutdown_session", shutdown_session)  # noqa: B010
