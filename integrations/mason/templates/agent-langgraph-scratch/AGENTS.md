# Agent Development Guide

A LangGraph agent backend for Databricks Apps, served from a from-scratch FastAPI app (no serving
framework). Local-first: runs with no database and no setup beyond a Databricks auth profile. MLflow
tracing is optional.

See `README.md` for the full run / deploy / client-contract docs. This file is the quick map for
making changes.

## Run it

```bash
cp .env.example .env          # set DATABRICKS_CONFIG_PROFILE=<your-profile>
uv run start-server           # http://localhost:8000
```

No database needed — conversation state uses an in-process LangGraph checkpointer by default.

## Sample requests

`input` is a list of LangChain message dicts; the reply is `{ "output": [...], "session_id": "..." }`
where `output` is LangChain messages (native shape). The session id travels in the `X-Routing-Key`
header — omit it for a new conversation, send it back to continue one.

```bash
# Sync — run a turn to completion
curl -sX POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "What time is it? Use your tool."}]}'

# Streaming — SSE frames ending with `data: [DONE]`
curl -sN -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Count to 3."}], "stream": true}'
# frames: {"type":"message","message":{...}} (completed) and {"type":"delta","content":"...","id":"..."}

# Background — returns an inv_ id immediately; poll it
curl -sX POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Do something slow."}], "background": true}'
# -> {"id": "inv_1a2b3c4d5e6f7g8h9i0j1k2l", "status": "in_progress"}
curl -s http://localhost:8000/invocations/inv_1a2b3c4d5e6f7g8h9i0j1k2l
# -> {"id": "inv_1a2b3c4d5e6f7g8h9i0j1k2l", "status": "completed", "output": [...], "session_id": "..."}

# Human-in-the-loop — the gated send_message tool pauses for approval
curl -sX POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" -H "X-Routing-Key: S" \
  -d '{"input": [{"role": "user", "content": "Send a message to alice@x.com saying hi. Use send_message."}]}'
# -> {"output": [..., {"type":"interrupt","id":"int_...","value":{"action_requests":[...]}}], "session_id":"S", "status":"interrupted"}
# Resume with the same routing key and a native decision (approve | edit | reject | respond):
curl -sX POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" -H "X-Routing-Key: S" \
  -d '{"resume": {"decisions": [{"type": "approve"}]}}'
# -> {"output": [...], "session_id": "S", "status": "completed"}

# Multi-turn — send back the returned session_id as X-Routing-Key (same process; in-memory checkpointer)
curl -sX POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "My name is Alice."}]}'
curl -sX POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" -H "X-Routing-Key: <session-id>" \
  -d '{"input": [{"role": "user", "content": "What is my name?"}]}'
```

## Where things live

| You want to… | Edit |
| --- | --- |
| Change model / instructions | `agent/agent.py` (`create_agent_graph`) |
| Add a function tool | new `*.py` in `agent/tools/` with a `@tool` function (auto-collected) |
| Require human approval for a tool | add its name to `REQUIRE_APPROVAL` in `agent/agent.py` |
| Add an MCP server | append a `DatabricksMCPServer` to `build_mcp_servers()` in `agent/mcps.py` |
| Change how a request maps to a run | `agent/agent.py` (`invoke_handler` / `stream_handler`) |
| Change the session checkpointer | `agent/mason/session_store.py` |
| Change the HTTP surface (routes, SSE, background wiring) | `runtime/runtime.py` |
| Change the background-run store (make it durable) | `agent/mason/background.py` |
| Add a test | `tests/` (hermetic; gate model calls on a workspace profile — see `test_agent.py`) |

`runtime/runtime.py` is **SDK-agnostic** — it wires two generic handlers (`invoke_handler`/`stream_handler`,
plain `dict -> dict` / `dict -> AsyncGenerator[dict]`) to the endpoints. The agent SDK lives entirely
behind those handlers in `agent/agent.py`, so the serving layer is the same regardless of SDK.

`agent/mason/` holds plumbing (session checkpointer, tracing, MCP tool loading, background store)
slated to move into Databricks SDKs — grouped so that migration is localized.

## How tools register

`agent/tools/all_tools()` auto-imports every module in the package and collects every
`@tool`-decorated `BaseTool` it finds. So a tool registers just by existing in a file there —
`create_agent_graph()` calls `all_tools()`. **Do not** edit `agent/agent.py` to add a tool — just add
a file to `agent/tools/`.

## Sessions & durability

- Default: `agent/mason/session_store.py`'s `checkpointer()` returns an in-process `InMemorySaver`,
  keyed per request by `thread_config(session_id)` — no database, multi-turn works in-process.
- The session id arrives in the `X-Routing-Key` header; `runtime/runtime.py` copies it into the
  request dict as `session_id` before calling the handler.
- For durable, shared history, swap the checkpointer for a `PostgresSaver` over Lakebase.
- Background mode is in-memory / single-process — non-durable. The store is `agent/mason/background.py`
  (wired in `runtime/runtime.py`); swap it for a durable backend for cross-restart/replica recovery.
- Human-in-the-loop: tools in `REQUIRE_APPROVAL` (`agent/agent.py`) pause via LangChain's
  `HumanInTheLoopMiddleware`. The pause is checkpointed on the session thread and resumed by sending
  `resume` with the same routing key — no runtime change; it rides `/invocations` through the handlers.
  Durability follows the checkpointer (in-memory by default).

## MLflow tracing

Optional. Set both a destination (`MLFLOW_TRACKING_URI` or `MLFLOW_TRACING_DESTINATION`) and an
experiment (`MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`) to enable (`mlflow.langchain.autolog()`);
leave either half unset to skip. `runtime/runtime.py` opens a per-request span regardless.

## Quick commands

| Task | Command |
| --- | --- |
| Run locally | `uv run start-server` |
| Test | `uv run pytest` (hermetic; live model test runs only with a profile) |
| Deploy | `mason deploy agent-langgraph-scratch --source .` |

## Notes for maintainers

- The event serialization in `agent/agent.py` (`_serialize_events`) is LangGraph-specific: it turns
  `astream` `updates`/`messages` events into native LangChain-message JSON dicts (and relays
  `__interrupt__` as an `interrupt` event), without reshaping to another contract. `runtime/runtime.py`
  is SDK-agnostic.
