# Agent Development Guide

An OpenAI Agents SDK agent backend for Databricks Apps, served from a FastAPI app (no serving
framework). Local-first: runs with no database and no setup beyond a Databricks auth profile. MLflow
tracing is optional.

See `README.md` for the full run / deploy / client-contract docs. This file is the quick map for
making changes.

## Run it

```bash
cp .env.example .env          # set DATABRICKS_CONFIG_PROFILE=<your-profile>
uv run start-server           # http://localhost:8000
```

No database needed — conversation state uses an in-process session (`SQLiteSession`) by default.

## Sample requests

`input` is a list of message dicts; the reply is `{ "output": [...], "session_id": "..." }` where
`output` is normalized message dicts (`{role, content, tool_calls?}`). The
`__Host-databricks-app-router` cookie is both the Apps routing key and application session id; never
send `session_id` in the JSON body. Use a cookie jar locally so the server's `mason-local-session`
fallback is reused.

The examples below use local `/invocations` routes. Deployed Databricks Apps also expose the same
handlers under `/api/invocations` so OAuth Bearer-token calls pass through the Apps API gateway.

```bash
COOKIE_JAR=/tmp/mason-agent.cookies
curl -s -c "$COOKIE_JAR" http://localhost:8000/health

# Sync — run a turn to completion
curl -sb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "What time is it? Use your tool."}]}'

# Streaming — SSE frames ending with `data: [DONE]`
curl -sNb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Count to 3."}], "stream": true}'
# frames: {"type":"message","message":{...}} (completed) and {"type":"delta","content":"...","id":"..."}

# Background — returns an inv_ id immediately; poll it
curl -sb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Do something slow."}], "background": true}'
# -> {"id": "inv_1a2b3c4d5e6f7g8h9i0j1k2l", "status": "in_progress"}
curl -sb "$COOKIE_JAR" http://localhost:8000/invocations/inv_1a2b3c4d5e6f7g8h9i0j1k2l
# -> {"id": "inv_1a2b3c4d5e6f7g8h9i0j1k2l", "status": "completed", "output": [...], "session_id": "..."}

# Human-in-the-loop — the gated send_message tool pauses for approval
curl -sb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "Send a message to alice@x.com saying hi. Use send_message."}]}'
# -> {"output": [..., {"type":"interrupt","id":"call_...","value":{"action_requests":[...]}}], "session_id":"S", "status":"interrupted"}
# Resume with the same cookie and a decision (approve | reject):
curl -sb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"resume": {"decisions": [{"type": "approve"}]}}'
# -> {"output": [...], "session_id": "S", "status": "completed"}
# NOTE: paused runs are in-process only — resume on the same process (see README HITL note).

# Multi-turn — reuse the same cookie jar (same process; in-memory session)
curl -sb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "My name is Alice."}]}'
curl -sb "$COOKIE_JAR" -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "What is my name?"}]}'
```

## Where things live

| You want to… | Edit |
| --- | --- |
| Change model / instructions | `agent/agent.py` (`create_agent`) |
| Add a function tool | new `*.py` in `agent/tools/` with a `@function_tool` function (auto-collected) |
| Add a Databricks-managed integration | `mason tools add sandbox`, `mcp`, or `uc-function` (updates `agent/databricks_tools.py`) |
| Require human approval for a tool | `needs_approval=True` on the tool + its name in `REQUIRE_APPROVAL` (`agent/agent.py`) |
| Add an MCP server | append an `McpServer` to `build_mcp_servers()` in `agent/mcps.py` |
| Change how a request maps to a run | `agent/agent.py` (`invoke_handler` / `stream_handler`) |
| Change the conversation session | `databricks_mason/openai/sessions.py` |
| Change the HTTP surface (routes, SSE, background wiring) | `runtime/runtime.py` |
| Change the background-run store (make it durable) | `databricks_mason/runtime/background.py` |
| Add a test | `tests/` (hermetic; gate model calls on a workspace profile — see `test_agent.py`) |

`runtime/runtime.py` is **SDK-agnostic** — it wires two generic handlers (`invoke_handler`/`stream_handler`,
plain `dict -> dict` / `dict -> AsyncGenerator[dict]`) to the endpoints. The agent SDK lives entirely
behind those handlers in `agent/agent.py`, so the serving layer is the same regardless of SDK.

`databricks_mason.runtime` (from the `databricks-mason` package) holds framework-neutral plumbing
(tracing, workspace client, background store), and `databricks_mason.openai` holds the OpenAI-specific
pieces (session store, integration binding, memory tools), so the template ships only your agent code.

## How tools register

`agent/tools/all_tools()` auto-imports every module in the package and collects every
`@function_tool`-decorated `FunctionTool` it finds. So a tool registers just by existing in a file
there — `create_agent()` calls `all_tools()`. **Do not** edit `agent/agent.py` to add a tool — just
add a file to `agent/tools/`.

## Sessions

- Default: `databricks_mason/openai/sessions.py`'s `session_store()` returns an in-process
  `SQLiteSession` (`:memory:`), cached per session id — no database, multi-turn works in-process.
- The `__Host-databricks-app-router` cookie is both the Apps routing key and the session id. The
  runtime injects it into the internal handler request; body `session_id` values are ignored.
- TODO: replace the cookie with `X-Routing-Key` when Databricks Apps supports it.
- For durable history, set `AGENT_SESSION_STORE` to a managed Session Store name: `session_store()`
  returns a `DatabricksSessionStore` — an Agents SDK `Session` that stores each Responses item via the
  Session Store REST API (no DB connection), so the transcript survives restarts/replicas. Over a
  vendored REST client (`session_store_client.py`); swap for the published package when it lands.
  `AGENT_SESSION_ACTOR_ID` selects the actor partition.
- Background mode is in-memory / single-process — non-durable. The store is
  `databricks_mason/runtime/background.py` (wired in `runtime/runtime.py`); swap it for a durable
  backend for cross-restart/replica recovery.
- Human-in-the-loop: tools with `needs_approval=True` (and listed in `REQUIRE_APPROVAL`,
  `agent/agent.py`) pause via the Agents SDK. The paused `RunState` is stashed **in-process** keyed by
  session id and resumed by sending `resume` with the same cookie. Unlike the transcript, a paused run
  is **not** durable — it does not survive a restart or reach another replica, even with
  `AGENT_SESSION_STORE` set, because an Agents SDK `Session` persists conversation items, not run
  state.

## MLflow tracing

Optional. Set both a destination (`MLFLOW_TRACKING_URI` or `MLFLOW_TRACING_DESTINATION`) and an
experiment (`MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`) to enable (`mlflow.openai.autolog()`);
leave either half unset to skip. `runtime/runtime.py` opens a per-request span regardless.

## Quick commands

| Task | Command |
| --- | --- |
| Run locally | `uv run start-server` |
| Test | `uv run pytest` (hermetic; live model test runs only with a profile) |
| Deploy | `mason deploy agent-openai --source .` |

## Notes for maintainers

- The event serialization in `agent/agent.py` (`_serialize_events`) is Agents-SDK-specific: it turns
  the SDK's `raw_response_event`/`run_item_stream_event` stream into the runtime's generic JSON
  contract (token `delta`s, normalized `message` dicts, and an `interrupt` for a pending approval),
  without leaking SDK item types to the client. `runtime/runtime.py` is SDK-agnostic.
- Message normalization (`_normalize_item`) maps Agents SDK run items to `{role, content, tool_calls?}`
  so the browser UI — shared in shape with the LangGraph template — renders them unchanged.
