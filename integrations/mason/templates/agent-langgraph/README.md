# Agent — LangGraph (FastAPI)

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent **backend** for Databricks Apps,
served from a **FastAPI app** — no serving framework. It runs locally with **no
database and no setup** — just an auth profile. It speaks LangGraph's **native** shape on both ends:
`POST /invocations` takes an `input` list of LangChain message dicts (streaming via SSE, plus an
in-memory `background` mode with `GET /invocations/{id}`) and returns LangChain messages — nothing is
reshaped into another contract.

The HTTP surface is hand-written in `runtime/runtime.py` (routes, SSE framing, tracing spans,
background wiring), so the template shows exactly how the agent is served — request and response
bodies are plain dicts, no wrapper types.

This template is API-first. Call it with `curl` or use it from your own client.

Local clients can use `/invocations`. For a deployed Databricks App, use the equivalent
`/api/invocations` route with an OAuth Bearer token; Databricks Apps reserves `/api/*` for
programmatic token authentication. Polling and health checks likewise have `/api` aliases.

## Project layout

```
agent/                 # the agent (reasoning plane) — this is what you edit
  agent.py             #   invoke / stream handlers + create_agent_graph() + event serialization
  tools/               #   function tools — drop a *.py file here to add one (auto-collected)
    sample_tool.py     #     get_current_time — a working example (@tool)
    send_message.py    #     a side-effecting tool gated by human approval (see REQUIRE_APPROVAL)
  mcps.py              #   MCP servers: none by default; add to build_mcp_servers() to offer some
runtime/               # the HTTP surface — SDK-agnostic; rarely edited
  runtime.py           #   build_app(): FastAPI routes, SSE framing, tracing spans, background wiring
  main.py              #   entry point: loads config, builds the app, runs uvicorn
tests/
  test_agent.py        #   hermetic smoke tests + one gated live model call
```

You edit `agent/agent.py`, `agent/tools/`, and `agent/mcps.py`; the plumbing (session checkpointer,
tracing, MCP tool loading, background store) lives in the `databricks-mason` package —
framework-neutral pieces under `databricks_mason.runtime`, LangGraph-specific ones under
`databricks_mason.langgraph` — so the template ships only your agent code. `runtime/runtime.py` is the
SDK-agnostic HTTP surface — it wires two generic handlers (`invoke_handler`/`stream_handler`) to the
endpoints, so the agent SDK lives entirely behind them in `agent/agent.py`. `tools/` is a drop-in
package: add a `*.py` with a `@tool` function and it's auto-collected (no edits to existing code).
`mcps.py` exposes `build_mcp_servers()` (empty by default — add servers to offer them).

## Run locally

No database required. Conversation state is kept in an in-process LangGraph checkpointer.

```bash
# 1. Configure a Databricks auth profile (used only to call the model)
cp .env.example .env
# edit .env: set DATABRICKS_CONFIG_PROFILE=<your-profile>

# 2. Start the server (installs deps via uv on first run)
uv run start-server        # serves at http://localhost:8000

# 3. Send a request
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "What time is it? Use your tool."}]}'
```

The model call goes to your Databricks workspace (via the profile). Everything else — session
storage, tracing — is off by default and requires no setup.

## Client contract

The Databricks Apps `__Host-databricks-app-router` cookie is the single session identifier. It both
keeps requests on the same App replica and keys LangGraph checkpoints, HITL resumes, Session Store
records, and the durability demo. Do **not** send `session_id` in request bodies. The runtime ignores
an old body value and injects the cookie value before calling the agent. Browsers resend the Apps
cookie automatically; API clients must preserve it in a cookie jar. Localhost has no Apps router, so
the server sets an HTTP-only `mason-local-session` fallback cookie instead.

TODO: switch the client contract to `X-Routing-Key` when Databricks Apps supports it. Until then use
the [documented Apps routing cookie](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/horizontal-scaling#api-clients).

```bash
curl -X POST "https://<app>.databricksapps.com/invocations" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -b "__Host-databricks-app-router=<routing-key>" \
  -d '{"input":[{"role":"user","content":"hi"}]}'
```

The examples below use a localhost cookie jar so every request addresses the same session:

```bash
BASE=http://localhost:8000
COOKIE_JAR=/tmp/mason-agent.cookies
curl -s -c "$COOKIE_JAR" "$BASE/health"
```

When the chat app is enabled, `GET /api/demo/config` returns the resolved `session_id`, process
`instance_id`, execution identity, and the enabled state for streaming, background, Session Store,
Memory Store, durability, heartbeat, and app control. The UI uses this response to color capability
indicators automatically. Only the sync/streaming/background selector is a manual client choice.

**Non-streaming:**

```bash
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"hi"}]}'
```

The response is `{ "output": [...], "session_id": "...", "status": "completed" }`. `output`
contains native LangChain message dictionaries.

**Streaming** adds `"stream": true` and returns SSE. Completed messages use
`data: {"type":"message","message":{...}}`; token chunks use
`data: {"type":"delta","content":"...","id":"..."}`; interruptions use
`data: {"type":"interrupt",...}`; the final frame is `data: [DONE]`.

```bash
curl -sN -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"Count to three"}],"stream":true}'
```

**Background** (add `"background": true`) returns an `inv_...` id immediately; poll it:

```bash
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"do something"}],"background":true}'
# -> {"id":"inv_...","status":"in_progress"}

curl -s -b "$COOKIE_JAR" "$BASE/invocations/inv_..."
# -> in_progress, completed + output, or failed + error
```

Background runs and polling are in-memory and single-process. The routing cookie is required so the
poll reaches the same replica, but the run itself does not survive a restart.

### Chat app state APIs

When initialized with the chat app (the default for `mason init --framework langgraph`, unless
`--disable-chat-app` is passed), the browser also calls:

- `POST /api/session/new` to generate a fresh session id and replace the routing cookie. The request
  has no body-level `session_id`; the response includes the new and previous ids.
- `POST /api/demo/sessions` to create or resolve the current cookie-backed managed session.
- `GET /api/demo/sessions` to list recent sessions for the configured actor. In local in-memory mode
  it returns only the current browser session.
- `POST /api/demo/sessions/{session_id}/open` to verify an actor-scoped managed session, replace the
  routing cookie, and load that session's transcript and pending state.
- `GET /api/demo/session/items` to load the current transcript. Without a managed Session Store it
  reconstructs messages and pending interrupts from the in-process LangGraph checkpoint. Managed
  responses filter out checkpoint fragments and durability events before returning items to the UI.
- `POST /api/demo/session/items` to mirror user, assistant, tool, and human-decision items into the
  managed Session Store.
- `GET /api/demo/memory/entries`, `POST /api/demo/memory/entries`, and
  `POST /api/demo/memory/search` for managed long-term memory. Created entries are tagged with the
  current cookie-backed session id; actor partitioning comes from `AGENT_MEMORY_ACTOR_ID`.

### Human-in-the-loop (tool approval)

Tools named in `REQUIRE_APPROVAL` (in `agent/agent.py`) pause for human approval before they run. The
template ships with one gated demo tool, `send_message` — ask the agent to send a message and instead
of running the tool, the run **pauses** and emits an `interrupt` event describing the pending call:

```json
{ "type": "interrupt", "id": "int_...",
  "value": { "action_requests": [{ "name": "send_message",
             "args": { "recipient": "alice@x.com", "body": "hi" } }], "review_configs": [...] } }
```

The paused run is checkpointed on the cookie-backed session thread. **Resume** with the same cookie
and a native LangGraph `resume` payload — one decision per pending call:

```bash
# Approve — the tool runs
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"resume":{"decisions":[{"type":"approve"}]}}'

# Reject — the tool is skipped; the message is fed back to the model
#   { "type": "reject", "message": "Not allowed." }
# Edit — run the tool with changed args
#   { "type": "edit", "edited_action": { "name": "send_message", "args": { "recipient": "...", "body": "..." } } }
# Respond — answer on the tool's behalf without running it
#   { "type": "respond", "message": "..." }
```

Non-streaming replies to a gated turn come back with `"status": "interrupted"` and the `interrupt`
event as the last `output` item; approved/rejected resumes return `"status": "completed"`.

To gate more tools, add their names to `REQUIRE_APPROVAL`; empty the dict to disable approval
entirely. Which decisions are allowed per tool is configurable — see LangChain's
`HumanInTheLoopMiddleware`.

> Like sessions, a paused run lives in the checkpointer — **in-memory and single-process** by default,
> so it survives only within the running process. Back the checkpointer with a durable store (below)
> for pauses that survive restarts / span replicas.

**Multi-turn** needs no body bookkeeping: reuse the same cookie jar for each turn.

```bash
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"My name is Alice"}]}'
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"What is my name?"}]}'
```

### Stop/start durability demo

The chat app always includes Start App and Stop App; there are no enable-stop or enable-crash flags.
With `AGENT_SESSION_STORE` configured:

1. `POST /api/demo/app/start` starts `tool_step_1` through `tool_step_4` in a sequential LangGraph.
2. Each completed tool output is checkpointed before the next node begins.
3. `databricks_mason/runtime/durability.py` appends attempt, owner, heartbeat, completion, and failure events to
   Session Store. Heartbeats default to every 3 seconds and become stale after 10 seconds.
4. `POST /api/demo/app/stop` schedules `os._exit(86)`. Databricks Apps restarts the process; locally
   restart `uv run start-server` with a supervisor or manually.
5. The browser keeps the same routing cookie, waits for a new `instance_id`, polls
   `GET /api/demo/recovery`, and calls `POST /api/demo/app/start` after the old heartbeat is stale.
6. LangGraph restores the checkpoint, skips completed tools, and continues at the first incomplete
   node. A tool interrupted before its output checkpoint commits can run again (at-least-once).

This is intentionally demo-grade durability. Session Store does not provide an atomic
compare-and-swap lease here, so ownership is last-writer-wins (`atomic_claim: false`). A production
runtime also needs transactional ownership, server-side stale-run scanning, idempotent side effects,
and durable event replay.

## Customize the agent

- **Model / instructions:** `create_agent_graph()` in `agent/agent.py`.
- **Add a tool:** drop a new file in `agent/tools/` with a `@tool`-decorated function; it's
  collected automatically (see `agent/tools/sample_tool.py`). No wiring to edit.
- **Require approval for a tool:** add its name to `REQUIRE_APPROVAL` in `agent/agent.py` (see the
  human-in-the-loop section above); empty the dict to disable gating.
- **Add an MCP server:** append a `DatabricksMCPServer` to `build_mcp_servers()` in `agent/mcps.py`.
- **Make state durable:** set `AGENT_SESSION_STORE` (see "Enable durable state" below); the
  checkpointer lives in `databricks_mason/langgraph/session_store.py`.
- **Add long-term memory:** set `AGENT_MEMORY_STORE` to a managed memory store ID; `create_agent_graph()`
  then includes the `remember`/`recall` tools from `databricks_mason/langgraph/memory.py` (persist/search facts across
  conversations). Unset → the model isn't offered them.
- **Change the HTTP surface:** `runtime/runtime.py` — routes, SSE framing, background wiring (the run
  store itself is `databricks_mason/runtime/background.py`).

### Use Unity Catalog skills (LangGraph only)

Discover the skills in a Unity Catalog schema, then attach an exact skill binding to this project:

```bash
mason skills list --schema catalog.schema
mason skills add uc catalog.schema.skill --source .
```

Bindings live in `agent.toml`:

```toml
[[skills]]
id = "review"
source = { kind = "uc", name = "catalog.schema.skill" }
```

The agent receives each declared skill's name and description as metadata. Instructions are fetched
only when it calls `load_skill` with the binding ID; referenced files are fetched remotely from UC
through `read_skill_file` and the `get_skill_files` endpoint. Skill instructions and reference files
are never copied into the project or deployment.

Only exact UC bindings are supported. There are no local, custom, or path-based sources, automatic
discovery, or versions. The deployed Databricks App's service principal needs `USE_CATALOG` on the
catalog, `USE_SCHEMA` on the schema, and `READ_SKILL` on the bound skill.

## Test

```bash
uv run pytest                 # hermetic smoke tests (tools, session, event serialization)
```

The smoke tests need no auth. `tests/test_agent.py` also has one end-to-end test that calls the
model; it runs only when a workspace profile is configured (`DATABRICKS_CONFIG_PROFILE` or
`DATABRICKS_HOST`+`DATABRICKS_TOKEN`) and skips otherwise.

## Deploy

Deploy with the [Mason](../../README.md) CLI:

```bash
mason deploy agent-langgraph --source .
```

Add `--with-memory-store <name> --with-session-store <name> --actor-id <actor>` to wire managed
state. Mason provisions or resolves the stores, injects the store/actor env vars, and deploys the
App.

`app.yaml` carries the app's start command and env. By default the deployed app is the same lean
backend: in-process session state, tracing off.

### Enable MLflow tracing (optional)

Tracing turns on when MLflow has **both a destination and an experiment** — set one of each, in
whichever form you have. The app code needs no change; MLflow resolves the specific value.

- **Destination:** `MLFLOW_TRACKING_URI` (e.g. `"databricks"`) or `MLFLOW_TRACING_DESTINATION`
  (an experiment id or a `catalog.schema`).
- **Experiment:** `MLFLOW_EXPERIMENT_ID` or `MLFLOW_EXPERIMENT_NAME`.

Set neither half → tracing stays off. Examples:

- **Local:** `MLFLOW_TRACKING_URI="databricks"` + `MLFLOW_EXPERIMENT_ID=<id>` (or `..._NAME=<name>`)
  in `.env`, pointing at an experiment in the workspace your profile targets.
- **Deployed:** set the same env in `app.yaml` and attach an `experiment` resource (its `valueFrom`
  binding injects `MLFLOW_EXPERIMENT_ID`).

When both halves are present the agent enables MLflow autolog (`mlflow.langchain.autolog()`) and tags
each trace with the session id. Otherwise it disables tracing outright, so the per-request span
`runtime/runtime.py` opens has nothing to export and no traces are created.

### Enable durable state (optional)

By default the agent uses an in-process LangGraph checkpointer (`InMemorySaver`) — multi-turn and
human-in-the-loop pauses work within a running process but do not survive restarts or span replicas.

Set **`AGENT_SESSION_STORE`** to a managed [Session Store](../../README.md) name and
`databricks_mason/langgraph/session_store.py` returns a `DatabricksSessionStoreSaver` instead. It's a LangGraph
`BaseCheckpointSaver` that serializes each checkpoint into ordered **session items** and stores them
through the managed Session Store **REST API** — no database the app connects to directly. Full graph
state — including paused HITL runs (pending writes + interrupts) — is durable across restarts and
replicas, over RPCs only. No agent code changes; the checkpointer swap is the only difference.

> The saver is adapted from the first-party `databricks_agent_client.langgraph` prototype, over a
> small vendored REST client (`databricks_mason/runtime/session_store_client.py`) so the template needs no
> unpublished dependency. Swap both for the published package when it lands. The store must already
> exist; access uses the caller's normal Databricks auth (the deployed app's service principal, or
> your profile locally — whichever the Session Store grants).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABRICKS_CONFIG_PROFILE` | `DEFAULT` | Auth profile used to call the model (local dev) |
| `PORT` | `8000` | Port the server listens on |
| `AGENT_MEMORY_STORE` | _unset_ | Managed memory store ID → registers `remember`/`recall` long-term-memory tools |
| `AGENT_MEMORY_ACTOR_ID` | `agent` | Actor partition used by memory tools |
| `AGENT_SESSION_STORE` | _unset_ | Managed Session Store name → durable checkpointer (REST-backed); unset = in-process `InMemorySaver` |
| `AGENT_SESSION_ACTOR_ID` | session id | Actor used by the durable saver |
| `MLFLOW_TRACKING_URI` | _unset_ | Trace destination (e.g. `databricks`). A destination + an experiment enables tracing |
| `MLFLOW_TRACING_DESTINATION` | _unset_ | Alt destination — experiment id or `catalog.schema` (either destination var works) |
| `MLFLOW_EXPERIMENT_ID` | _unset_ | Experiment to trace to (by id) |
| `MLFLOW_EXPERIMENT_NAME` | _unset_ | Experiment to trace to (by name; alternative to the id) |

## Notes

- **The event serialization in `agent/agent.py` (`_serialize_events`) is LangGraph-specific** — it
  turns LangGraph's native `astream` events into JSON dicts without reshaping them into another
  contract. **`runtime/runtime.py` is SDK-agnostic** — it hosts any agent exposing the
  `invoke_handler`/`stream_handler` dict contract.
- **Background mode is in-memory** (`databricks_mason/runtime/background.py`, wired in `runtime/runtime.py`) —
  non-durable, single-process; see the note under the client contract.
