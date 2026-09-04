# Agent — LangGraph (FastAPI)

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent **backend** for Databricks Apps,
served by `databricks_mason.DurableAgentApp`. It runs locally with **no database and
no setup** beyond an auth profile. It speaks LangGraph's **native** shape on both ends:
`POST /invocations` takes an `input` list of LangChain message dicts (streaming via SSE, plus a
`background` mode with `GET /invocations/{id}`) and returns LangChain messages — nothing is reshaped
into another contract.

The public application is a thin HTTP layer over Mason's durable runtime engine. The application
owns routes, SSE framing, and browser sessions; the engine
owns persisted execution state, heartbeats, claims, and event replay. Only the application surface
is public for now, leaving the engine available to expose as a standalone runtime library later.

This template is API-first. Call it with `curl` or use it from your own client.

Local clients can use `/invocations`. For a deployed Databricks App, use the equivalent
`/api/invocations` route with an OAuth Bearer token; Databricks Apps reserves `/api/*` for
programmatic token authentication. Polling and health checks likewise have `/api` aliases.

## Project layout

```
agent/                 # the agent (reasoning plane) — this is what you edit
  agent.py             #   invoke / recover callbacks + create_agent_graph() + event serialization
  tools/               #   function tools — drop a *.py file here to add one (auto-collected)
    sample_tool.py     #     get_current_time — a working example (@tool)
    send_message.py    #     a side-effecting tool gated by human approval (see REQUIRE_APPROVAL)
  mcps.py              #   MCP servers: none by default; add to build_mcp_servers() to offer some
runtime/               # thin generated entry point; rarely edited
  main.py              #   loads config and runs the SDK-provided application
tests/
  test_agent.py        #   hermetic smoke tests + one gated live model call
```

You edit `agent/agent.py`, `agent/tools/`, and `agent/mcps.py`; the plumbing (durable application,
execution runtime, session checkpointer, tracing, and MCP tool loading) lives in the
`databricks-mason` package. Framework-neutral pieces are under `databricks_mason.runtime`, while
LangGraph-specific pieces are under `databricks_mason.langgraph`. `runtime/main.py` passes the plain
`invoke` and `recover` callbacks from `agent/agent.py` to `DurableAgentApp`.
`tools/` is a drop-in package: add a `*.py` with a `@tool` function and it is auto-collected. `mcps.py`
exposes `build_mcp_servers()` (empty by default — add servers to offer them).

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
  -d '{"id":"inv_quickstart_1","input": [{"role": "user", "content": "What time is it? Use your tool."}]}'
```

The model call goes to your Databricks workspace (via the profile). Everything else — session
storage, tracing — is off by default and requires no setup.

## Client contract

The Databricks Apps `__Host-databricks-app-router` cookie is the single session identifier. It both
keeps requests on the same App replica and keys LangGraph checkpoints, HITL resumes, and Session
Store records. Do **not** send `session_id` in request bodies; the fixed request model rejects it.
The runtime injects the cookie value before calling the agent. Browsers resend the Apps
cookie automatically; API clients must preserve it in a cookie jar. Localhost uses the same cookie;
the runtime creates it when the first request arrives.

Every invocation also requires a client-generated `id`. Reuse an `id` only to retry the exact same
request; reusing it with a different payload returns `409 Conflict`.

```bash
curl -X POST "https://<app>.databricksapps.com/invocations" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -b "__Host-databricks-app-router=<routing-key>" \
  -d '{"id":"inv_api_1","input":[{"role":"user","content":"hi"}]}'
```

The examples below use a localhost cookie jar so every request addresses the same session:

```bash
BASE=http://localhost:8000
COOKIE_JAR=/tmp/mason-agent.cookies
```

When the chat app is enabled, `GET /api/demo/config` returns the resolved `session_id`, process
`instance_id`, and the enabled state for streaming, background, Session Store,
and Memory Store. The UI uses this response to color capability indicators automatically. Only the
sync/streaming/background selector is a manual client choice.

**Non-streaming:**

```bash
curl -s -c "$COOKIE_JAR" -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"id":"inv_sync_1","input":[{"role":"user","content":"hi"}]}'
```

The response is `{ "output": [...] }`; `output` contains native LangChain message dictionaries.

**Streaming** adds `"stream": true` and returns SSE. Completed messages use
`data: {"type":"message","message":{...}}`; token chunks use
`data: {"type":"delta","content":"...","id":"..."}`; interruptions use
`data: {"type":"interrupt",...}`. The stream ends when the execution completes or fails.

```bash
curl -sN -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"id":"inv_stream_1","input":[{"role":"user","content":"Count to three"}],"stream":true}'
```

**Background** (add `"background": true`) accepts your id immediately; poll it:

```bash
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"id":"inv_background_1","input":[{"role":"user","content":"do something"}],"background":true}'
# -> {"id":"inv_background_1","status":"queued","attempt":0}

curl -s -b "$COOKIE_JAR" "$BASE/invocations/inv_background_1"
# -> queued, active, completed + output, or failed + error
```

Locally, background runs and polling use the in-memory durability store. After `mason deploy`, they
use the selected Lakebase database, so polling, event replay, and stale-heartbeat recovery work
across process replacement and replicas.

### Chat app state APIs

When initialized with the chat app (the default for `mason init --framework langgraph`, unless
`--disable-chat-app` is passed), the browser also calls:

- `POST /api/session/new` to generate a fresh session id and replace the routing cookie. The request
  has no body-level `session_id`; the response includes the new and previous ids.
- `POST /api/demo/sessions` to create or resolve the current cookie-backed managed session.
- `GET /api/demo/sessions` to list the current routing session. With no separate user identity, old
  sessions are intentionally outside the new routing session's partition.
- `POST /api/demo/sessions/{session_id}/open` to verify that a managed session belongs to the current
  routing-session partition before loading its transcript and pending state.
- `GET /api/demo/session/items` to load the current transcript. Without a managed Session Store it
  reconstructs messages and pending interrupts from the in-process LangGraph checkpoint. Managed
  responses filter out checkpoint fragments before returning items to the UI.
- `POST /api/demo/session/items` to mirror user, assistant, tool, and human-decision items into the
  managed Session Store.
- `GET /api/demo/memory/entries`, `POST /api/demo/memory/entries`, and
  `POST /api/demo/memory/search` for managed long-term memory. Created entries are tagged with the
  current cookie-backed session id; entries use that same routing session as their actor partition.

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
  -d '{"id":"inv_hitl_resume_1","resume":{"decisions":[{"type":"approve"}]}}'

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
  -d '{"id":"inv_turn_1","input":[{"role":"user","content":"My name is Alice"}]}'
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" -H "Content-Type: application/json" \
  -d '{"id":"inv_turn_2","input":[{"role":"user","content":"What is my name?"}]}'
```

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
- **Add application routes:** register them on the FastAPI `app` in `runtime/main.py`, following the
  chat overlay's `runtime/ui.py`. Mason's durable server and runtime stay package-owned.

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

Add `--memory <name> --session <name>` to wire managed state. Mason provisions or resolves the
stores (creating them if missing), injects the store env vars, and deploys the App. Memory and
session data use the routing session as their actor partition.

`app.yaml` carries the app's start command and env. LangGraph checkpoints remain in-process unless a
Session Store is configured, and tracing remains off unless configured. Runtime invocation state is
different: every deployed LangGraph app gets one Lakebase durability database automatically.

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
each trace with the session id. Otherwise it disables tracing outright and no traces are exported.

### Runtime durability

Local execution uses an in-memory durability store, so no database is required. `mason deploy`
selects exactly one Lakebase database for invocation state, background polling, emitted events,
heartbeats, and recovery:

1. The configured Session Store's Lakebase database, if present.
2. Otherwise a dedicated `<app>-durability` Lakebase project, reused or provisioned by Mason.

Mason creates only the `databricks_mason_runtime` schema and its execution/event tables in that
database. The managed Session and Memory Store services continue to use their own schemas and REST
APIs.

Pass `--no-create-stores` to require every selected managed store or dedicated durability project to
already exist. Mason attaches the durability database before application startup so the App service
principal creates and owns the Mason schema on first start.

### Durable LangGraph checkpoints (optional)

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
> unpublished dependency. Swap both for the published package when it lands. `mason deploy
> --session <name>` provisions or resolves the store unless `--no-create-stores` is passed. Access
> uses the caller's normal Databricks auth (the deployed app's service principal, or your profile
> locally — whichever the Session Store grants).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABRICKS_CONFIG_PROFILE` | `DEFAULT` | Auth profile used to call the model (local dev) |
| `PORT` | `8000` | Port the server listens on |
| `AGENT_MEMORY_STORE` | _unset_ | Managed memory store ID → registers `remember`/`recall` long-term-memory tools |
| `AGENT_SESSION_STORE` | _unset_ | Managed Session Store name → durable checkpointer (REST-backed); unset = in-process `InMemorySaver` |
| `DATABRICKS_MASON_RUNTIME_ENDPOINT` | _deployed by Mason_ | Internal Lakebase endpoint selected for runtime durability; do not set manually in generated apps |
| `MLFLOW_TRACKING_URI` | _unset_ | Trace destination (e.g. `databricks`). A destination + an experiment enables tracing |
| `MLFLOW_TRACING_DESTINATION` | _unset_ | Alt destination — experiment id or `catalog.schema` (either destination var works) |
| `MLFLOW_EXPERIMENT_ID` | _unset_ | Experiment to trace to (by id) |
| `MLFLOW_EXPERIMENT_NAME` | _unset_ | Experiment to trace to (by name; alternative to the id) |

## Notes

- **The event serialization in `agent/agent.py` (`_serialize_events`) is LangGraph-specific** — it
  turns LangGraph's native `astream` events into JSON dicts without reshaping them into another
  contract. The public `DurableAgentApp` delegates those events to an internal,
  framework-neutral runtime engine.
- **Background mode follows the runtime store** — in-memory for local development, Lakebase-backed
  after `mason deploy`.
