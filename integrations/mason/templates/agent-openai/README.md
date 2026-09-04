# Agent — OpenAI Agents SDK (FastAPI)

An [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) agent **backend** for
Databricks Apps, served by `databricks_mason.DurableAgentApp`. It runs locally with **no database
and no setup** beyond an auth profile. The model runs on Databricks (via `databricks-openai`), and
`POST /invocations` takes an `input` list of message dicts (streaming via SSE, plus a `background`
mode with `GET /invocations/{id}`).

The public application is a thin HTTP layer over Mason's durable runtime engine. The application
owns routes, SSE framing, and browser sessions; the engine owns persisted execution state,
heartbeats, claims, and event replay. The agent translates the Agents SDK's native run events into
that generic contract in `agent/agent.py`.

This template is API-first. Call it with `curl` or use it from your own client.

Local clients can use `/invocations`. For a deployed Databricks App, use the equivalent
`/api/invocations` route with an OAuth Bearer token; Databricks Apps reserves `/api/*` for
programmatic token authentication. Polling and health checks likewise have `/api` aliases.

## Project layout

```
agent/                 # the agent (reasoning plane) — this is what you edit
  agent.py             #   invoke / recover callbacks + create_agent() + event serialization
  tools/               #   function tools — drop a *.py file here to add one (auto-collected)
    sample_tool.py     #     get_current_time — a working example (@function_tool)
    send_message.py    #     a side-effecting tool gated by human approval (needs_approval=True)
  mcps.py              #   MCP servers: none by default; add to build_mcp_servers() to offer some
runtime/               # thin generated entry point; rarely edited
  main.py              #   loads config and runs the SDK-provided application
tests/
  test_agent.py        #   hermetic smoke tests + one gated live model call
```

You edit `agent/agent.py`, `agent/tools/`, and `agent/mcps.py`; the plumbing (durable application,
execution runtime, session store, tracing, and MCP server construction) lives in the
`databricks-mason` package. Framework-neutral pieces are under `databricks_mason.runtime`, while
OpenAI-specific pieces are under `databricks_mason.openai`. `runtime/main.py` passes the plain
`invoke` and `recover` callbacks from `agent/agent.py` to `DurableAgentApp`. `tools/` is a drop-in
package: add a `*.py` with a `@function_tool` function and it's auto-collected (no edits to existing
code). `mcps.py` exposes `build_mcp_servers()` (empty by default — add servers to offer them).

## Run locally

No database required. Conversation state is kept in an in-process session (`SQLiteSession` backed by
`:memory:`).

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
keeps requests on the same App replica and keys the conversation session, resumes, and Session Store
records. Do **not** send `session_id` in request bodies. The runtime ignores an old body value and
injects the cookie value before calling the agent. Browsers resend the Apps cookie automatically; API
clients must preserve it in a cookie jar. Localhost has no Apps router, so the server sets an
HTTP-only `mason-local-session` fallback cookie instead.

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
`instance_id`, the signed-in viewer, and the enabled state for streaming, background, Session Store,
and Memory Store. The UI uses this response to color capability indicators automatically. Only the
sync/streaming/background selector is a manual client choice.

**Non-streaming:**

```bash
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"hi"}]}'
```

The response is `{ "output": [...], "session_id": "...", "status": "completed" }`. `output` contains
normalized message dictionaries (`{role, content, tool_calls?}`).

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

Locally, background runs and polling use the in-memory durability store. After `mason deploy`, they
use the selected Lakebase database, so polling, event replay, and stale-heartbeat recovery work
across restarts and replicas.

### Chat app state APIs

When initialized with the chat app (the default for `mason init --framework openai`, unless
`--disable-chat-app` is passed), the browser also calls:

- `POST /api/session/new` to generate a fresh session id and replace the routing cookie. The request
  has no body-level `session_id`; the response includes the new and previous ids.
- `POST /api/demo/sessions` to create or resolve the current cookie-backed managed session.
- `GET /api/demo/sessions` to list recent sessions for the configured actor. In local in-memory mode
  it returns only the current browser session.
- `POST /api/demo/sessions/{session_id}/open` to verify an actor-scoped managed session, replace the
  routing cookie, and load that session's transcript.
- `GET /api/demo/session/items` to load the current transcript. Without a managed Session Store it
  reconstructs messages from the in-process session. Managed responses filter out non-message items
  before returning items to the UI.
- `POST /api/demo/session/items` to mirror user, assistant, tool, and human-decision items into the
  managed Session Store.
- `GET /api/demo/memory/entries`, `POST /api/demo/memory/entries`, and
  `POST /api/demo/memory/search` for managed long-term memory. Created entries are tagged with the
  current cookie-backed session id; entries are partitioned by the signed-in user (the request's
  forwarded-identity header), so the panel shows that user's own memory.

### Human-in-the-loop (tool approval)

Tools declared with `needs_approval=True` (and named in `REQUIRE_APPROVAL` in `agent/agent.py`) pause
for human approval before they run. The template ships with one gated demo tool, `send_message` — ask
the agent to send a message and instead of running the tool, the run **pauses** and emits an
`interrupt` event describing the pending call:

```json
{ "type": "interrupt", "id": "call_...",
  "value": { "action_requests": [{ "name": "send_message",
             "args": { "recipient": "alice@x.com", "body": "hi" } }] } }
```

**Resume** with the same cookie and a decision payload — one decision per pending call:

```bash
# Approve — the tool runs
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" \
  -H "Content-Type: application/json" \
  -d '{"resume":{"decisions":[{"type":"approve"}]}}'

# Reject — the tool is skipped; an optional message is fed back to the model
#   { "type": "reject", "message": "Not allowed." }
```

Non-streaming replies to a gated turn come back with `"status": "interrupted"` and the `interrupt`
event as the last `output` item; approved/rejected resumes return `"status": "completed"`.

To gate more tools, add `needs_approval=True` to the tool and its name to `REQUIRE_APPROVAL`; empty
the set to disable approval entirely.

> **Paused runs are in-process only.** Unlike the conversation transcript, a paused run (the Agents
> SDK `RunState`) is held in memory keyed by session id — it does **not** survive a restart or reach
> another replica, even with `AGENT_SESSION_STORE` set. An Agents SDK `Session` persists the
> transcript, not paused run state; durable HITL would stash the `RunState` separately. Resume a
> pause on the same process that created it.

**Multi-turn** needs no body bookkeeping: reuse the same cookie jar for each turn.

```bash
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"My name is Alice"}]}'
curl -s -b "$COOKIE_JAR" -X POST "$BASE/invocations" -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"What is my name?"}]}'
```

## Customize the agent

- **Model / instructions:** `create_agent()` in `agent/agent.py`.
- **Add a tool:** drop a new file in `agent/tools/` with a `@function_tool`-decorated function; it's
  collected automatically (see `agent/tools/sample_tool.py`). No wiring to edit.
- **Require approval for a tool:** add `needs_approval=True` to the tool and its name to
  `REQUIRE_APPROVAL` in `agent/agent.py` (see the human-in-the-loop section above); empty the set to
  disable gating.
- **Add an MCP server:** append an `McpServer` to `build_mcp_servers()` in `agent/mcps.py`.
- **Make history durable:** set `AGENT_SESSION_STORE` (see "Enable durable state" below); the session
  store lives in `databricks_mason/openai/sessions.py`.
- **Add long-term memory:** set `AGENT_MEMORY_STORE` to a managed memory store ID; `create_agent()`
  then includes the `remember`/`recall` tools from `databricks_mason/openai/memory.py` (persist/search
  facts across conversations). Unset → the model isn't offered them.
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
mason deploy agent-openai --source .
```

Add `--memory <name> --session <name>` to wire managed state. Mason provisions or resolves the
stores (creating them if missing), injects the store env vars, and deploys the App. Memory and
session data are partitioned per signed-in user automatically (see `_actor` in `agent/agent.py`).

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

When both halves are present the agent enables MLflow autolog (`mlflow.openai.autolog()`) and tags
each trace with the session id. Otherwise tracing remains disabled.

### Runtime durability

Local development uses an in-memory durability store. `mason deploy` selects exactly one Lakebase
database for invocation state, background polling, emitted events, heartbeats, and recovery:

1. The configured Session Store's Lakebase database, if present.
2. Otherwise the configured Memory Store's Lakebase database, if present.
3. Otherwise a dedicated `<app>-durability` Lakebase project, reused or provisioned by Mason.

Mason creates only the `databricks_mason_runtime` schema and its tables in that database. Runtime
recovery replays the persisted input against the same OpenAI session. This gives at-least-once
execution, so tools with external side effects must still be idempotent. Paused Agents SDK
`RunState` remains process-local, as described below.

### Enable durable conversation history (optional)

By default the agent uses an in-process session (`SQLiteSession` backed by `:memory:`) — multi-turn
history works within a running process but does not survive restarts or span replicas.

Set **`AGENT_SESSION_STORE`** to a managed [Session Store](../../README.md) name and
`databricks_mason/openai/sessions.py` returns a `DatabricksSessionStore` instead. It's an Agents
SDK `Session` that stores each Responses item as an ordered **session item** through the managed
Session Store **REST API** — no database the app connects to directly. The conversation transcript is
durable across restarts and replicas, over RPCs only. No agent code changes; the session swap is the
only difference.

> The session store is over a small vendored REST client
> (`databricks_mason/runtime/session_store_client.py`) so the template needs no unpublished
> dependency. Swap it for the published package when it lands. The store must already exist; access
> uses the caller's normal Databricks auth (the deployed app's service principal, or your profile
> locally — whichever the Session Store grants).
>
> **It persists the transcript, not paused runs.** An Agents SDK `Session` stores conversation items
> only, so a paused human-in-the-loop run stays in-process even when this is set (see the HITL note
> above).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABRICKS_CONFIG_PROFILE` | `DEFAULT` | Auth profile used to call the model (local dev) |
| `PORT` | `8000` | Port the server listens on |
| `AGENT_MEMORY_STORE` | _unset_ | Managed memory store ID → registers `remember`/`recall` long-term-memory tools |
| `AGENT_SESSION_STORE` | _unset_ | Managed Session Store name → durable transcript (REST-backed); unset = in-process `SQLiteSession` |
| `DATABRICKS_MASON_RUNTIME_ENDPOINT` | _deployed by Mason_ | Internal Lakebase endpoint selected for runtime durability; do not set manually in generated apps |
| `MLFLOW_TRACKING_URI` | _unset_ | Trace destination (e.g. `databricks`). A destination + an experiment enables tracing |
| `MLFLOW_TRACING_DESTINATION` | _unset_ | Alt destination — experiment id or `catalog.schema` (either destination var works) |
| `MLFLOW_EXPERIMENT_ID` | _unset_ | Experiment to trace to (by id) |
| `MLFLOW_EXPERIMENT_NAME` | _unset_ | Experiment to trace to (by name; alternative to the id) |

## Notes

- **The event serialization in `agent/agent.py` (`_serialize_events`) is Agents-SDK-specific** — it
  turns the SDK's native run events into the runtime's generic JSON contract (normalizing message
  items to `{role, content, tool_calls?}`), which is why the browser UI is identical across
  frameworks. The package-owned runtime and server remain framework-neutral.
- **Background mode follows the runtime store** — in-memory for local development, Lakebase-backed
  after `mason deploy`.
