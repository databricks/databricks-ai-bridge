# Mason LangGraph Agent

A LangGraph agent served by `databricks_mason.AgentApp`. Mason keeps invocation state and events in
memory during `mason dev`. The generated app enables the durable runtime by default, so deployment
stores them in an app-owned Lakebase schema and recovers interrupted work. Initialize with
`--no-durable-runtime` for process-local deployed state instead.

## Run locally

```bash
mason dev
```

The API is available at `http://localhost:8000/api/invocations`. Every request supplies a UUID `id`.
That ID is the invocation identifier and idempotency key. Agent-specific values live inside the
opaque `input` object:

```bash
SESSION_ID=$(uuidgen)
INVOCATION_ID=$(uuidgen)

curl -sS http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"session_id\":\"$SESSION_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it? Use your tool.\"}]}}"
```

Reuse `SESSION_ID` for multi-turn conversation state. Generate a new `INVOCATION_ID` for each turn.
Retrying the same request with the same invocation ID returns the persisted result; changing the
request while reusing the ID returns `409`.

## Invocation modes

- Foreground: omit `background` and `stream`; the response contains the agent result under `output`.
- Foreground streaming: set `stream: true`; the response is SSE backed by persisted events.
- Background: set `background: true`; poll the returned `status_url`.
- Background streaming: set both flags; the `202` response includes `status_url` and `events_url`.

```bash
INVOCATION_ID=$(uuidgen)
curl -sN http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"session_id\":\"$SESSION_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to three.\"}]},\"stream\":true}"

INVOCATION_ID=$(uuidgen)
curl -sS http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"session_id\":\"$SESSION_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Summarize durable agents.\"}]},\"background\":true}" | jq
curl -sS "http://localhost:8000/api/invocations/$INVOCATION_ID" | jq
```

SSE records contain the events emitted by `agent/agent.py`: token `delta`s, completed `message`s,
and HITL `interrupt`s. Replay from a cursor with
`GET /api/invocations/{id}/events?after={sequence}`.

## Human approval

`send_message` is gated by `HumanInTheLoopMiddleware`. Start a turn asking the agent to use that
tool. When the output or event stream contains an `interrupt`, submit a new invocation with the same
application session:

```json
{
  "id": "<new-uuid>",
  "input": {
    "session_id": "<same-session-id>",
    "resume": {"decisions": [{"type": "approve"}]}
  }
}
```

The default checkpointer is process-local. Bind a managed Session Store to preserve multi-turn and
paused LangGraph state across restarts:

```bash
mason sessions bind my-agent-sessions
```

## Crash recovery

When `DURABLE_RUNTIME` is enabled, `runtime/main.py` registers both `@app.invoke` and
`@app.on_recovery`. The initial attempt writes the invocation ID into LangGraph checkpoint metadata
with synchronous checkpoint durability. A recovery attempt continues from that checkpoint when it
exists; otherwise it safely replays the persisted application input. Invocation state and emitted
events survive process loss in deployed Lakebase. With `--no-durable-runtime`, only `@app.invoke`
is registered and invocation state and events remain process-local.

External side effects are still at-least-once. Make tools idempotent because work performed between
the last checkpoint and a crash can run again.

## Chat app

The browser UI is included by default. It generates a stable application session ID in local
storage, places it inside each invocation's `input`, and generates a fresh invocation UUID per turn.
Use `mason init --framework langgraph --disable-chat-app` for API-only output.

## Configure and deploy

- Change model/instructions in `agent/agent.py`.
- Add local tools under `agent/tools/`; modules are auto-discovered.
- Add MCP servers in `agent/mcps.py` or with `mason tools add mcp`.
- Bind long-term memory with `mason memory bind <store>`.

```bash
mason --profile <profile> deploy agent-langgraph --source .
```

By default, `agent.toml` contains `[durability] enabled = true`. Deployment reuses a bound Session Store's
Lakebase database when available; otherwise Mason provisions or reuses the app's durability project.
Only the app-owned `databricks_mason_runtime_<hash>` schema and runtime tables are added. A project
initialized with `--no-durable-runtime` omits the durability binding and provisions no Lakebase.

The `__Host-databricks-app-router` cookie may be supplied independently for sticky replica routing.
It is not authentication and is not used as the template's application session ID.
