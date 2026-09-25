# OpenAI Agents template

An OpenAI Agents SDK agent served by `databricks_agentkit.DurableAgentServer`. The managed runtime keeps invocation state and
events in memory during `ab dev`. Deployment attaches a persistent Runtime Store, so invocation
state and events survive process loss and interrupted work can be recovered.

The generated project separates portable agent execution from the managed HTTP protocol:

```text
client -> runtime/main.py -> runtime/adapter.py -> agent/agent.py:run_agent
```

- `agent/agent.py` owns the framework-native agent, sessions, tools, MCP lifetime, HITL state, and
  `run_agent`.
- `runtime/adapter.py` owns the agent-author integration hooks: runtime input/output translation plus
  `invoke` and `recover`.
- `runtime/main.py` constructs the server and registers those hooks.

To bring an existing Agents SDK agent, keep its normal execution code in `agent/agent.py`, expose a
`run_agent` function that returns the native streaming result, and make only the small payload/event
mapping changes needed in `runtime/adapter.py`.

## Run locally

```bash
ab dev
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

Reuse `SESSION_ID` for multi-turn conversation history. Generate a new `INVOCATION_ID` for each
turn. Retrying the same request with the same invocation ID returns the persisted result; changing
the request while reusing the ID returns `409`.

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

SSE records contain events translated by `runtime/adapter.py`: token `delta`s, completed `message`s,
and HITL `interrupt`s. Replay from a cursor with
`GET /api/invocations/{id}/events?after={sequence}`.

## Human approval

`send_message` requires approval. When output or the event stream contains an `interrupt`, submit a
new invocation with the same application session:

```json
{
  "id": "<new-uuid>",
  "input": {
    "session_id": "<same-session-id>",
    "resume": {"decisions": [{"type": "approve"}]}
  }
}
```

The paused Agents SDK `RunState` is process-local. A managed Session Store preserves transcript
history, but not a pending approval across restarts or replicas.

## Crash recovery

`runtime/main.py` always registers the adapter's `invoke` and `recover` hooks. Both call the same
`agent.agent.run_agent` function. OpenAI Agents SDK does not currently expose LangGraph-style node
checkpoints, so `recover` replays the original application input against the same session. The
adapter prepends a developer instruction telling the agent that this is a recovery attempt and that
some tool calls or external side effects may already have completed or may still be in progress.
When deployment attaches a Runtime Store, invocation state and emitted events survive process loss
and the runtime can call `recover` on a replacement worker. Without a Runtime Store, invocation state
remains process-local and interrupted work is not automatically recovered. External side effects
remain at-least-once and must be idempotent.

## Chat app

The browser UI is included by default. It generates a stable application session ID in local
storage, places it inside each invocation's `input`, and generates a fresh invocation UUID per turn.
Use `ab init --framework openai --disable-chat-app` for API-only output.

## Configure and deploy

- Change the model, instructions, tools, and framework-native execution in `agent/agent.py`.
- Change `runtime/adapter.py` only to map a different application input/output contract.
- Add local tools under `agent/tools/`; modules are auto-discovered.
- Add MCP servers in `agent/mcps.py` or with `ab tools add mcp`.
- Bind long-term memory with `ab memory bind <store>`.
- Bind durable transcript history with `ab sessions bind <store>`.

```bash
ab --profile <profile> deploy agent-openai --source .
```

When deployment provisions a dedicated Runtime Store, only the app-owned
`databricks_agentkit_runtime_<hash>` schema and runtime tables are added. Managed Runtime Store
deployments use their own default schema.

The `__Host-databricks-app-router` cookie may be supplied independently for sticky replica routing.
It is not authentication and is not used as the template's application session ID.

# Request-user authorization

Declared tools in `agent.toml` select `auth = "user"` or `auth = "app"`; legacy missing auth
continues to use the application/default identity. Deployed user tools require the request resolver
and never fall back to application credentials. Model, memory-service, session-service, and custom
MCP server credentials are unchanged.

`runtime/main.py` derives the invocation policy after `configure()`. The runtime adapter namespaces
public session IDs and actor values for the request owner, then passes only `workspace_client_for`
to the framework-native agent. Internal session keys are never returned to clients.

User-policy invocations use the existing Runtime for synchronous, streaming, and background calls,
including status polling, event replay, and invocation-ID idempotency. The Runtime Store persists no
credential; request-user authentication remains process-local for the active first attempt. A
replacement attempt fails with `MCP_USER_AUTH_RECOVERY_UNSUPPORTED` before agent code runs. Approval
interruptions remain unsupported and fail with `MCP_USER_AUTH_HITL_UNSUPPORTED`, without storing a
credential-bearing `RunState`. Existing namespaced memory and conversation-store behavior is
unchanged; OBO does not add another saver.
