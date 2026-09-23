# Mason LangGraph Agent

A LangGraph agent served by `databricks_mason.DurableAgentServer`. Mason keeps invocation state and events in
memory during `mason dev`. Deployment attaches a persistent Runtime Store, so invocation state and
events survive process loss and interrupted work can be recovered.

The generated project separates portable agent execution from Mason's HTTP protocol:

```text
client -> runtime/main.py -> runtime/adapter.py -> agent/agent.py:run_agent
```

- `agent/agent.py` owns the framework-native agent, sessions, tools, MCP lifetime, and `run_agent`.
- `runtime/adapter.py` owns the agent-author integration hooks: Mason input/output translation plus
  `invoke` and `recover`.
- `runtime/main.py` constructs the server and registers those hooks.

To bring an existing LangGraph agent, run `mason init --framework langgraph --existing .` in its
project and follow the generated prompt in your coding agent. The migration skill and this template share
[MASON_CONTRACT.md](MASON_CONTRACT.md), which owns integration requirements. This README owns
configuration and client examples; [AGENTS.md](AGENTS.md) provides the development map.

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

SSE records contain events translated by `runtime/adapter.py`: token `delta`s, completed `message`s,
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

Deployment with a Runtime Store supports invocation recovery after worker loss. Graph checkpoint
persistence requires a Session Store separately. Recovery can replay work, so tools must tolerate
repeated side effects. See [Recovery and durability](MASON_CONTRACT.md#recovery-and-durability).

## Chat app

The browser UI is included by default. It generates a stable application session ID in local
storage, places it inside each invocation's `input`, and generates a fresh invocation UUID per turn.
Use `mason init --framework langgraph --disable-chat-app` for API-only output.

## Configure and deploy

- Change the model, instructions, tools, graph, and framework-native execution in `agent/agent.py`.
- Change `runtime/adapter.py` only to map a different application input/output contract.
- Add local tools under `agent/tools/`; modules are auto-discovered.
- Add MCP servers in `agent/mcps.py` or with `mason tools add mcp`.
- Bind long-term memory with `mason memory bind <store>`.

```bash
mason --profile <profile> deploy agent-langgraph --source .
```

Deployment provisions or reuses the app's dedicated Runtime Store. Only the app-owned
`databricks_mason_runtime_<hash>` schema and runtime tables are added.

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
interruptions remain unsupported and fail with `MCP_USER_AUTH_HITL_UNSUPPORTED`. The agent's
existing namespaced memory, conversation store, and checkpointer behavior is unchanged; OBO does not
add another saver.
