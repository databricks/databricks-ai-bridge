# LangGraph agent template

A LangGraph agent served by `databricks_agentkit.DurableAgentServer`. The managed runtime keeps invocation state and events in
memory during `agentbricks dev`. Deployment attaches a persistent Runtime Store, so invocation state and
events survive process loss and interrupted work can be recovered.

The generated project separates portable agent execution from the managed HTTP protocol:

```text
client -> runtime/main.py -> runtime/adapter.py -> agent/agent.py:run_agent
```

- `agent/agent.py` owns the framework-native agent, sessions, tools, MCP lifetime, and `run_agent`.
- `runtime/adapter.py` owns the agent-author integration hooks: runtime input/output translation plus
  `invoke` and `recover`.
- `runtime/main.py` constructs the server and registers those hooks.

To bring an existing LangGraph agent, run `agentbricks init --framework langgraph --existing .` in its
project and follow the generated prompt in your coding agent. The migration skill and this template share
[AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md), which owns integration requirements. This README owns
configuration and client examples; [AGENTS.md](AGENTS.md) provides the development map.

## Run locally

```bash
agentbricks dev
```

The API is available at `http://localhost:8000/api/invocations`. Every POST supplies a stable
application session in `X-Routing-Key` and a UUID `id`. The header is the Runtime and LangGraph
session identity; the UUID identifies one invocation and is its idempotency key. Agent-specific
values live inside the opaque `input` object:

```bash
SESSION_ID=$(uuidgen)
INVOCATION_ID=$(uuidgen)

curl -sS http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -H "X-Routing-Key: $SESSION_ID" \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"What time is it? Use your tool.\"}]}}"
```

Reuse `SESSION_ID` in the header for multi-turn conversation state. Generate a new `INVOCATION_ID`
for each turn. Invocations in one session execute in durable acceptance order; different sessions
can execute concurrently. Retrying the same request and session with the same invocation ID returns
the persisted result; changing the request or session while reusing the ID returns `409`. The
server rejects a missing header instead of using `input.session_id`, the invocation ID, or a random
fallback.

## Invocation modes

- Foreground: omit `background` and `stream`; the response contains the agent result under `output`.
- Foreground streaming: set `stream: true`; the response is SSE backed by persisted events.
- Background: set `background: true`; poll the returned `status_url`.
- Background streaming: set both flags; the `202` response includes `status_url` and `events_url`.

```bash
INVOCATION_ID=$(uuidgen)
curl -sN http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -H "X-Routing-Key: $SESSION_ID" \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"Count to three.\"}]},\"stream\":true}"

INVOCATION_ID=$(uuidgen)
curl -sS http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -H "X-Routing-Key: $SESSION_ID" \
  -d "{\"id\":\"$INVOCATION_ID\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"Summarize durable agents.\"}]},\"background\":true}" | jq
curl -sS "http://localhost:8000/api/invocations/$INVOCATION_ID" \
  -H "X-Routing-Key: $SESSION_ID" | jq
```

SSE records contain events translated by `runtime/adapter.py`: token `delta`s, completed `message`s,
and HITL `interrupt`s. Replay from a cursor with
`GET /api/invocations/{id}/events?after={sequence}`.

## Human approval

`send_message` is gated by `HumanInTheLoopMiddleware`. Start a turn asking the agent to use that
tool. When the output or event stream contains an `interrupt`, submit a new invocation with the same
`X-Routing-Key` header and this body:

```json
{
  "id": "<new-uuid>",
  "input": {
    "resume": {"decisions": [{"type": "approve"}]}
  }
}
```

The default checkpointer is process-local. Bind a managed Session Store to preserve multi-turn and
paused LangGraph state across restarts:

```bash
agentbricks sessions bind my-agent-sessions
```

## Crash recovery

Deployment with a Runtime Store supports invocation recovery after worker loss. Graph checkpoint
persistence requires a Session Store separately. Recovery can replay work, so tools must tolerate
repeated side effects. See [Recovery and durability](AGENTKIT_CONTRACT.md#recovery-and-durability).

## Chat app

The browser UI is included by default. When a conversation starts, it creates a stable application
session ID in local storage, sends it as `X-Routing-Key` on session-scoped requests, and generates a
fresh invocation UUID per turn. It does not duplicate the session inside `input`.
Use `agentbricks init --framework langgraph --disable-chat-app` for API-only output.

## Configure and deploy

- Change the model, instructions, tools, graph, and framework-native execution in `agent/agent.py`.
- Change `runtime/adapter.py` only to map a different application input/output contract.
- Add local tools under `agent/tools/`; modules are auto-discovered.
- Add MCP servers in `agent/mcps.py` or with `agentbricks tools add mcp`.
- Bind long-term memory with `agentbricks memory bind <store>`.

```bash
agentbricks --profile <profile> deploy agent-langgraph --source .
```

When deployment provisions a dedicated Runtime Store, only the app-owned
`databricks_agentkit_runtime_<hash>` schema and runtime tables are added. Managed Runtime Store
deployments use their own default schema.

Send the application session ID in `X-Routing-Key` on every session-scoped call. The required
header is the canonical Runtime and LangGraph session identity and also pins the session to one app
replica. It is not authentication.

# Request-user authorization

Declared tools in `agent.toml` select `auth = "user"` or `auth = "app"`; legacy missing auth
continues to use the application/default identity. Deployed user tools require the request resolver
and never fall back to application credentials. Model, memory-service, session-service, and custom
MCP server credentials are unchanged.

`runtime/main.py` derives the invocation policy after `configure()`. The server namespaces the
public `X-Routing-Key` once for the request owner. The runtime adapter uses
`InvocationContext.session_id` unchanged, namespaces actor values, and passes only
`workspace_client_for` to the framework-native agent. Internal session keys are never returned to
clients.

User-policy invocations use the existing Runtime for synchronous, streaming, and background calls,
including status polling, event replay, and invocation-ID idempotency. The Runtime Store persists no
credential; request-user authentication remains process-local for the active first attempt. A
replacement attempt fails with `MCP_USER_AUTH_RECOVERY_UNSUPPORTED` before agent code runs. Approval
interruptions remain unsupported and fail with `MCP_USER_AUTH_HITL_UNSUPPORTED`. The agent's
existing namespaced memory, conversation store, and checkpointer behavior is unchanged; OBO does not
add another saver.
