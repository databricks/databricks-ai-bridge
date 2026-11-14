# Managed runtime integration contract for a custom framework

`agent/agent.py` accepts plain messages and recalled memory strings. It has no dependency on
Agent Bricks request or context types. `runtime/adapter.py` translates between that native
entrypoint and the managed invocation protocol; customize both sides for your framework's state,
streaming, tools, interrupts, and outputs.

## Sessions and memory

Read the application session only from `InvocationContext.session_id`, supplied by the caller
as a top-level invocation field. Invocation UUIDs and the `X-Routing-Key` routing header are
separate identities. The adapter requires a session and accepts only new user messages under
`input.messages`, rather than accepting a second copy of the whole transcript each turn.

`agentbricks sessions bind` and `agentbricks memory bind` declare bindings in `agent.toml`.
`deploy` resolves them into `AGENT_SESSION_STORE` and `AGENT_MEMORY_STORE` and grants the application
access. `dev` runs with process-local sessions and memory disabled; it does not provision bound
stores. For explicit SDK store testing, run the server directly with those environment variables
and workspace credentials. The adapter uses public, framework-neutral
`AgentKitClient.session_stores` and `.memory_stores` resource APIs directly. A bound Session Store
holds completed conversation turns. Without one, the sample keeps the same records in process
memory and loses them at restart. There is no silent fallback if a bound
store fails.

Memory is opt-in: the adapter retrieves up to five relevant facts before calling the native
agent, and `input.remember` persists a fact at an invocation-specific path. This write omits
`session_id` so the same actor can recall it in another conversation. Repeating the write replaces
the same path. The default actor is the session ID; pass a stable `input.actor` to demonstrate
cross-conversation memory. Actor values partition data and do not authenticate users. Replace
the mapping with your application's trusted tenant/user identity before serving multiple users.
When request-user authentication is enabled, the runtime namespaces sessions and the adapter
namespaces explicit actors by the authenticated principal. Store access uses application credentials.

## Runtime persistence

Runtime Store persistence covers invocation status and emitted events. Session Store persistence
covers the sample's completed conversation turns. This initial scaffold registers an invoke hook;
application recovery and framework-native checkpoints are separate additions. Concurrent session
writes require coordination. Local development storage is process-local.

## Other capabilities

`configure_tracing()` runs at startup and `start_trace()` wraps execution, using tracing settings
resolved by Agent Bricks. `app.yaml` starts the managed server on `PORT` (default 8000).
The scaffold exposes the managed HTTP API without a browser UI. Its output is
`{"output":[{"role":"assistant","type":"ai","content":"...","id":"..."}],"status":"completed"}`;
message events wrap an output item as `{"type":"message","message":...}`.

`agentbricks tools add` does not support custom-framework projects. Wire your framework's Python
tools and MCP clients directly; framework-specific MCP adapters are not included. Human approvals
and resumable framework interrupts likewise need
explicit translations. Keep these limitations visible when extending the example.
