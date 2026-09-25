# Managed runtime integration contract for LangGraph

This document owns the requirements shared by generated projects and existing agents adopting
the managed runtime. The [README](README.md) owns commands and HTTP examples; adapter docstrings own API details.
The adjacent template is an example, not a required graph architecture. Runtime internals live in
the installed package's runtime guide.

## Command requirements

Agent Bricks CLI commands operate on an application directory; they do not rewrite its graph. Each capability
needs runtime wiring as well as manifest configuration.

| Command | Required integration |
| --- | --- |
| `dev`, `deploy` | Installable dependencies and an `app.yaml` command starting the application |
| `tools add mcp`, `uc-function`, `genie`, `sandbox` | Load bindings into model definitions and tool execution |
| `auth connections create`, `auth connections bind` | Call the declared alias through `databricks_agentkit.auth.context` inside an active Agent Bricks invocation |
| `sessions bind` | Select the bound checkpointer and supply session/actor identity |
| `memory bind` | Include bound, actor-scoped memory tools in definitions and execution |
| `tracing bind`, `tracing unbind` | Initialize tracing and open a root span from resolved config |
| `endpoint invoke` | An HTTP endpoint; the caller supplies its path and complete request body |

### Project and startup

Use `agent.toml` for framework, tool/store bindings, and tracing. The existing `.agentbricks/project.toml`
file records framework and template provenance; keep it when working with projects that have it.
Metadata alone does not integrate adapters. Install a compatible `databricks-agentbricks[langgraph]`
distribution, supply the real command in `app.yaml`, load configuration before adapters, and listen
on the app port. The Agent Bricks CLI and runtime find `agent.toml` from the working directory or
`AGENTBRICKS_PROJECT_ROOT`. Keep credentials out of `app.yaml`.

Store binding commands declare intent. `dev` and `deploy` resolve or provision declared stores and
supply runtime config; deploy grants app access. The resolved Memory Store ID reaches the runtime
as `AGENT_MEMORY_STORE`, rather than being stored as an ID in the manifest. Do not hardcode values
that make later bindings ineffective.

### Tools

Use `databricks_agentkit.langgraph.mcp_tools()` for MCP, UC function, and sandbox bindings, combining
them with existing servers without duplicates. It applies sandbox downscoping. Local Python tools
belong in source; the template discovers `BaseTool` objects through
[agent/tools/__init__.py](agent/tools/__init__.py).

Feed selected tools into both model definitions and graph execution while preserving approval
policy. New bindings must become effective on the next graph construction or restart without
another source edit.

### Governed external connections

Entries in `[[connections]]` are agent-owned HTTP or MCP dependencies, not framework tool
definitions. Import `context` with `from databricks_agentkit.auth import context` and call
`context.connections.client(alias).request(...)` inside graph nodes or tools. HTTP requests use a
relative path; MCP requests use an empty path or `/`. Do not accept or forward raw provider tokens,
Databricks authorization headers, cookies, or absolute provider URLs.

`principal = "user"` requires the trusted request-auth context supplied by `DurableAgentServer` and
works during foreground and first-attempt background execution. The caller needs Apps consent and
`USE CONNECTION`. Recovery fails before graph code runs because the credential is intentionally not
persisted; never fall back to App identity. `principal = "app"` uses the App service principal and
deploy grants it `USE_CONNECTION`. Direct UC Connection MCP routing is not a UC MCP Service and must
not be added to `mcp_tools()`.

### Sessions and memory

Use `checkpointer()` for managed state and merge `thread_config(session_id, actor)` into every
graph run. The helper selects the bound Session Store or an in-process saver; explicit arguments
and environment overrides take precedence. It caches the saver per process, so restart after a
binding change. The application state must be serializable by the selected saver.

Include `memory_tools(actor)` in model and execution wiring. It returns no tools when unconfigured.
Its closures capture actor identity, so never reuse them across users. The application owns trusted
actor/tenant authentication; a payload field alone does not establish identity.

### Tracing

Call `configure_tracing()` after loading config, then wrap agent execution, including streams, in
`start_trace(..., session_id=session_id)`. The root span lets framework autologging record children.
The helper enables tracing when destination and experiment settings are present and disables it
otherwise. Preserve existing tracing semantics when composing it.

## Managed server adapter

For the managed invocation protocol, construct `DurableAgentServer` in [runtime/main.py](runtime/main.py)
and register [runtime/adapter.py](runtime/adapter.py) hooks. Keep framework-native execution in
[agent/agent.py](agent/agent.py), independent of runtime request/context types.

The invoke hook translates opaque application input, runs the graph, translates native events,
calls `await context.emit(event)`, and returns JSON output. DurableAgentServer owns foreground/background
transport, polling, and replay. Invocation UUIDs differ from stable application session IDs; the
routing cookie is not the application session.

The example assumes `messages` state and message/update events. Custom state, outputs, and
interrupts require explicit mappings and must not be discarded to fit the example.

### Recovery and durability

Runtime Store persistence covers invocations and emitted events. Session Store persistence covers
graph checkpoints and paused interrupts. A durable Runtime Store alone does not preserve graph
state. `dev` uses process-local invocation storage; deployment attaches a Lakebase-backed
Runtime Store.

Register recovery when intended. Resume a checkpoint only when metadata associates it with the
current invocation; otherwise replay original input. Use synchronous checkpoint durability before
acknowledging progress. Recovery is at least once, so side effects must tolerate replay.

## Optional chat app

No CLI command requires the chat UI. The default overlay supplies it; `--disable-chat-app` omits
it. When adopting it, inspect `CHAT_APP.md`, `runtime/ui.py`, and browser code. Adapt model choice,
history, interrupts, and authentication to the actual graph. Its message-oriented assumptions must
not silently change custom application behavior.
