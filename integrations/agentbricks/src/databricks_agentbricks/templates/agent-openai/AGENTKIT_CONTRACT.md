# Managed runtime integration contract for OpenAI Agents

This document owns the requirements shared by generated projects and existing agents adopting
the managed runtime. The [README](README.md) owns commands and HTTP examples; adapter docstrings own API details.
The adjacent template is an example, not a required agent architecture. Runtime internals live in
the installed package's runtime guide.

## Command requirements

Agent Bricks CLI commands operate on an application directory; they do not rewrite its agent. Each capability
needs runtime wiring as well as manifest configuration.

| Command | Required integration |
| --- | --- |
| `dev`, `deploy` | Installable dependencies and an `app.yaml` command starting the application |
| `tools add mcp`, `uc-function`, `genie`, `sandbox` | Load bindings into agent tool lists and tool execution |
| `sessions bind` | Select the bound session store and supply session/actor identity |
| `memory bind` | Include bound, actor-scoped memory tools in the agent tool list |
| `tracing bind`, `tracing unbind` | Initialize tracing and open a root span from resolved config |
| `endpoint invoke` | An HTTP endpoint; the caller supplies its path and complete request body |

### Project and startup

Use `agent.toml` for framework, tool/store bindings, and tracing. The existing `.agentbricks/project.toml`
file records framework and template provenance; keep it when working with projects that have it.
Metadata alone does not integrate adapters. Install a compatible `databricks-agentbricks[openai]`
distribution, supply the real command in `app.yaml`, load configuration before adapters, and listen
on the app port. The Agent Bricks CLI and runtime find `agent.toml` from the working directory or
`AGENTBRICKS_PROJECT_ROOT`. Keep credentials out of `app.yaml`.

Store binding commands declare intent. `dev` and `deploy` resolve or provision declared stores and
supply runtime config; deploy grants app access. The resolved Memory Store ID reaches the runtime
as `AGENT_MEMORY_STORE`, rather than being stored as an ID in the manifest. Do not hardcode values
that make later bindings ineffective.

The model provider is a choice, not a requirement. The template's [agent/agent.py](agent/agent.py)
wires the Databricks Unity Catalog AI Gateway (`AsyncDatabricksOpenAI(..., use_ai_gateway=True)`
plus a `system.ai.*` model), but an agent may keep its existing provider instead, e.g. direct
OpenAI via `OPENAI_API_KEY` and a model string like `gpt-4o-mini`. Whichever is chosen,
`configure()` sets it up at startup, and deployment must ensure the chosen model is reachable
from the app.

### Tools

Use `databricks_agentkit.openai.mcp_servers()` for MCP, UC function, and sandbox bindings, combining
them with agent-owned servers without duplicates. It applies sandbox downscoping. Genie Agent
bindings arrive as tools from `genie_tools()`. Local Python tools belong in source; the template
discovers `@function_tool` functions through [agent/tools/__init__.py](agent/tools/__init__.py).

Feed selected tools into the agent's tool list while preserving approval policy. Tools that gate on
approval declare `needs_approval=True` and appear in the template's `REQUIRE_APPROVAL` set so
pending calls surface as interrupts. New bindings must become effective on the next agent
construction or restart without another source edit.

### Sessions and memory

Pass `session_store(session_id)` as the `session` argument of `Runner.run(agent, messages,
session=...)` for managed conversation state. The helper selects the bound Session Store or
an in-process session; explicit arguments and environment overrides take precedence. Supply the
actor so the durable store partitions transcripts per user. It caches the in-process session per
process, so restart after a binding change.

Include `memory_tools(actor)` in the agent's tool list. It returns no tools when unconfigured.
Its closures capture actor identity, so never reuse them across users. The application owns
trusted actor/tenant authentication; a payload field alone does not establish identity.

### Tracing

Call `configure_tracing()` after loading config; it binds `mlflow.openai.autolog`. Then wrap agent
execution, including streams, in `start_trace(..., session_id=session_id)`. The root span lets
framework autologging record children. The helper enables tracing when destination and experiment
settings are present and disables it otherwise. Preserve existing tracing semantics when composing
it.

## Managed server adapter

For the managed invocation protocol, construct `DurableAgentServer` in [runtime/main.py](runtime/main.py)
and register [runtime/adapter.py](runtime/adapter.py) hooks. Keep framework-native execution in
[agent/agent.py](agent/agent.py), independent of runtime request/context types.

The invoke hook translates opaque application input, runs the agent through the framework `Runner`,
translates native events, calls `await context.emit(event)`, and returns JSON output. DurableAgentServer owns
foreground/background transport, polling, and replay. Invocation UUIDs differ from stable
application session IDs; the routing cookie is not the application session.

The example assumes `messages` input and message, delta, and interrupt events. Custom inputs,
outputs, and interruptions require explicit mappings and must not be discarded to fit the example.

### Recovery and durability

Runtime Store persistence covers invocations and emitted events. Session Store persistence covers
the conversation transcript, not in-flight run state. The Agents SDK has no node-level checkpoint
continuation: OpenAI HITL `RunState` is process-local and does not survive worker loss even with a
Session Store bound. `dev` uses process-local invocation storage; deployment attaches a
Lakebase-backed Runtime Store.

Register recovery when intended. The recover hook replays the persisted application input against
the same session and prepends a developer instruction warning that the prior attempt may have
partially completed. Recovery is at least once, so side effects must be idempotent and tools must
tolerate replay.

## Optional chat app

No CLI command requires the chat UI. The default overlay supplies it; `--disable-chat-app` omits
it. When adopting it, inspect `CHAT_APP.md`, `runtime/ui.py`, and browser code. Adapt model choice,
history, approval interrupts, and authentication to the actual agent. Its message-oriented
assumptions must not silently change custom application behavior.
