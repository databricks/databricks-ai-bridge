# Microsoft Agent Framework on Agent Bricks

This template runs a native Microsoft Agent Framework `Agent` through `DurableAgentServer`, with
managed conversation history, long-term memory, the Databricks AI Gateway, MCP tools, and the browser
chat app. Requires Agent Framework core 1.21 / OpenAI integration 1.16 or later within version 1.

```sh
agentbricks init my-agent --framework maf --profile <profile>
cd my-agent
agentbricks dev
uv run pytest
agentbricks deploy <app-name> --source .
```

`agent/agent.py` contains the framework-native agent. Add local `@agent_framework.tool` functions in
`agent/tools/`; they are discovered automatically. `runtime/adapter.py` translates application messages
and emits durable runtime events. Use `--disable-chat-app` to generate only the API backend.

## Managed state

`Agent(context_providers=[session_store(actor)])` uses a native `HistoryProvider` backed by the managed
Session Store. It preserves complete native messages, including tool calls/results, across restarts
and replicas when a store is bound. Without a binding, history stays within the current process.
The actor is captured by the provider; a managed session owned by another actor is rejected.

`memory_tools(actor)` adds `remember`, `recall`, `forget`, and `list_memories` only when a Memory Store
is configured. Facts are scoped to the application-selected actor and persist across conversations.
The model cannot choose the actor. Configure stores with `agentbricks sessions bind <name>` and
`agentbricks memory bind <name>`; the default manifest declares both for deployment.

MCP services, sandbox downscoping, UC functions, Genie One, and native Genie Agent tools are read from
`agent.toml`. MCP connections stay open for the request. This first adapter supports App authentication;
request-user tool bindings fail before execution. Invocation and OpenAI model calls use MLflow tracing
when a destination and experiment are configured.

## Invocation and recovery

Send new messages only; the history provider loads earlier turns:

```json
{"id":"<uuid>","session_id":"stable-session","input":{"messages":[{"role":"user","content":"Hello"}]},"stream":true}
```

`POST /api/invocations` supports foreground, streaming, and background execution. Use a stable
`session_id` and the same value in `X-Routing-Key` for sticky routing; generate a fresh `id` per turn.
`input.actor` optionally selects the application's memory identity; it defaults to the session ID.
Applications must authorize actor/session selection at their boundary.

The runtime persists invocation state/events when a Runtime Store is attached. Recovery replays the
original input against the same conversation with a recovery instruction. It cannot resume within an
Agent Framework tool call; replay can repeat side effects, so tools must be idempotent.

This template does not implement approval continuation or framework workflow checkpoints. `input.resume`
is rejected. Do not add approval-requiring tools without implementing durable pause/decision handling.
The UI's session-history controls work with managed history; approval controls have no pending actions.

Read [AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md) before changing the runtime integration. To test
unreleased SDK code, follow the repository's CONTRIBUTING.md and add a local `[tool.uv.sources]` override.

History commit and runtime completion are separate operations. If a worker stops after history
commits, recovery can append the original user message and its reply again. These are at-least-once
transcript and execution semantics. The recovery instruction does not enforce idempotence; external
side effects need application-owned idempotency keys or reconciliation. Public input accepts only
new `user` messages with string content; prior assistant/tool history comes from the managed provider.
