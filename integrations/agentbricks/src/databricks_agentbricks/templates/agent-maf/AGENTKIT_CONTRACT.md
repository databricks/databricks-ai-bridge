# AgentKit integration contract

- `agent/agent.py` owns native Agent Framework code; `runtime/adapter.py` owns wire translation.
- Keep `DurableAgentServer` and register both `invoke` and `recover` hooks.
- Read the invocation's top-level `context.session_id`; payloads must not supply runtime IDs.
- Emit each translated delta/message through `context.emit` before returning it to clients.
- Configure managed history with `session_store(actor)` as an Agent `context_provider` and run with
  `AgentSession(session_id=...)`. Do not combine it with another primary history provider.
- Native Message dictionaries are opaque Session Store items. Preserve tool calls and results.
- Compose `memory_tools(actor)`, `genie_tools()`, and request-scoped `mcp_tools()` with local tools.
- Bind model services via `resolve_model_service`; the OpenAI client uses the Databricks AI Gateway.
- The browser imports `read_history` from `agent.agent`; preserve that export.
- Recovery replays the original input and can repeat side effects. Tools must be idempotent.
- Request-user tool auth and approval continuation are unsupported and must fail explicitly.

Invocation example:

```sh
agentbricks endpoint invoke --url http://localhost:8000 --path /api/invocations --json '{"id":"<uuid>","session_id":"demo","input":{"messages":[{"role":"user","content":"Hello"}]}}'
```

Use `agentbricks endpoint invoke --help` to select the local URL or deployed application. The HTTP body is
`{"id":"<uuid>","session_id":"demo","input":{"messages":[{"role":"user","content":"Hello"}]}}`.

History commit and runtime completion are separate operations. If a worker stops after history
commits, recovery can append the original user message and its reply again. These are at-least-once
transcript and execution semantics. The recovery instruction does not enforce idempotence; external
side effects need application-owned idempotency keys or reconciliation. Public input accepts only
new `user` messages with string content; prior assistant/tool history comes from the managed provider.
