# Bring your own framework or harness

This runnable, model-free echo example hosts your code on `DurableAgentServer`. It demonstrates
managed conversation history, long-term memory, tracing, and whole-turn crash recovery without
requiring LangGraph or the OpenAI Agents SDK. Replace `run_agent` in `agent/agent.py` with your
framework's entrypoint and add its dependencies to `pyproject.toml`. Adjust `runtime/adapter.py`
when your framework needs different inputs, outputs, tools, or checkpoints.

```bash
agentbricks dev
```

The scaffold exposes an HTTP API and has no browser chat app. Send new messages only; the
adapter loads previous turns from the session. Use a fresh UUID for each new turn and keep
`session_id` stable within one conversation:

```bash
curl -sS http://localhost:8000/api/invocations \
  -H 'Content-Type: application/json' \
  -d '{"id":"11111111-1111-4111-8111-111111111111","session_id":"conversation-1","input":{"messages":[{"role":"user","content":"Hello"}]}}'
```

The final invocation response contains an assistant message with `Echo: Hello`. The managed API
also supports background invocation, polling, and event replay. See [AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md)
for state, authentication, recovery, and customization requirements.

## Managed sessions and memory

Bind stores for deployment. Consult the commands for existing-store and provisioning options:

```bash
agentbricks sessions bind --help
agentbricks memory bind --help
```

`agentbricks dev` keeps sessions in process memory and disables managed memory. During deployment,
bindings resolve to `AGENT_SESSION_STORE` and `AGENT_MEMORY_STORE`, and the same adapter persists
turns and completed responses and enables actor-scoped memory search. For direct SDK testing,
run `uv run start-server` with those variables and workspace credentials set explicitly.
On a server with managed memory configured, use `remember` to save a fact:

```json
{
  "id": "22222222-2222-4222-8222-222222222222",
  "session_id": "memory-write",
  "input": {
    "actor": "demo-user",
    "messages": [{"role": "user", "content": "Remember my preference"}],
    "remember": "My favorite color is teal."
  }
}
```

Then send a new invocation with a different `session_id`, the same authenticated identity and
`input.actor`, and `messages: [{"role":"user","content":"What is my favorite color?"}]`.
The echo agent includes relevant retrieved memories in its reply. `memory_query` optionally
overrides the search text. This illustrates cross-conversation memory; ordinary same-session
history uses the Session Store. Replace the demo actor field with trusted identity mapping for
your application.

```bash
uv run pytest
agentbricks --profile <profile> deploy <name> --source .
```

Tests run without workspace credentials or model calls. Deployment attaches persistent Runtime
Store storage; bind a Session Store as well to retain the sample's checkpoints across failures.
Recovery reruns unfinished agent execution at least once, so protect side effects with
idempotency keys. `agentbricks tools add` does not support custom-framework projects; wire your
framework's own tools and MCP clients directly. Approval flows require explicit adapter support.
