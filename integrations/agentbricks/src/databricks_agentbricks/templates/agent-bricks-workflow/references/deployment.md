# Development, deployment, and verification

## Pick the shortest appropriate loop

For an end-to-end deployment request, scaffold or inspect the project, make the requested changes,
run relevant offline checks, then deploy and validate the deployed behavior. Running
`agentbricks dev` first is optional. For local iteration, run dev and use the local endpoint or chat
UI; do not deploy just to test a local edit.

`agentbricks doctor <directory>` checks onboarding statically and offline. It does not verify
runtime behavior, workspace permissions, deployed resources, or model/tool calls. Run relevant
project tests as well when code changes require them.

`agentbricks deploy <name> --source <directory>` provisions the stores declared in `agent.toml`,
reconciles app access, and deploys a Databricks App. It does not require a running local dev server.
The app name is stable: reusing `<name>` deploys new source to the existing `agent-bricks-<name>` app,
not a separate app. Choose a unique name in the selected workspace for an isolated deployment;
reuse a name only when updating that app is intended.
Provisioning takes real time; do not start repeated deployments merely because one is still running.
Inspect `agentbricks deployments get <app-name>` or logs for a failed deployment before retrying.

Under `agentbricks dev`, Runtime Store invocation state is in memory. Managed conversation
history is replaced by local state and managed long-term memory is off. Do not claim that a local
chat validated deployed session or memory persistence. Deployment is where declared managed stores
are provisioned and wired into the app.

## Bounded proof of life

Choose checks from the user's success criteria rather than exploring every API:

1. Confirm the deployed app's status and URL.
2. Send a representative request with `agentbricks endpoint invoke`; use the project's transport
   contract, not a guessed payload. A successful HTTP status without a useful response is not proof
   of agent behavior.
3. If conversation persistence is required, use the same application `session_id` for two turns
   with distinct invocation UUIDs. Check that the second turn uses the first turn's context.
4. If memory is required, write a distinct test fact and recall it in a separate conversation
   for the same trusted actor. Use the [memory payload example](#cross-conversation-memory-payload)
   for generated templates. Confirm the bound store contains the intended entry when needed.
   Remove only test data you created, and only when cleanup is authorized.
5. If tracing or tools are part of the task, inspect one relevant trace or exercise the requested
   tool. A bindings declaration is not evidence that the deployed principal can access it.

Do not perform live model/tool calls or create/delete resources without authorization. Stop after
the agreed checks pass, or report the concrete failure and remaining limitation.

## Invocation identity

For `DurableAgentServer` templates, use `/api/invocations`. The transport's top-level `id` is an
invocation UUID and idempotency key; `session_id` is the stable application conversation identifier.
Framework-specific input belongs under `input`; consult `AGENTKIT_CONTRACT.md` and `runtime/adapter.py`
for its shape. `X-Routing-Key` only controls sticky routing; it is neither identity nor session state.

### Cross-conversation memory payload

In the generated LangGraph and OpenAI Agents SDK `DurableAgentServer` templates, put `actor` inside
the `input` object alongside `messages`, not at the top level of the invocation:

```json
{
  "id": "11111111-1111-4111-8111-111111111111",
  "session_id": "memory-write",
  "input": {
    "messages": [
      {"role": "user", "content": "Remember this test fact: my test color is teal."}
    ],
    "actor": "memory-test-actor"
  }
}
```

Use fresh invocation UUIDs and a distinct test fact for your authorized check. Once the fact is
stored, send a recall question with a different `session_id`, keeping `input.actor` and the
authenticated identity unchanged. Two turns in the same session test conversation history,
not cross-conversation memory.

A top-level `actor` is rejected by the transport. Message-list shorthand (`"input": [...]`) has no
actor field; the generated adapter defaults actor identity to `session_id`, so changing sessions
also changes the memory partition. Use object-form input for this check. Custom or migrated
adapters may use a different contract; inspect their adapter rather than imposing this example.

The actor field is not authentication. Preserve the application's trusted identity mapping and
[authorization boundaries](state-and-tools.md#actor-session-and-authorization).
