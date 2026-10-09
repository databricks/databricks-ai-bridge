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
   for the same trusted actor. Confirm the bound store contains the intended entry when needed.
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
