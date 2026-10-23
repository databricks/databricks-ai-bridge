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
reuse a name only when updating that app is intended. App names cap at 30 characters, so `<name>`
can be up to 17 lowercase letters, digits, and hyphens. `agentbricks init` records a name that fits
as `[agent].deployment_name` and prints it as `Deploy name`; from the project directory, a bare
`agentbricks deploy` uses it.
Provisioning takes real time; do not start repeated deployments merely because one is still running.
Inspect `agentbricks deployments get <app-name>` or logs for a failed deployment before retrying.

### Deploying from a coding agent

A first deploy takes ~3 min (mostly starting app compute); redeploys are faster. Either run
`agentbricks deploy` in the foreground with a long command timeout (for example, Claude Code's Bash
`timeout: 600000`), or run `agentbricks deploy <name> --no-wait` and then
`agentbricks deployments wait <app-name>` until it exits 0. Exit 75 means it is still deploying;
run the same command again. Exit 1 means it failed: the output carries the error and the log tail.
Don't poll with `sleep`. A second `agentbricks deploy` of the same app is refused while one is still
running from the project.

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
   for the same trusted actor and authenticated identity. Use the project's `AGENTKIT_CONTRACT.md`
   for the request shape and actor mapping; generated managed-server projects include a memory
   payload example there. Confirm the bound store contains the intended entry when needed.
   Remove only test data you created, and only when cleanup is authorized.
5. When the requested behavior uses tools, exercise a representative configured tool through the
   agent and check its result. Tools are part of the agent's logic; a bindings declaration is not
   evidence that the deployed principal can access them. Preserve required approvals.
6. Tracing is enabled by default. Inspect the representative invocation's trace in the configured
   experiment to verify visibility and diagnose failures; do not add unrelated model calls just
   to generate traces. If the project intentionally disables tracing, preserve that choice.

Do not perform live model/tool calls or create/delete resources without authorization. Stop after
the agreed checks pass, or report the concrete failure and remaining limitation.

## Runtime contract

Read the project's `AGENTKIT_CONTRACT.md` for invocation paths, request fields, session and actor
mapping, and routing requirements. Inspect `runtime/adapter.py` for application input translation.
Custom or migrated servers may expose different contracts; use their documented API and adapter
rather than imposing a generated template's payload.
