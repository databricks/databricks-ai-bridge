# State, tools, and diagnostics

## Bindings and resource identity

Inspect the project's `agent.toml` before listing workspace resources or creating new ones.
`memory bind`, `sessions bind`, and `tracing bind` declare the resources a project should use.
A declaration does not prove the resource exists or that the deployed app can access it.
Deployment reconciles declared stores and app grants.

Memory store display names are not unique identifiers. Use the store ID or resource name returned
by the CLI; session stores use their store name. Workspace-wide lists can include other projects'
resources. Do not select a store solely because its display name resembles this project.

## Actor, session, and authorization

Derive `actor_id` from trusted application context, not a model-generated value or unvalidated caller
field. Preserve the project's existing identity mapping. Actor partitions are not access-control
boundaries: a principal with store access can reach other actors in that store. Use a separate store
per required security boundary rather than assuming actor filtering provides authorization.

Session Store persists conversation history. Runtime Store tracks invocation state and recovery.
Memory Store holds durable facts for recall across conversations. Do not confuse their identifiers
or assume that changing a binding migrates existing history.

LangGraph can resume checkpoints; OpenAI Agents SDK recovery replays persisted application input
against the session. Preserve approval flows and account for at-least-once tool side effects.
OpenAI HITL `RunState` is process-local even when a Session Store is bound.

## Focused troubleshooting

- **Invocation failure:** inspect the app status, startup/deployment logs, and the project's request
  contract before editing the agent. `endpoint invoke` is a low-level HTTP client, not a payload
  translator.
- **Missing memory/history:** distinguish local dev from deployment, inspect the binding, verify
  the store and app grants, and check actor/session identity before creating a replacement store.
- **Tool failure:** inspect configured `agent.toml` bindings and the requested tool's auth mode.
  `tools list` discovers available workspace integrations; it is not the configured tool inventory.
  Keep user-auth versus app-auth requirements and approvals intact.
- **Tracing:** local dev traces to a local MLflow store; deployment uses the bound workspace
  experiment. Inspect the appropriate store and one relevant invocation, rather than adding a
  second tracing implementation or changing the tracking backend speculatively.
- **Preview/permission error:** report the workspace capability or permission that is missing.
  Do not retry a deterministic failure indefinitely or switch profiles without the user's intent.

Use the relevant subcommand's `--help` for exact current options. Preserve existing state and never
delete an unrelated resource to make a smoke test pass.
