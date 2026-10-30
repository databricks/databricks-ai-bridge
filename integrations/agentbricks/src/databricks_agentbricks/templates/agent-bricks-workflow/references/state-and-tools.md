# State, tools, and diagnostics

## Bindings and resource identity

Inspect the project's `agent.toml` before listing workspace resources or creating new ones.
`memory bind`, `sessions bind`, and `tracing bind` declare the resources a project should use.
A declaration does not prove the resource exists or that the deployed app can access it.
Deployment reconciles declared stores and app grants.

Memory store display names are not unique identifiers. Use the store ID or resource name returned
by the CLI; session stores use their store name. Workspace-wide lists can include other projects'
resources. Do not select a store solely because its display name resembles this project.

## Store permissions

Use `agentbricks sessions stores permissions` or `agentbricks memory stores permissions` for
focused access checks and changes; inspect the relevant subcommand's `--help` for options.
Session stores use their store name or resource name. Memory stores use an ID or resource name;
permission `get` also supports the existing display-name lookup and rejects ambiguous matches.

- `get STORE` reports the authenticated caller's effective permission, not the deployed app's.
  To verify app access, authenticate as the app's service principal when authorized, or validate
  its actual store operations. READ permits reads; WRITE includes reads and writes.
  PERMISSION_LEVEL_UNSPECIFIED means no access and is shown as "No access" in text output.
- `list STORE` shows explicit user and service-principal grants. Implicit ownership and inherited
  access are excluded, so an empty list does not establish that nobody can access the store.
  Use `--all` when the full grant inventory is needed.
- `grant STORE` and `revoke STORE` take exactly one `--user` email or `--service-principal`
  application/client ID. A numeric SCIM principal ID is not the application ID.
  Grant requires `--permission READ` or `--permission WRITE`; granting READ preserves existing WRITE.
  To downgrade WRITE to READ, revoke the principal and then grant READ. This is not atomic: the
  principal has no access between those two commands.
- Grant, List, and Revoke require the store creator or a workspace admin; a WRITE grantee cannot
  manage permissions. Revoke removes both READ and WRITE grants, is safe to repeat, and cannot
  revoke ownership. Inherited access is not removed. Use `--yes` only when revocation is intended.

Deployment reconciles the app's grants; a later deploy can restore access previously revoked from
that app. Unbinding a store does not revoke existing access. Preview or disabled-feature errors
require workspace enablement; changing the CLI or repeating the request will not enable an API.

## Actor, session, and authorization

Derive `actor_id` from trusted application context, not a model-generated value or unvalidated caller
field. Preserve the project's existing identity mapping. Actor partitions are not access-control
boundaries: a principal with store access can reach other actors in that store. Use a separate store
per required security boundary rather than assuming actor filtering provides authorization.

Session Store persists conversation history. Runtime Store tracks invocation state and recovery.
Memory Store holds durable facts for recall across conversations. Do not confuse their identifiers
or assume that changing a binding migrates existing history.

Read the project's `AGENTKIT_CONTRACT.md` and adapter for its actor mapping, recovery behavior,
and approval-state durability. For cross-conversation memory checks, keep the trusted actor and
authenticated identity stable while changing the conversation and invocation IDs; use the
project's documented payload rather than assuming a shared framework contract. Preserve approval
flows and account for at-least-once tool side effects; persisted conversation history alone does
not prove that paused execution or approvals survive worker loss.

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
