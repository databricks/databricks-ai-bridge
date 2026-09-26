# Declarative User Auth Scopes Design

## Goal

Allow code-first tools to declare the Databricks Apps OAuth scopes they need when they execute as
the invoking user, while preserving Agent Bricks' automatic scope inference for managed tools.

Source requirement: [Mason Authentication and Governed Tools/MCP](https://docs.google.com/document/d/1XuNfXLc6N5pyaU2sDiZWvORNFyvwzHu52vY_rTNIvII/edit?tab=t.0#heading=h.k0utup5zsql0).

## User contract

An `agent.toml` project may opt into request-user authentication for code-first tools and add the
scopes that Agent Bricks cannot infer from source code:

```toml
[agent]
framework = "langgraph"
server = "agentbricks"

[auth.user]
required = true
additional_api_scopes = ["sql"]

[[tools]]
id = "web_search"
auth = "user"
source = { kind = "mcp", service = "system.ai.web_search" }
```

The resulting configured App scopes are the set union of:

1. scopes inferred from managed bindings with `auth = "user"`;
2. `[auth.user].additional_api_scopes`; and
3. unrelated scopes already configured on an existing App.

The example therefore requests `ai-gateway` for managed web search and `sql` for the custom tool.
Existing App scopes remain preserved by the current reconciliation path.

## Compatibility

- Existing manifests that use managed `auth = "user"` bindings without `[auth.user]` continue to
  enable request-user authentication and managed scope inference.
- App-auth and legacy manifests remain unchanged.
- `[auth.user].required = true` is an additional request-auth trigger, not a new prerequisite for
  managed user-auth bindings.
- Custom Python tools remain code-first. This feature does not restore Python `[[tools]]` manifest
  entries or infer scopes from Python source.
- The feature applies only to `[agent].server = "agentbricks"`, matching the existing request-user
  runtime contract. Custom servers continue to own their authentication wiring.

## Manifest model and validation

A shared, immutable `UserAuthConfig` parser in `databricks_agentkit.runtime.tool_manifest` will be
used by both the comment-preserving CLI model and the runtime policy reader. This avoids divergent
interpretations at deploy and invocation time.

Defaults when `[auth.user]` is absent are `required = false` and no additional scopes. When present:

- `[auth]` and `[auth.user]` must be TOML tables;
- `required` must be a boolean and defaults to `false`;
- `additional_api_scopes` must be an array of non-empty strings and defaults to an empty array;
- duplicate scopes are deduplicated deterministically;
- non-empty additional scopes require `required = true`;
- unsupported extra keys fail validation rather than being ignored.

Scope names are not restricted to a client-side allowlist. Databricks Apps is the source of truth
for supported scopes, and the purpose of this escape hatch is to support APIs not present in the
managed-tool catalog. Local validation rejects empty values, surrounding whitespace, and control
characters so malformed declarations fail before an Apps mutation.

`AgentProject` exposes the parsed configuration as `user_auth`. It preserves comments and existing
manifest formatting because reads do not rewrite the document.

## Deployment behavior

`requires_user_auth(project)` returns true when either:

- a supported managed binding declares `auth = "user"`; or
- `project.user_auth.required` is true.

The existing checks still reject request-user auth for a non-Agent-Bricks server and require every
managed binding to make its identity explicit once any managed binding selects user auth.

`required_user_api_scopes(project)` starts with the explicit additional scope set, then adds the
existing managed-tool inference. The current App update plan remains additive and conservative:

- new Apps receive forwarding and the complete scope set at creation;
- an existing App missing scopes requires `--allow-user-scope-update`;
- unrelated configured scopes are preserved;
- configured and effective scopes must converge before source rollout;
- scope removal remains manual.

Deploy help and error messages refer to declarative user auth rather than requiring a managed tool.

## Runtime behavior

`InvocationAuthPolicy` gains an explicit `user_required` flag in addition to `user_tools`.
`requires_user` is true if either field requires it. `from_manifest()` obtains both values from the
same parsed manifest snapshot:

- `user_tools` retains the IDs of managed bindings that need a user client;
- `user_required` records `[auth.user].required` for code-first consumers.

This ensures a custom-tool-only project fails closed when a deployed invocation lacks the forwarded
user credential. Existing request isolation, token non-persistence, recovery rejection, streaming,
and background-execution behavior remain unchanged.

The generated framework adapters already pass `RequestAuthContext.client_for` through the
request-bound `workspace_client_for` seam. This PR does not add implicit injection into every
auto-discovered decorated tool. The e2e fixture will author a request-bound custom tool factory,
demonstrating the supported code-first integration without widening the SDK surface.

## Testing

### Unit and functional coverage

- CLI and runtime parse identical valid and invalid `[auth.user]` documents.
- Comments survive an `AgentProject` round trip.
- Managed-only, additional-only, combined, and duplicate scope sets produce stable unions.
- Explicit request auth works without managed tools and remains Agent-Bricks-server-only.
- Runtime policy requires a credential for custom-tool-only manifests.
- Existing managed-tool-only behavior stays green.
- Deploy accepts `--allow-user-scope-update` for declarative custom-tool auth and keeps its existing
  conservative reconciliation tests.
- Both generated framework templates continue passing the resolver into request-bound agent code.

### Deployed e2e matrix

The focused matrix covers:

| Axis | Values |
| --- | --- |
| Framework harness | LangGraph, OpenAI Agents SDK |
| Scope source | additional-only custom SQL; combined custom SQL + managed web search |
| Scope | explicit `sql`; inferred `ai-gateway`; combined union |
| Identity control | OBO success on a user-only SQL asset; App principal denied |
| Invocation mode | foreground as the required gate; background/streaming retained where the shared harness supports them |

Each generated project contains a code-first SQL Statements API tool that resolves the user client
through `workspace_client_for("user")`. The combined cases also invoke
`system.ai.web_search`. Evidence records the source commit and wheel hashes, exact configured and
effective App scopes, observed caller identity, positive marker, negative marker absence, terminal
status, logs, and cleanup verification.

Local `agentbricks dev` validates parsing and runtime wiring but cannot prove deployed Apps OAuth
consent, so deployed calls are the acceptance gate.

## Reporting and PR

The e2e evidence is rendered into a dated HTML metric report, published to the standard dogfood
metric-report viewer, and linked in the standalone PR description together with the source design,
exact unit/lint/type-check commands, and live endpoint proof. The implementation, tests, docs,
e2e harness, and report linkage form one self-contained PR because the tests depend directly on the
new manifest and runtime behavior.
