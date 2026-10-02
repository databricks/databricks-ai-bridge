# Agent tools

For projects with `[agent].server = "agentbricks"` (the default from `agentbricks init`),
`agent.toml` declares Databricks-managed tool bindings. `agentbricks tools add` updates
that manifest; direct TOML edits have the same behavior. Both framework adapters read
the bindings at runtime without generating or patching agent source:

```sh
agentbricks tools add sandbox --scope table:samples.nyctaxi.trips
agentbricks tools add mcp system.ai.web_search
agentbricks tools add uc-function catalog.schema.lookup_ticket
agentbricks tools add genie-one
agentbricks tools add genie-agent SPACE_ID
agentbricks tools remove mcp system.ai.web_search
agentbricks tools list
```

## Choose tool identity

`agentbricks tools add mcp`, `agentbricks tools add sandbox`, `agentbricks tools add genie-one`, and
`agentbricks tools add genie-agent` write explicit `auth = "user"` by default.
Use `--auth app` for the App service principal instead. This field is on the tool entry, not
inside `source` or `policy`:

```toml
[[tools]]
id = "web_search"
auth = "user"
source = { kind = "mcp", service = "system.ai.web_search" }
```

Direct UC-function bindings remain app/default identity and do not accept `--auth user`.
Managed-tool add commands write the selected identity to `agent.toml`; inspect that manifest to
review configured bindings. Missing legacy auth continues to mean App identity at runtime; it is
never silently upgraded to user identity.

## Automatic App-identity access on deploy

`agentbricks deploy` reconciles least-privilege access for resources explicitly declared by App/default
identity tool bindings. It skips every `auth = "user"` binding because those calls use the request
user's permissions instead of the App service principal.

| Explicit `agent.toml` resource | Automatic App service-principal access |
| --- | --- |
| UC function | Apps `uc_securable`: `FUNCTION` / `EXECUTE` |
| Genie Agent space | Apps `genie_space`: `CAN_RUN` |
| Sandbox table scope | Apps `uc_securable`: `TABLE` / `SELECT` or `MODIFY` |
| Sandbox volume scope | Apps `uc_securable`: `VOLUME` / `READ_VOLUME` or `WRITE_VOLUME` |
| Sandbox Workspace path | Workspace ACL: `CAN_READ` or `CAN_EDIT` |
| External MCP service | Unity Catalog: effective `EXECUTE` plus `USE_SCHEMA` and `USE_CATALOG` on its named parents |
| Built-in `system.ai` MCP service, including Sandbox and Genie One | Platform-managed access defaults; Agent Bricks does not mutate system securables |

Native Genie One has no resource identifier in its binding, so it does not add a resource-specific
grant. Use a Genie Agent binding when the App identity should be scoped to one explicit Genie Space.

Apps-backed tool resources are named deterministically and reconciled to the manifest on each
deploy: removing a binding removes that Agent Bricks-owned Apps resource while preserving Runtime
Store, tracing, and user-owned resources. MCP and Workspace ACL grants are additive in this release
because their permission APIs do not expose trustworthy Agent Bricks ownership metadata; removing
those bindings does not revoke an independently valid grant.

Only direct resources are automatic. Agent Bricks never discovers or grants tables and warehouses
used by a Genie Space, objects called by a UC function, or resources wrapped by an MCP service.
Grant those transitive dependencies manually when the called service uses the App identity. UC and
Workspace grant checks, plus initial App-resource attachment, happen before source upload. Final
Agent Bricks-owned App-resource reconciliation runs after rollout and can fail after the source has
been uploaded.

## Request-user authentication

`DurableAgentServer` derives its request-auth policy directly from the managed tool bindings in
`agent.toml`. Projects do not maintain a separate request-auth contract marker: a managed tool with
`auth = "user"` requires a transient request-user credential. Code-first tools can declare any
additional API scopes that Agent Bricks cannot infer from Python:

```toml
[auth.user]
required = true
additional_api_scopes = ["sql"]
```

`additional_api_scopes` is additive: deploy unions it with scopes inferred from managed bindings,
deduplicates the result, and preserves unrelated scopes already configured on the App. Scope names
are not restricted to a client-side allowlist; Databricks Apps validates whether a requested scope
is supported. Entries must be non-empty strings without surrounding whitespace or control
characters, and a non-empty list requires `required = true`. This request-auth contract is supported
only with `[agent].server = "agentbricks"`.

Generated framework adapters pass the request-bound resolver to agent construction. A code-first
tool should obtain its user client from that resolver inside the active invocation rather than
creating or persisting a user credential:

```python
from langchain_core.tools import tool


def sql_tools(workspace_client_for):
    @tool
    def run_statement(statement: str) -> str:
        client = workspace_client_for("user")
        response = client.statement_execution.execute_statement(
            warehouse_id="...",
            statement=statement,
        )
        return str(response.result)

    return [run_statement]
```

The resolver is request-bound and closes after the attempt. Agent Bricks does not inject it into
arbitrary auto-discovered decorated tools; build those tools from the resolver passed to the
generated request-aware agent function.

Request-user invocations use the same synchronous, streaming, background, status, event-replay,
and idempotency APIs as app-auth invocations. The Runtime Store records only token-free request
state, events, and results. The forwarded credential stays process-local for the active first
attempt and closes when that attempt completes, fails, or is cancelled. A replacement attempt after
failure recovery stops with `MCP_USER_AUTH_RECOVERY_UNSUPPORTED` because no user credential is
available; neither the invoke nor recovery handler runs for that attempt.

Before deploying user-auth tools from an older project, migrate its request handler and framework
adapter to the current request-auth-aware `DurableAgentServer` template, then explicitly choose `user` or
`app` on **every** managed MCP, sandbox, or Genie entry. Changing `agent.toml` alone does not
upgrade copied Python adapter code. Outdated adapters fail closed rather than silently using App
identity. App-only legacy projects and generic bring-your-own source directories keep the existing
path.

### Apps user scopes

Deploy derives Apps user scopes from explicit `auth = "user"` bindings and unions them with
`[auth.user].additional_api_scopes`:

| Binding | Requested Apps scopes |
| --- | --- |
| Managed MCP (governed ingress) | `ai-gateway` |
| `system.ai.dbsql` | `ai-gateway`, `sql` |
| `system.ai.genie_one_mcp` | `ai-gateway`, `genie` |
| Sandbox with a Volume downscope | `ai-gateway`, `files` |
| First-class Genie One or Genie Agent | `genie` |
| Sandbox with token injection | `ai-gateway`, `workspace.workspace` |
| Sandbox with token injection disabled | `ai-gateway` |

For example, bind Genie tools in a current project with `server = "agentbricks"`:

```sh
agentbricks tools add mcp system.ai.genie_one_mcp --auth user
agentbricks tools add genie-agent SPACE_ID --auth user
```

Mixed bindings and explicit additions request the union. App-auth and legacy bindings add no user
scopes. These are
explicit service-consent scopes, not a claim that gateway access alone authorizes the downstream
resource. OAuth consent does not grant Unity Catalog privileges: the user still needs access to
the configured Genie Space and its underlying data.

For a new App, deploy explicitly enables user-token forwarding and includes these scopes in the
initial typed SDK create request before uploading source. An existing App that is missing a required
scope needs one-time explicit permission:

```sh
agentbricks --profile my-workspace deploy my-agent --allow-user-scope-update
```

The `system.ai.dbsql` managed MCP additionally requests the Apps `sql` user scope. This is full SQL
API consent, not `sql:restricted-query`; read-only enforcement remains the service policy plus the
requesting user's Unity Catalog grants. DBSQL does not use Databricks Connect.

When a user-auth sandbox has `databricks_access_token_included = true`, it requests the Apps
`workspace.workspace` user scope so the injected credential can call workspace APIs. A sandbox
binding with a Volume downscope additionally requests the Apps `files` user scope.
OAuth consent does not grant Volume access: the requesting user still needs the corresponding
Unity Catalog privileges, and the sandbox downscope remains authoritative. A sandbox binding with
token injection disabled does not request `workspace.workspace`; its other resource-derived scopes
still apply. Databricks Apps rejects the legacy bare `workspace` scope, so Agent Bricks requests
`workspace.workspace`. These scopes are requested only for `auth = "user"`; `auth = "app"` uses
the App service principal's permissions instead.

Review the target App's scopes and coordinate with its other owners before allowing the update. Once
those scopes are present, later deploys do not need the flag. The CLI preserves unrelated scopes,
updates only user scopes and any explicitly requested instance counts, and checks requested **and
effective** scopes before source rollout. It checks for scope changes since preflight, but Apps
read/write is **not atomic**; this is not a lock or a compare-and-swap guarantee. Polling is bounded
and a mismatch stops source deployment.
Users may need to sign out and **re-consent** after changing scopes; effective-scope verification
does not refresh an existing user's consent.

Apps may report `iam.access-control:read` and `iam.current-user:read` as implicit effective
scopes. The CLI permits these platform defaults during verification but does not request them
as configurable scopes. Explicitly disabled user-token forwarding stops deployment; enable
forwarding and restart the App compute before retrying.

Removing a tool or switching back to app-only auth **does not remove Apps scopes**. Remove
unneeded scopes explicitly in Databricks Apps, and verify both configured and effective scopes
before declaring removal complete. The CLI does not send empty-list scope updates: the SDK's
`App.as_dict()` omits empty lists, so that would not prove removal succeeded. No scopes are
managed for generic bring-your-own apps without this managed user contract.

App-auth tools execute with workload privileges. Restrict App `CAN USE` to callers trusted
for **all** App-auth tools, or deploy those tools separately. Models, custom MCP servers,
Memory/Session Stores, and tracing keep their existing credentials.

## Discover and manage bindings

For MCP services, the remove command accepts the same service name as the add command. You can also
remove any binding by its `id` in `agent.toml`, for example `agentbricks tools remove web_search`.
Every successful add (including an already-configured no-op) points you to the target project's
`agent.toml` to review configured managed tools and MCP bindings. With `--source`, the message
points to that project's file. JSON add output includes its path in `manifest`.

`agentbricks tools list` discovers **available integrations to add**, not configured bindings. By default
it shows built-in add recipes and caller-visible MCP Services in `system.ai`. A recipe may still
need your resources: sandbox scopes, a concrete UC function name, or a Genie Space ID. Genie One
needs no additional argument. `system.ai.sandbox` is represented by its scoped recipe rather than
a second unscoped add command. The list does not enumerate every workspace schema, individual
operations inside MCP services, or custom Python tools.

`agentbricks tools add mcp` looks up the service in the selected workspace before writing `agent.toml`.
Use `agentbricks --profile <profile> tools add mcp <service>` to select a workspace. A missing service or
failed lookup (including authentication or permission errors) leaves the project unchanged. This
checks service metadata access, not whether every tool can be executed at runtime. Removing local
bindings does not require workspace access.

```sh
agentbricks tools list
agentbricks tools list --kind mcp
agentbricks tools list --kind mcp --schema main.tools
agentbricks tools list --kind sandbox
agentbricks tools list --kind genie-one
agentbricks tools list --kind genie-agent
agentbricks --output json tools list
```

No agent project is required for discovery. MCP discovery uses your Databricks profile; the
`sandbox`, `uc-function`, `genie-one`, and `genie-agent` kind filters show local recipes without
authentication. `--schema` requires `--kind mcp` and replaces the default `system.ai` scope. An
API/authentication failure returns nonzero and marks discovery incomplete, while retaining local
recipes; it is not reported as an empty successful discovery. Listing metadata does not verify
runtime execution permissions.

**Migration:** the former configured `tools list` view and its `--source` option are removed.
Read `agent.toml` (its `[[tools]]` entries) to inspect configured bindings. Discovery JSON uses
`schema_version: 2`, with `available_tools` (`name`, `kind`, `add_command`), `mcp_schema` (null for
local-only recipes), `complete`, and `errors`. Replace old scripts that read configured-list JSON
with TOML inspection. Replace `agentbricks mcp list [--schema catalog.schema]` with
`agentbricks tools list --kind mcp [--schema catalog.schema]`; the former command is removed. Use
`agentbricks tools list --help` for the new discovery contract.

## Custom Python tools

In managed-server templates, custom Python tools are code-first. Write them with the framework's native
decorator in `agent/tools/`: LangGraph uses `@tool`, while OpenAI Agents uses `@function_tool`. The
templates auto-discover decorated tools from that package and add them to the agent; there is no CLI
command or `agent.toml` entry to keep in sync. Customer-managed MCP servers are likewise ordinary
code in `agent/mcps.py` and are joined with the managed bindings by `mcp_tools(...)` or
`mcp_servers(...)`. See the [custom-tool quickstart](../README.md#add-a-custom-tool)
for a working example in both frameworks.

Projects created with `--server custom` do not auto-discover `agent/tools/` or load managed tool
bindings from `agent.toml`, so `agentbricks tools add` rejects those projects. Wire framework-native Python
tools and MCP servers directly in `agent/agent.py` instead.

If a manifest with `server = "agentbricks"` contains `source = { kind = "python", ... }`, remove that
`[[tools]]` entry; the decorated tool in `agent/tools/` remains active. `agentbricks dev` and `agentbricks deploy`
do not generate or patch Python tool code, and do not alter the manifest's `[[tools]]` bindings.

## Sandbox policy

Sandbox scopes default to read-only access. Repeat `--scope` to allow more than one resource, use
`volume:` or `workspace:` for those resource types, and use `--permission read_write` only when the
agent needs writes. Every sandbox call carries this fixed downscope in MCP `_meta`, outside the tool
arguments controlled by the model. New sandbox bindings also expose the selected Databricks
credential to sandbox code by default:

```toml
[[tools]]
id = "sandbox"
auth = "user"
source = { kind = "sandbox", service = "system.ai.sandbox" }
policy = { downscope = [{ resource = "workspace:/Workspace/Shared", permission = "read_only" }], databricks_access_token_included = true }
```

With `databricks_access_token_included = true`, the sandbox receives `DATABRICKS_HOST`, a short-lived
`DATABRICKS_TOKEN`, and `DATABRICKS_AUTH_TYPE`, so code such as
`WorkspaceClient().current_user.me()` can call workspace APIs. This policy does not choose the
identity: `auth = "user"` uses the request user's OBO credential, while `auth = "app"` uses the
Databricks App service principal. Use `--no-databricks-access-token-included` when adding a sandbox that
does not need workspace API access. Existing manifests that omit `databricks_access_token_included`
remain disabled until explicitly updated.

## Genie tools

Genie One and Genie Agent support ship with Agent Bricks CLI, but bindings are opt-in, like sandbox tools.
Installing the current `databricks-agentbricks` distribution does not configure a Genie Space ID or enable a Genie binding. Add only the
capabilities your agent needs:

```sh
agentbricks tools add genie-one --name genie_one --auth user
agentbricks tools add genie-agent SPACE_ID --name genie_agent --auth user
agentbricks tools list --kind genie-one
agentbricks tools list --kind genie-agent
agentbricks tools remove genie_one
agentbricks tools remove genie_agent
```

`--name` is optional and defaults to `genie_one` or `genie_agent`, respectively. `--auth` defaults
to `user`; choose `--auth app` deliberately for App service-principal execution. Existing manifests
without `auth` preserve App/default identity. Both add commands and `remove` accept `--source PATH`
to select a project instead of the current directory. Discovery needs no project; read that
project's `agent.toml` to inspect configured bindings. For scripted output, put the global
`-o json` option before `tools`, as in
`agentbricks -o json tools add genie-one --source ./my-agent`. Adding a binding is offline: it updates
`agent.toml` without contacting Genie or checking permissions. The corresponding sources are:

```toml
[[tools]]
id = "genie_one"
auth = "user"
source = { kind = "genie_one" }

[[tools]]
id = "genie_agent"
auth = "user"
source = { kind = "genie_agent", space_id = "<your-space-id>" }
```

Replace `SPACE_ID` or `<your-space-id>` with an existing space's 32-character lowercase hexadecimal
ID. `genie-one` connects to the workspace-wide MCP endpoint
`https://<workspace-hostname>/api/2.0/mcp/genie`, without a space suffix. `genie-agent` uses the
native Genie **Chat-mode** conversation API through the Databricks SDK, not the streaming
Agent-mode API or the per-space MCP endpoint.

Each native binding exposes `{id}_ask`, `{id}_poll`, and `{id}_query_result`, where `{id}` is its
binding name. Ask accepts an optional `conversation_id` for follow-ups. Ask and poll share a
120-second budget per call, including client setup and submission. If the response is still
running, they return `timed_out` with the conversation and message IDs so the caller can poll
again. If submission times out before a message ID is received, ask returns
`INDETERMINATE_SUBMISSION`: the request may still complete, so do not resubmit automatically.
`NOT_SUBMITTED` means client setup timed out before sending the question. Query results include
the first 100 rows, column schema, a truncation indicator, and a deep link to the conversation.

Both framework modules, `databricks_agentkit.langgraph` and `databricks_agentkit.openai`, export
`genie_tools()`. New managed-server templates (`server = "agentbricks"`) use it automatically for native Genie Agent bindings;
Genie One uses the existing managed MCP helpers. In an existing project with `server = "agentbricks"`, import
`genie_tools` from your framework module and add `*genie_tools()` to the agent's existing tool list.
The CLI does not patch existing Python code.

Both paths use Databricks authentication and the routed workspace. Genie One requires the
Managed MCP Servers workspace preview; delegated access requires the `genie` OAuth scope.
The effective caller needs access to the data, the SQL warehouse, and the selected Genie space
where applicable. Agent Bricks CLI does not grant permissions or promise a service-principal fallback when
caller credentials lack access. An offline add succeeding does not establish runtime access.
