# `databricks-mason`

Mason is an experimental CLI for Databricks custom agent preview APIs and
deployments. It manages memory, sessions, tracing, and deployments from one
authenticated command.

> The underlying APIs are in preview and may need workspace enablement.

## Prerequisites

- **Python ≥3.10** — the mason CLI installs and runs on any Python 3.10+. The
  `memory`, `sessions`, `tracing`, and `tools` commands need nothing else.
- **[`uv`](https://docs.astral.sh/uv/)** — needed to scaffold, run, and deploy an
  agent (`mason init` → `mason dev` → `mason deploy`): the scaffolded project builds
  its environment and launches with `uv run`, both locally and in the deployed Apps
  runtime. Not needed for the store/session/tracing/tools commands above.
- **[Databricks CLI](https://docs.databricks.com/dev-tools/cli/)** — needed for
  browser-based `mason login`. If a profile is already authenticated, Mason uses it
  directly and the Databricks CLI is optional.

## Installation

From PyPI:

```sh
pip install databricks-mason
```

From source:

```sh
pip install 'git+https://github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/mason'
```

The base package includes the CLI, store SDK, and `AgentApp` HTTP runtime. Generated projects
declare their framework dependencies automatically.

## Shell completion
Add this to `~/.zshrc`:
```sh
eval "$(_MASON_COMPLETE=zsh_source mason)"
```

## Authentication

Mason uses [Databricks authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
Ask Mason to authenticate and remember a named profile:

```sh
mason login --profile <profile>
mason sessions stores list
```

`mason login` validates existing credentials first. If credentials are missing or rejected in
an interactive terminal, Mason runs `databricks auth login --profile <profile>`, revalidates the
profile, and stores the selection in `~/.mason/config.json`. This browser-based setup requires
the Databricks CLI. In non-interactive environments, authenticate the profile before running
Mason. `mason logout` forgets the saved selection without revoking the underlying credentials.

If Databricks SDK default authentication is already configured, you can skip `mason login`.
You can also pass the global `--profile/-p` option before an individual command, for example
`mason --profile <profile> tools list`. Use `--output json` for scripting.

## Quickstart

The shortest path from a blank directory to a running and deployed agent:

```sh
mason login --profile <profile>
mason init my-agent
cd my-agent
mason dev                 # run locally
mason deploy my-agent     # deploy to Databricks
```

`mason dev` runs the agent locally on `http://localhost:8000`, wrapping the Databricks Apps
local runtime so local behavior matches a deployment.

`mason deploy my-agent` deploys a Databricks App named `agent-mason-my-agent`, provisions the
stores declared in `agent.toml`, and grants the app's service principal access to them. `mason
deployments list` shows what you have deployed, and `mason deployments get my-agent` prints its
URL and status.

`mason init` declares default memory and session stores in `agent.toml` (named `<name>-memory` and
`<name>-session`), so the deployed agent has long-term memory and durable conversation history —
`mason deploy` creates them if they don't exist yet. Point the agent at stores you already have with
`mason memory bind <name>` / `mason sessions bind <name>`, or scaffold without stores using
`mason init --server custom` (see [Initialize the chat app demo](#initialize-the-chat-app-demo)).

To exercise the agent — locally under `mason dev` or once deployed — `mason endpoint invoke` sends
it an HTTP request. MLflow tracing is on by default; `mason tracing list` shows the traces it
produces.

## Python SDK

`MasonClient` adds a small resource-oriented layer over the Mason API. Pass it an
authenticated Databricks `WorkspaceClient`, or omit the argument to use the
Databricks SDK's default authentication resolution:

```python
from databricks.sdk import WorkspaceClient
from databricks_mason import MasonClient

mason = MasonClient(WorkspaceClient(profile="my-workspace"))

session_store = mason.session_stores.create("support-agent-sessions")
session = session_store.add(actor_id="customer-123", session_id="case-456")
session.append_items(
    [
        {"type": "message", "role": "user", "content": "I need help with my cluster."},
        {"type": "message", "role": "assistant", "content": "Let's take a look."},
    ]
)

memory_store = mason.memory_stores.create("coding-agent-memory")
memory = memory_store.add(
    actor_id="alice",
    path="/preferences/style.md",
    content="The user prefers concise answers.",
)
results = memory_store.search(
    actor_id="alice",
    query="response preferences",
    limit=10,
)
memory = memory.update(content="The user prefers very concise answers.")
memory.delete()
```

The root collections manage stores: `mason.memory_stores.create/get/list` and
`mason.session_stores.create/get/list`. A returned store owns operations on its
contents, such as `memory_store.add()`, `memory_store.get("memory-id")`,
`memory_store.list()`, and `memory_store.search()`, or `session_store.add()`,
`session_store.get("session-id")`, and `session_store.list()`. Returned memories,
sessions, and stores own their `update()` and `delete()` operations.

All `list()` methods return iterators that automatically consume server pages. List
`page_size` and search `limit` values must be between 1 and 100. `session.list_items()`
also auto-pages. `session.fork(...)` creates an independent copy, optionally through
a specific item. Deleting a session with descendants requires
`session.delete(force=True)` to cascade the deletion.

The resource layer intentionally does not mirror every API method. Its private
transport will be replaced by the generated `WorkspaceClient.mason` service when that
is released, without changing this public surface. Deployment, sandbox, tracing, and
the existing CLI commands remain separate.

## Runtime

`AgentApp` runs your agent through one HTTP API for synchronous, streaming, and background
invocations. Register an `@app.invoke` handler, publish progress with `await context.emit(event)`,
and return a JSON result. You can also add your own FastAPI endpoints.

Start from a template, edit the agent code in `agent/`, and run it locally before deploying:

```sh
mason init my-agent --framework langgraph --server mason --profile <profile>
cd my-agent
mason dev
# Stop the local server when ready to deploy.
mason --profile <profile> deploy my-agent
```

Use `--framework openai` for OpenAI Agents. Templates keep agent code separate from the runtime
adapter and declare default Session and Memory Store bindings in `agent.toml`.

Each managed run is an **invocation**. Send a client-generated UUID `id` and your agent's `input`:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "input": {"messages": [{"role": "user", "content": "Hello"}]},
  "background": true,
  "stream": true
}
```

| Endpoint | Behavior |
| --- | --- |
| `POST /api/invocations` | Defaults to synchronous execution: `200` with the result under `output`. `stream: true` returns SSE events. `background: true` returns `202` with a status URL; adding `stream: true` also includes an events URL. |
| `GET /api/invocations/{id}` | Returns the invocation status and, when completed, its output. |
| `GET /api/invocations/{id}/events?after={cursor}` | Streams events after the last received event ID, allowing clients to reconnect. |

The UUID also acts as an idempotency key: repeating the same request reuses the existing invocation
while its record is retained; using the ID for a different request returns `409`.

`mason dev` keeps execution state in process and loses it on restart. For projects with
`[agent].server = "mason"`, `mason deploy` provisions a persistent Runtime Store for requests,
status, events, and results. Register `@app.recover` to restart interrupted work after worker
failures. Recovery is at-least-once, so external side effects must be idempotent. Session and
Memory Stores separately preserve the state used by your agent.

Use `server = "custom"` to deploy your own HTTP server without provisioning a Runtime Store.
Changing the server type of an existing deployment is not supported. To use a different server,
scaffold a new project with the desired `mason init --server` option and deploy it under a new name.
See the [runtime guide](src/databricks_mason/runtime/README.md) for agent hooks, full API examples,
and recovery behavior.

## Commands

```text
mason [-p <profile>] [-o text|json]
  login        [--profile P]
  logout
  init         [--framework openai|langgraph] [--server mason|custom]
               [--disable-chat-app]
               [--memory-store NAME] [--session-store NAME]
               [--profile P] [directory]
  dev          [--source PATH] [--prepare-environment] [--app-port PORT]
               [--with-traces C.S]
  memory
    bind         STORE [--source PATH]
    unbind       [--source PATH]
    stores     create | list | get | update | delete
    entries    create | get | list | search | update | delete
  sessions     create | list | get | update | delete | fork
    bind         STORE [--source PATH]
    unbind       [--source PATH]
    stores     create | list | get | update | delete
    items      list | append | pop | clear
  tracing
    configure  [--experiment E] [--source PATH]
    disable    [--source PATH]
    list | get
  tools
    add sandbox      --scope SCOPE [--scope SCOPE ...] [--source PATH]
    add mcp          SERVICE [--name NAME] [--source PATH]
    add uc-function  FUNCTION [--name NAME] [--source PATH]
    add genie-one    [--name NAME] [--source PATH]
    add genie-agent  SPACE_ID [--name NAME] [--source PATH]
    list             [--kind sandbox|mcp|uc-function|genie-one|genie-agent]
                     [--schema CATALOG.SCHEMA]
    remove           TOOL_ID [MCP_SERVICE] [--source PATH]
  deploy       <name> --source PATH [--with-traces C.S] [--instances N]
  deployments  list | get | logs | start | stop | delete
  endpoint
    invoke      [APP] --path PATH [--url URL] [--json JSON] [--sse]
```

## Invoke HTTP endpoints

`mason endpoint invoke` is a low-level HTTP command. It resolves and authenticates a deployed
Databricks App, or targets localhost and arbitrary servers through `--url`. It does not assume an
agent protocol: provide the method, path, query parameters, and complete JSON body required by the
server.

```sh
mason --profile <profile> endpoint invoke mason-my-agent \
  --path /api/invocations \
  --json '{"id":"00000000-0000-4000-8000-000000000001","input":[{"role":"user","content":"Hello"}]}'

mason endpoint invoke --url http://localhost:8000 \
  --path /api/invocations \
  --json '{"id":"00000000-0000-4000-8000-000000000001","input":[{"role":"user","content":"Hello"}]}'
```

The JSON body remains explicit even for Mason-generated agents. For example, Mason Runtime agents require
a client-generated invocation ID, and streaming servers require their own streaming field plus
`--sse` so the CLI consumes the response as Server-Sent Events.

```sh
INVOCATION_ID=$(uuidgen)
mason --profile <profile> endpoint invoke mason-my-agent \
  --path /api/invocations \
  --json "{\"id\":\"$INVOCATION_ID\",\"input\":[{\"role\":\"user\",\"content\":\"Run the report\"}]}"

mason --profile <profile> endpoint invoke mason-my-agent \
  --path /api/invocations \
  --sse \
  --json "{\"id\":\"$INVOCATION_ID\",\"input\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true}"
```

`--session-id` preserves one application session across calls by setting the Databricks Apps routing
cookie. This also works with a direct App URL and with the generated runtime on localhost. OAuth and
session headers are managed by Mason; arbitrary custom request headers are intentionally not exposed
by this command.

## Command help

Use the conventional help flag at any command level. Every command's help includes runnable
examples:

```sh
mason --help
mason deploy --help
mason sessions items append --help
```

## Agent tools

For projects with `[agent].server = "mason"` (the default from `mason init`), `agent.toml` is the
declarative source of truth for Databricks-managed infrastructure: the Runtime Store, sandbox,
managed MCP, Genie and Unity Catalog function bindings, plus memory and session resources. `mason tools
add` updates only this file; direct TOML edits have the same behavior. Both Mason-server framework
adapters read the managed bindings at runtime without generating or patching agent source:

```sh
mason tools add sandbox --scope table:samples.nyctaxi.trips
mason tools add mcp system.ai.web_search
mason tools add uc-function catalog.schema.lookup_ticket
mason tools add genie-one
mason tools add genie-agent SPACE_ID
mason tools remove mcp system.ai.web_search
mason tools list
```

For MCP services, the remove command accepts the same service name as the add command. You can also
remove any binding by its `id` in `agent.toml`, for example `mason tools remove web_search`.
Every successful add (including an already-configured no-op) points you to the target project's
`agent.toml` to review configured managed tools and MCP bindings. With `--source`, the message
points to that project's file. JSON add output includes its path in `manifest`.

`mason tools list` discovers **available integrations to add**, not configured bindings. By default
it shows built-in add recipes and caller-visible MCP Services in `system.ai`. A recipe may still
need your resources: sandbox scopes, a concrete UC function name, or a Genie Space ID. Genie One
needs no additional argument. `system.ai.sandbox` is represented by its scoped recipe rather than
a second unscoped add command. The list does not enumerate every workspace schema, individual
operations inside MCP services, or custom Python tools.

```sh
mason tools list
mason tools list --kind mcp
mason tools list --kind mcp --schema main.tools
mason tools list --kind sandbox
mason tools list --kind genie-one
mason tools list --kind genie-agent
mason --output json tools list
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
with TOML inspection. The deprecated, hidden `mason mcp list [--schema catalog.schema]` alias
retains its MCP-only schema-version-1 JSON for compatibility. Use `mason tools list --help` for
the new discovery contract.

Read-only live discovery can be checked against the installed wheel without creating a project
or deploying an agent:

```sh
MASON_E2E_PROFILE=<profile> .venv-functional/bin/pytest tests/e2e/tool_discovery_test.py -v
```

The live checks compare default and MCP-filtered discovery with the compatibility service list.
Set `MASON_E2E_SCHEMA=catalog.schema` to exercise an additional schema. The installed CLI's local
add/review/remove flows and all updated help pages are covered by `tests/functional/cli_smoke_test.py`.

In Mason-server templates, custom Python tools are code-first. Write them with the framework's native
decorator in `agent/tools/`: LangGraph uses `@tool`, while OpenAI Agents uses `@function_tool`. The
templates auto-discover decorated tools from that package and add them to the agent; there is no CLI
command or `agent.toml` entry to keep in sync. Customer-managed MCP servers are likewise ordinary
code in `agent/mcps.py` and are joined with the managed bindings by `mcp_tools(...)` or
`mcp_servers(...)`.

Projects created with `--server custom` do not auto-discover `agent/tools/` or load managed tool
bindings from `agent.toml`, so `mason tools add` rejects those projects. Wire framework-native Python
tools and MCP servers directly in `agent/agent.py` instead.

If an older Mason-server manifest contains `source = { kind = "python", ... }`, remove that
`[[tools]]` entry; the decorated tool in `agent/tools/` remains active. `mason dev` and `mason deploy`
do not generate or patch Python tool code, and do not alter the manifest's `[[tools]]` bindings.

Sandbox scopes default to read-only access. Repeat `--scope` to allow more than one resource, use
`volume:` or `workspace:` for those resource types, and use `--permission read_write` only when the
agent needs writes. Every sandbox call carries this fixed downscope in MCP `_meta`, outside the tool
arguments controlled by the model.

### Genie tools

Genie One and Genie Agent support ship with Mason, but bindings are opt-in, like sandbox tools.
Installing Mason does not configure a Genie Space ID or enable a Genie binding. Add only the
capabilities your agent needs:

```sh
mason tools add genie-one --name genie_one
mason tools add genie-agent SPACE_ID --name genie_agent
mason tools list --kind genie-one
mason tools list --kind genie-agent
mason tools remove genie_one
mason tools remove genie_agent
```

`--name` is optional and defaults to `genie_one` or `genie_agent`, respectively. Both add commands
and `remove` accept `--source PATH` to select a project instead of the current directory. Discovery
needs no project; read that project's `agent.toml` to inspect configured bindings. For scripted
output, put the global `-o json` option before `tools`, as in
`mason -o json tools add genie-one --source ./my-agent`. Adding a binding is offline: it updates
`agent.toml` without contacting Genie or checking permissions. The corresponding sources are:

```toml
[[tools]]
id = "genie_one"
source = { kind = "genie_one" }

[[tools]]
id = "genie_agent"
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

Both framework modules, `databricks_mason.langgraph` and `databricks_mason.openai`, export
`genie_tools()`. New Mason-server templates use it automatically for native Genie Agent bindings;
Genie One uses the existing managed MCP helpers. In an existing Mason-server project, import
`genie_tools` from your framework module and add `*genie_tools()` to the agent's existing tool list.
The CLI does not patch existing Python code.

Both paths use Mason's existing authentication and routed workspace. Genie One requires the
Managed MCP Servers workspace preview; delegated access requires the `genie` OAuth scope.
The effective caller needs access to the data, the SQL warehouse, and the selected Genie space
where applicable. Mason does not grant permissions or promise a service-principal fallback when
caller credentials lack access. An offline add succeeding does not establish runtime access.

The opt-in live tests exercise both frameworks against the configured workspace and an existing
Genie space. From `integrations/mason`, with both framework extras installed:

```sh
DATABRICKS_CONFIG_PROFILE=my-workspace RUN_MASON_GENIE_TESTS=1 \
  MASON_GENIE_SPACE_ID=SPACE_ID \
  uv run pytest tests/integration_tests/genie_tools_test.py
```

By default they ask for the row count of `samples.nyctaxi.trips`. Set `MASON_GENIE_QUESTION` for
another dataset and `MASON_GENIE_EXPECTED_VALUE` to assert a known result cell.

## Initialize the chat app demo

The chat app is a LangGraph-specific init overlay, not a command that mutates an existing project.
It is included by default for `--framework langgraph`; pass `--disable-chat-app` to scaffold the
API-only backend instead.

```sh
mason init --framework langgraph \
  --profile <profile> \
  ./my-agent
cd ./my-agent
mason dev
```

The chat app includes synchronous, SSE streaming, background polling, Session Store, Memory Store,
and HITL resume UI. The framework-specific overlay adds `ui/`, `runtime/ui.py`, the UI-enabled
`runtime/main.py`, and UI tests.

For the full deployed demo, bind both managed stores, then deploy:

```sh
mason sessions bind mason-demo-sessions
mason memory bind mason-demo-memory
mason --profile <profile> deploy mason-agent-demo --source .
```

(`bind` declares the store name in `agent.toml`; `mason deploy` creates any declared-but-missing
store and grants the app's service principal access to it. The memory store id flows to the runtime
via the `AGENT_MEMORY_STORE` env var, injected by `deploy` and `mason dev` — it is not persisted
in `agent.toml`.)

The chat UI generates a stable application session UUID in browser local storage, places it inside
the invocation's opaque `input`, and creates a fresh invocation UUID per turn. The
`__Host-databricks-app-router` cookie remains independent: API clients may reuse it for sticky
replica routing, but it is neither authentication nor the template's application session state.

The generated `README.md` documents every request the client makes: config discovery, sync and SSE
invocations, background submission and polling, session transcript loading, HITL resume, and memory
entry operations. Capability colors are automatic from `/api/demo/config`; only the
sync/streaming/background transport selector is manual.

## Developing Mason

Templates ship **inside** the `databricks_mason` package (`src/databricks_mason/templates/`), so
`mason init` copies the template that matches the installed CLI — the scaffold can't drift from the
`databricks-mason` it runs against.

For an **editable install** (`pip install -e integrations/mason`), two things run straight from your
working tree with no rebuild or commit:

- **CLI** — the `mason` command (`databricks_mason.cli` and the command modules) runs from the
  checkout, since the editable install is the entrypoint.
- **Templates** — `mason init` reads them via `importlib.resources`, which for an editable install
  resolves to the source tree, so editing a template file changes the next scaffold immediately.

```sh
pip install -e integrations/mason     # editable install of the CLI
mason init /tmp/scratch-agent         # scaffolds from your working-tree template
cd /tmp/scratch-agent && mason dev
```

The editable install is one-and-done per venv and follows the working tree, so switching branches
needs no reinstall — **except** a dependency change (a branch that adds or bumps a package in
`integrations/mason/pyproject.toml`), which needs a reinstall to pick it up:

```sh
pip install -e integrations/mason     # only when dependencies changed
```

### Running a scaffold against unreleased Mason (SDK changes)

A scaffold uses a normal `databricks-mason` PyPI dependency, so `mason dev` and `mason deploy`
install the **released** SDK — editing `databricks_mason.runtime`/`.langgraph`/`.openai` in your
checkout does **not** change what a scaffold runs. To exercise local or unreleased SDK changes in a
scaffolded project, add a `[tool.uv.sources]` override to the scaffold's `pyproject.toml`. It's a
dev-loop-only edit — don't ship it in a real deployment.

**`mason dev` — your local checkout (editable, picks up uncommitted edits):**

```toml
[tool.uv.sources]
databricks-mason = { path = "/abs/path/to/databricks-ai-bridge/integrations/mason", editable = true }
```

`mason dev` builds the scaffold's venv from this, so your working-tree SDK edits run live.

**`mason deploy` — a pushed git ref (the Apps build can't reach a local path):**

```toml
[tool.uv.sources]
databricks-mason = { git = "https://github.com/<you>/databricks-ai-bridge", rev = "<pushed-sha>", subdirectory = "integrations/mason" }
```

Commit and push first — the Apps build clones that commit. A `path` or `file://` pin won't resolve
in the build sandbox, so use a git ref (or a released version) for deploys.
