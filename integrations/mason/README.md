# `databricks-mason`

Mason is an experimental CLI for Databricks custom agent preview APIs and
deployments. It manages memory, sessions, tracing, and deployments from one
authenticated command.

> The underlying APIs are in preview and may need workspace enablement.

## Installation

From PyPI:

```sh
pip install databricks-mason
```

From source:

```sh
pip install 'git+https://github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/mason'
```

The CLI installation intentionally excludes HTTP-server and agent-framework dependencies.
Mason-generated projects declare the correct runtime extra automatically. To add Mason Runtime to
an existing agent, install the extra for its framework:

```sh
# LangGraph
pip install 'databricks-mason[runtime]'

# OpenAI Agents SDK
pip install 'databricks-mason[runtime-openai]'
```

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
`mason --profile <profile> mcp list`. Use `--output json` for scripting.

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

## Agent application

`AgentApp` provides Mason's invocation HTTP contract, including foreground, streaming, background,
polling, and event endpoints. By default its state is process-local. When Mason attaches a
Lakebase-backed Runtime Store during deployment, it persists invocation state, heartbeats, and
recovery coordination:

See [Mason Runtime](RUNTIME.md) for the invocation API, streaming and background modes, Runtime
Store lifecycle, and recovery semantics.

```python
from databricks_mason import AgentApp, InvocationContext

app = AgentApp()


@app.invoke
async def invoke(input: object, context: InvocationContext) -> object:
    return await run_agent(input, session_id=context.session_id)


@app.recover
async def recover(input: object, context: InvocationContext) -> object:
    return await recover_agent(input, session_id=context.session_id)
```

The Mason server exposes `POST /api/invocations`, `GET /api/invocations/{invocation_id}`, and
`GET /api/invocations/{invocation_id}/events?after={cursor}`. Databricks Apps bearer-token requests
must use `/api/` routes
([Apps documentation](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/connect-local)).
The client supplies a UUID `id`, which is also the idempotency key for every invocation mode:

- foreground sync returns `200` with the result under `output`;
- background sync returns `202` with a status URL;
- foreground streaming returns `200` server-sent events; and
- background streaming returns `202` with status and event URLs.

`input` and `output` may be any JSON value. Transport fields are not passed to the callback. A
top-level `session_id` is rejected, but a framework template may carry its own stable application
session inside `input`. Polling uses only the invocation ID and relies on Databricks Apps
authentication. Without a Lakebase-backed Runtime Store, request state and events exist only in
the serving process and horizontally scaled clients need sticky routing. With a Lakebase-backed
Runtime Store, Mason persists the input, attempt status, heartbeats, lifecycle events, application
events, and output.

Durability is enabled by default for both framework templates. Mason writes the durability setting
to `agent.toml`, and `mason deploy` reuses or provisions a dedicated `<app>-durability` Lakebase
project. Mason adds its `databricks_mason_runtime_<app-hash>` schema and tables to that database,
giving each app one owned schema. A replacement worker claims a stale heartbeat and calls the
`@app.recover` handler. If that handler is omitted, startup warns that automatic crash recovery
is disabled; register the same function for both decorators when replaying the initial invocation is
safe. Agent checkpoint restoration and idempotent external side effects remain the developer's
responsibility.

Bare `mason init`, `--framework langgraph`, and `--framework openai` scaffold `AgentApp`. Pass
`--server custom` for a minimal FastAPI server with one foreground `/invocations` route and no
Mason Runtime. Use `--disable-chat-app` independently for API-only Mason server output. `AgentApp`
is a FastAPI application, so developers can add their own endpoints alongside Mason's invocation
API.

`mason init` records that choice as `[agent].server = "mason"` or `"custom"` in `agent.toml`.
`mason deploy` uses this field as the source of truth: Mason-server deployments create or reuse an
isolated Runtime Store, while custom-server deployments do not provision one.

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
  mcp
    list             [--schema CATALOG.SCHEMA]
  tools
    add sandbox      --scope SCOPE [--scope SCOPE ...] [--source PATH]
    add mcp          SERVICE [--name NAME] [--source PATH]
    add uc-function  FUNCTION [--name NAME] [--source PATH]
    list             [--source PATH]
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

For the shortest path from a blank directory to a running and deployed agent:

```sh
mason login --profile <profile>
mason init my-agent
cd my-agent
mason dev
mason deploy my-agent
```

## Agent tools

For projects with `[agent].server = "mason"` (the default from `mason init`), `agent.toml` is the
declarative source of truth for Databricks-managed infrastructure: the Runtime Store, sandbox,
managed MCP and Unity Catalog function bindings, plus memory and session resources. `mason tools
add` updates only this file; direct TOML edits have the same behavior. Both Mason-server framework
adapters read the managed bindings at runtime without generating or patching agent source:

```sh
mason tools add sandbox --scope table:samples.nyctaxi.trips
mason tools add mcp system.ai.web_search
mason tools add uc-function catalog.schema.lookup_ticket
mason tools remove mcp system.ai.web_search
mason tools list
```

For MCP services, the remove command accepts the same service name as the add command. You can also
remove any binding by the ID shown in `mason tools list`, for example `mason tools remove
web_search`. `mason tools list` reports these managed bindings; it does not inventory custom code.

Discover the MCP Services available to your user before adding one. By default Mason lists the
Databricks-managed services in `system.ai`; pass `--schema catalog.schema` for another Unity Catalog
schema. Text output includes a copyable add command, while `--output json` returns normalized service
records for scripts:

```sh
mason mcp list
mason mcp list --schema main.tools
```

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
