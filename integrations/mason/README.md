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

For tracing commands, install Mason with tracing extras:

```sh
pip install 'databricks-mason[tracing]'
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
session = session_store.create_session(actor_id="customer-123", session_id="case-456")
session.append_items(
    [
        {"type": "message", "role": "user", "content": "I need help with my cluster."},
        {"type": "message", "role": "assistant", "content": "Let's take a look."},
    ]
)

memory_store = mason.memory_stores.create("coding-agent-memory")
memory = memory_store.create_memory(
    actor_id="alice",
    path="/preferences/style.md",
    content="The user prefers concise answers.",
)
results = memory_store.search_memories(
    actor_id="alice",
    query="response preferences",
    limit=10,
)
memory = memory.update(content="The user prefers very concise answers.")
memory.delete()
```

The root collections manage stores: `mason.memory_stores.create/get/list` and
`mason.session_stores.create/get/list`. A returned store owns operations on its
contents, such as `memory_store.create_memory()`,
`memory_store.get_memory("memory-id")`, `memory_store.list_memories()`,
`session_store.create_session()`, `session_store.get_session("session-id")`, and
`session_store.list_sessions()`. Returned memories, sessions, and stores own their
`update()` and `delete()` operations.

All `list()` methods return iterators that automatically consume server pages. List
`page_size` and search `limit` values must be between 1 and 100. `session.list_items()`
also auto-pages. `session.fork(...)` creates an independent copy, optionally through
a specific item, and deleting a session cascades to its descendants.

The resource layer intentionally does not mirror every API method. Its private
transport will be replaced by the generated `WorkspaceClient.mason` service when that
is released, without changing this public surface. Deployment, sandbox, tracing, and
the existing CLI commands remain separate.

## Commands

```text
mason [-p <profile>] [-o text|json]
  login        [--profile P]
  logout
  init         [--framework openai|langgraph] [--disable-chat-app]
               [--profile P] [--repo URL] [--ref REF] [directory]
  dev          [--source PATH] [--prepare-environment] [--app-port PORT]
               [--memory/-m N] [--session/-s N]
               [--with-traces C.S] [--no-create-stores]
  memory
    stores     create | list | get | update | delete
    entries    create | get | list | search | update | delete
  sessions     create | list | get | update | delete | fork
    stores     create | list | get | update | delete
    items      list | append | pop | clear
  tracing
    setup      --catalog C --schema S [--experiment E]
    list | get | instrument
  mcp
    list             [--schema CATALOG.SCHEMA]
  tools
    add sandbox      --scope SCOPE [--scope SCOPE ...] [--source PATH]
    add mcp          SERVICE [--name NAME] [--source PATH]
    add uc-function  FUNCTION [--name NAME] [--source PATH]
    add python       NAME [--source PATH]
    list             [--source PATH]
  deploy       <name> --source PATH [--memory/-m N]
               [--session/-s N] [--actor-id ID]
               [--with-traces C.S] [--no-create-stores]
  deployments  list | get | logs | start | stop | delete
```

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

`mason init` writes portable tool intent to `agent.toml` and template provenance to
`.mason/project.toml`. The manifest runtime is currently implemented only by the in-repository
`agent-langgraph` template; `mason tools add` fails explicitly for other frameworks until they
provide an adapter at the same runtime seam.

Remote tools update only `agent.toml`; they do not generate framework source. The LangGraph runtime
loads the manifest and materializes its native MCP tools when the agent runs, so a direct manifest
edit and a CLI edit have the same behavior:

```sh
mason tools add sandbox --scope table:samples.nyctaxi.trips
mason tools add mcp system.ai.web_search
mason tools add uc-function catalog.schema.lookup_ticket
mason tools add python lookup-ticket
mason tools remove mcp system.ai.web_search
mason tools list
```

For MCP services, the remove command accepts the same service name as the add command. You can also
remove any binding by the ID shown in `mason tools list`, for example `mason tools remove
web_search`. Removal updates only `agent.toml`; Python source and test files remain user-owned.

Discover the MCP Services available to your user before adding one. By default Mason lists the
Databricks-managed services in `system.ai`; pass `--schema catalog.schema` for another Unity Catalog
schema. Text output includes a copyable add command, while `--output json` returns normalized service
records for scripts:

```sh
mason mcp list
mason mcp list --schema main.tools
```

The Python command additionally creates user-owned `agent/tools/<name>.py` and
`tests/tools/test_<name>.py` files using the LangGraph-native `@tool` decorator. `mason dev` and
`mason deploy` preserve `agent.toml`; they do not generate or patch agent source.

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
uv run start-server
```

The chat app includes synchronous, SSE streaming, background polling, Session Store, Memory Store,
and HITL resume UI. The framework-specific overlay adds `ui/`, `runtime/ui.py`, the UI-enabled
`runtime/main.py`, and UI tests.

For the full deployed demo, connect both managed stores:

```sh
mason --profile <profile> deploy mason-agent-demo --source . \
  --session mason-demo-sessions \
  --memory mason-demo-memory \
  --actor-id alice
```

(Missing stores are created automatically; pass `--no-create-stores` to require they already exist.)

The Databricks Apps `__Host-databricks-app-router` cookie is both the sticky routing key and the
application session id. The browser sends it automatically; API clients must reuse it as a cookie.
Request bodies never carry `session_id`. A localhost-only `mason-local-session` cookie provides the
same behavior outside Databricks Apps. TODO: move to `X-Routing-Key` when Apps supports it.

The generated `README.md` documents every request the client makes: config discovery, sync and SSE
invocations, background submission and polling, session transcript loading, HITL resume, and memory
entry operations. Capability colors are automatic from `/api/demo/config`; only the
sync/streaming/background transport selector is manual.
