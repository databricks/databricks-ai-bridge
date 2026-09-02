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

## Authentication

Mason uses [Databricks authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
If you do not already have credentials, authenticate a named profile first. You can
then ask Mason to validate and remember that profile:

```sh
databricks auth login --profile <profile>
mason login --profile <profile>
mason sessions stores list
```

`mason login` does not create credentials; it stores the selected profile in
`~/.mason/config.json`. `mason logout` forgets that selection without revoking the
underlying credentials. If Databricks SDK default authentication is already configured,
you can skip `mason login`. You can also pass `--profile/-p` for an individual command.
Use `--output json` for scripting.

## Python SDK

The same memory and session APIs are available programmatically through
`MasonClient`, which authenticates exactly like the CLI (a `.databrickscfg` profile
or the SDK's default resolution):

```python
from databricks_mason import MasonClient

client = MasonClient(profile="my-workspace")  # or MasonClient() for default auth

store = client.create_memory_store("my-store")
print(store.name, store.display_name)  # typed attribute access

client.create_memory_entry("my-store", actor_id="alice", path="/notes/1.md", content="hi")
for entry in client.list_memory_entries("my-store", actor_id="alice").entries:
    print(entry.path, entry.content)
```

Each method maps to one `/api/agents/v1` operation. Responses come back as typed
models (`MemoryStore`, `Session`, `SessionItemList`, ...) that expose attribute
accessors (`store.name`) while remaining plain dicts underneath — so `store["name"]`,
`json.dumps(store)`, and any new server-side fields keep working. API errors raise
`databricks_mason.AgentCliError`. Deployment, sandbox, and tracing remain CLI-only.

## Commands

```text
mason [-p <profile>] [-o text|json]
  login        [--profile P]
  logout
  init         [--framework openai|langgraph] [--disable-chat-app]
               [--profile P] [--repo URL] [--ref REF] [directory]
  dev          [--source PATH] [--prepare-environment] [--app-port PORT]
               [--with-memory-store N] [--with-session-store N]
               [--with-traces C.S] [--create-stores]
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
  deploy       <name> --source PATH [--with-memory-store N]
               [--with-session-store N] [--actor-id ID]
               [--with-traces C.S] [--create-stores] [--min-instances N]
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
mason login --profile my-workspace
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
mason tools list
```

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
  --with-session-store mason-demo-sessions \
  --with-memory-store mason-demo-memory \
  --actor-id alice \
  --create-stores
```

The Databricks Apps `__Host-databricks-app-router` cookie is both the sticky routing key and the
application session id. The browser sends it automatically; API clients must reuse it as a cookie.
Request bodies never carry `session_id`. A localhost-only `mason-local-session` cookie provides the
same behavior outside Databricks Apps. TODO: move to `X-Routing-Key` when Apps supports it.

The generated `README.md` documents every request the client makes: config discovery, sync and SSE
invocations, background submission and polling, session transcript loading, HITL resume, and memory
entry operations. Capability colors are automatic from `/api/demo/config`; only the
sync/streaming/background transport selector is manual.
