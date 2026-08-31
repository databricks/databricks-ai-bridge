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
  init         [--framework openai|langgraph] [--enable-chat-app]
               [--profile P] [--repo URL] [--ref REF] [directory]
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
  init          [--framework openai|langgraph] [--profile P] [DIRECTORY]
  tools
    add sandbox      --scope SCOPE [--scope SCOPE ...] [--source PATH]
    add mcp          SERVICE [--name NAME] [--source PATH]
    add uc-function  FUNCTION [--name NAME] [--source PATH]
    list             [--source PATH]
    check            [NAME] [--source PATH]
    run              NAME --input JSON [--source PATH]
  deploy       <name> --source PATH [--with-memory-store N]
               [--with-session-store N] [--actor-id ID]
               [--with-traces C.S] [--create-stores]
  deployments  list | get | logs | start | stop | delete
```

## Agent tools

`mason init` writes portable tool intent to `agent.toml` and template provenance to
`.mason/project.toml`. The LangGraph Python-tool runtime is provided by the
`databricks_mason.python_runtime` SDK module. The in-repository `agent-langgraph` template supplies
the remaining manifest and MCP adapters; `mason tools add` fails explicitly for other frameworks
until they provide an adapter at the same runtime seam.

There are two first-class authoring lanes: write a local Python function and activate it explicitly,
or attach an existing external service with the CLI. Both become exact `agent.toml` records consumed
by the same LangGraph runtime.

### Write and activate a local Python tool

Write normal typed Python directly. For example, create `agent/tools/lookup_ticket.py`:

```python
from langchain_core.tools import tool


@tool
def lookup_ticket(ticket_id: str) -> str:
    """Return a support-ticket summary by ticket ID."""
    return f"Summary for {ticket_id}"
```

Then add the exact activation record to `agent.toml`:

```toml
[[tools]]
id = "lookup-ticket"
source = { kind = "python", entrypoint = "agent.tools.lookup_ticket:lookup_ticket" }
```

Validate the derived contract and invoke the function directly before starting the agent:

```sh
mason tools check lookup-ticket
mason tools run lookup-ticket --input '{"ticket_id":"INC-123"}'
mason dev --source .
mason deploy my-agent --source .
```

An undeclared decorated function is inactive. As an editing aid, `mason tools check` performs a
best-effort scan of literal top-level decorators under `agent/tools/` and emits `MASON001` when one
looks undeclared; the manifest remains the sole activation source. Dynamic decorators and code
outside that directory may not be discovered by the warning scan.

Tests remain ordinary pytest files (for example, `tests/tools/test_lookup_ticket.py`), and project
dependencies remain in the single `pyproject.toml`/lockfile. If a tool needs dependencies that
conflict with the agent environment or requires process isolation, run it behind MCP and attach that
service instead.

### Attach an external tool

The CLI is the supported editor for sandbox, managed MCP, and Unity Catalog function attachments;
these commands update only `agent.toml` and do not generate framework source:

```sh
mason tools add sandbox --scope table:samples.nyctaxi.trips
mason tools add mcp system.ai.web_search
mason tools add uc-function catalog.schema.lookup_ticket
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

For an arbitrary pre-existing MCP endpoint, configure a `DatabricksMCPServer` in the template's
`agent/mcps.py`; Mason does not copy or host that service. `mason dev` and `mason deploy` preserve
`agent.toml` and user code; they do not generate or patch agent source.

Sandbox scopes default to read-only access. Repeat `--scope` to allow more than one resource, use
`volume:` or `workspace:` for those resource types, and use `--permission read_write` only when the
agent needs writes. Every sandbox call carries this fixed downscope in MCP `_meta`, outside the tool
arguments controlled by the model.

## Initialize the chat app demo

The chat app is a LangGraph-specific init overlay, not a command that mutates an existing project:

```sh
mason init --framework langgraph --enable-chat-app \
  --profile <profile> \
  ./my-agent
cd ./my-agent
uv run start-server
```

`--enable-chat-app` always includes synchronous, SSE streaming, background polling, Session Store,
Memory Store, HITL resume, Start App, Stop App, heartbeat, and recovery UI. There are no separate
stop/crash flags. The base agent owns `agent/mason/durability.py`, `agent/mason/recovery.py`, and
`agent/mason/long_running.py`; the framework-specific overlay only adds `ui/`, `runtime/ui.py`, the
UI-enabled `runtime/main.py`, and UI tests.

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
invocations, background submission and polling, session transcript loading, HITL resume, memory
entry operations, and stop/start recovery. Capability colors are automatic from `/api/demo/config`;
only the sync/streaming/background transport selector is manual.

Start App runs `tool_step_1` through `tool_step_4` in a checkpointed sequence. Each completed output
is committed before the next node. Stop App schedules `os._exit(86)`; Databricks Apps restarts the
process, the browser waits for a new instance and a stale heartbeat, and then starts a new attempt
with the same routing cookie. Completed tools are restored and skipped; an interrupted tool whose
output was not committed can run again.

The ownership log is intentionally demo-grade: Session Store records append-only attempts and
heartbeats, but the claim is last-writer-wins rather than atomic (`atomic_claim: false`). Production
durability also needs transactional ownership, server-side stale scanning, idempotent side effects,
and durable event replay.
