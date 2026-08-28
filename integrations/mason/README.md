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

## Commands

```text
mason [-p <profile>] [-o text|json]
  login        [--profile P]
  logout
  add
    ui         [--refresh] [directory]
  memory
    stores     create | list | get | update | delete
    entries    create | get | list | search | update | delete
  sessions     create | list | get | update | delete | fork
    stores     create | list | get | update | delete
    items      list | append | pop | clear
  tracing
    setup      --catalog C --schema S [--experiment E]
    list | get | instrument
  init          [--framework openai|langgraph] [--profile P] [DIRECTORY]
  tools
    add sandbox      --scope SCOPE [--scope SCOPE ...] [--source PATH]
    add mcp          SERVICE [--name NAME] [--source PATH]
    add uc-function  FUNCTION [--name NAME] [--source PATH]
    add python       NAME [--source PATH]
    list             [--source PATH]
  add-sandbox  --scope SCOPE [--scope SCOPE ...]
               [--permission read_only|read_write] [--source PATH]
               [--framework openai|langgraph]
  deploy       <name> --source PATH [--with-memory-store N]
               [--with-session-store N] [--actor-id ID]
               [--with-traces C.S] [--create-stores]
  deployments  list | get | logs | start | stop | delete
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

The Python command additionally creates user-owned `agent/tools/<name>.py` and
`tests/tools/test_<name>.py` files using the LangGraph-native `@tool` decorator. `mason dev` and
`mason deploy` preserve `agent.toml`; they do not generate or patch agent source.

Sandbox scopes default to read-only access. Repeat `--scope` to allow more than one resource, use
`volume:` or `workspace:` for those resource types, and use `--permission read_write` only when the
agent needs writes. Every sandbox call carries this fixed downscope in MCP `_meta`, outside the tool
arguments controlled by the model.

`mason add-sandbox` remains as a compatibility alias. For manifest-backed projects it follows the
same LangGraph-only behavior as `mason tools add sandbox`; its older source-editing path remains for
legacy projects that do not yet contain `agent.toml`.

## Add the demo UI

From a LangGraph scratch agent project, add the zero-build browser client:

```sh
cd ./my-agent
mason add ui
uv run start-server
```

Rerun `mason add ui --refresh` to update an existing generated app from the latest Mason-managed UI
template. Refresh intentionally overwrites the installed UI, recovery helper, and demo tests;
`mason dev` and `mason deploy` do not mutate project source.

The UI exercises streaming, sticky background polling, same-ID session resume, local checkpoint
history, managed Memory Store entries, managed Session Store transcript items, agent memory tools,
human approval, and runtime status. Capability dots are automatic: streaming/background reflect the
runtime contract, Session turns green when history is available, and Memory turns green only when
`AGENT_MEMORY_STORE` is configured. Durability and Heartbeat turn green when a managed Session Store
and the stop/start demo are enabled. The transport selector itself is manual.

The generated UI includes a demo-only endpoint that terminates the current app process. A deployed
Databricks App restarts the HTTP process; for a local run, restart `uv run start-server` yourself.
The tool workflow persists LangGraph checkpoints plus an append-only attempt and heartbeat log in
Session Store. Heartbeats default to every 3 seconds and become stale after 10 seconds. Once the old
owner is stale, the browser starts a new attempt with the same public session ID, restores completed
outputs, and resumes at the first incomplete tool.

For the full deployed demo, connect both managed stores:

```sh
mason add ui
mason --profile <profile> deploy mason-agent-demo --source . \
  --with-session-store mason-demo-sessions \
  --with-memory-store mason-demo-memory \
  --actor-id alice \
  --create-stores
```

`mason deploy` provisions or resolves the stores and injects their identifiers plus the shared actor id.
The UI can create, list, and search memory entries for the actor; it also creates a managed session
and mirrors user/assistant turns into Session Store items. It can pause on the sample approval-gated
tool, stop the app process, wait for a new process, and approve the same paused run. The durability
card runs `tool_step_1` through `tool_step_4` in a deterministic checkpointed graph. Stop App after a
few steps complete: after Databricks Apps restarts the process and the previous heartbeat becomes
stale, the browser automatically starts a new attempt, restores completed outputs, skips those
completed nodes, and continues at the first incomplete tool. A tool interrupted before its output
checkpoint is committed can run again.

This scaffold deliberately shows the mechanics rather than presenting Session Store as a complete
durable-task engine. Attempt claims are append-only, last-writer-wins demo leases; they are not an
atomic compare-and-swap across replicas. Production durability additionally needs transactional
ownership, proactive stale-run scanning, idempotent tool side effects, and durable event replay. The
UI reports `atomic_claim: false` so that limitation stays visible during the demo.
