# Mason CLI command reference

`mason` is the command-line interface for building and deploying custom AI agents on Databricks. It
scaffolds an agent project from a template, runs it locally with a chat UI, deploys it to Databricks
Apps, and manages the tools, memory, sessions, and tracing behind it - all from one authenticated
command.

This page is the full command reference: every command, subcommand, argument, and option. For
concepts, guides, and the Python SDK, see the [README](README.md). Every command also has built-in
help - append `--help` (or `-h`) at any level, for example `mason deploy --help` or
`mason sessions items append --help`.

> **Preview:** Mason is experimental. The CLI, its commands, and the underlying agent APIs are all in
> preview, may need to be enabled for your workspace, and are likely to change in
> backward-incompatible ways.

## Installation

```sh
pip install databricks-mason
```

See [Installation](README.md#installation) for installing from source and for shell completion.

## Authentication

Mason authenticates with a [Databricks configuration profile](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
Run `mason login` once to save a default profile, or pass `--profile` / `-p` on any command. Without
a profile, the Databricks SDK's default authentication resolution is used. See
[Authentication](README.md#authentication) for details.

## Global options

These options apply to every command. Pass them before the command name, for example
`mason -p my-profile -o json sessions stores list`.

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--profile <PROFILE>` (`-p`) | string | - | `~/.databrickscfg` profile to authenticate with. |
| `--output <text\|json>` (`-o`) | `text` \| `json` | `text` | Output format. Use `json` for scripting. |
| `--version` | flag | - | Show the version and exit. |
| `--help` (`-h`) | flag | - | Show help for the command and exit. Works at every level. |

## How to read this reference

- **Arguments** are positional; **options** are named flags. In a synopsis, `NAME` is required and
  `[NAME]` is optional; `<command>` marks a group that requires a subcommand; `[options]` stands in
  for the option list documented below each command.
- The **Required** column marks whether an argument or option must be provided.
- The **Default** column shows the value used when an option is omitted (`-` means none).
- Commands that delete or replace data prompt for confirmation; pass `--yes` / `-y` to skip the
  prompt in scripts.

## Command summary

| Command | Description |
| --- | --- |
| [`login`](#mason-login) | Authenticate and save a default profile |
| [`logout`](#mason-logout) | Forget the saved default profile |
| [`init`](#mason-init) | Scaffold a new agent project |
| [`dev`](#mason-dev) | Run the agent locally with a chat UI |
| [`memory`](#mason-memory) | Manage an agent's long-term memory |
| [`mcp`](#mason-mcp) | Discover managed MCP services |
| [`sessions`](#mason-sessions) | Manage an agent's conversation sessions |
| [`tracing`](#mason-tracing) | Set up and inspect agent tracing |
| [`deploy`](#mason-deploy) | Deploy an agent to Databricks Apps |
| [`deployments`](#mason-deployments) | Manage deployed agents |
| [`endpoint`](#mason-endpoint) | Invoke arbitrary HTTP endpoints. |
| [`tools`](#mason-tools) | Manage an agent's tools |

## Commands

### `mason login`

Authenticate a profile and save it as the default, so later commands can omit -p.

```
mason login [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--profile <PROFILE>` (`-p`) | string | - | no | Profile to authenticate with and remember as the default. |

### `mason logout`

Forget the saved profile selection without deleting its credentials.

```
mason logout
```

### `mason init`

Scaffold a local agent project from a mason template.

DIRECTORY is the target path to create (defaults to the template's own name). The directory must not already exist. Once scaffolded, deploy it with `mason deploy <name> --source <directory>`.

Pass --profile (or set a default via `mason login` / -p) to seed a local `.env` so the scaffolded project runs with `mason dev` right away.

The scaffold is preconfigured to call Databricks model serving through the AI Gateway using that profile, so it can talk to a model with no separate endpoint or API key to set up.

The default Mason server supports foreground, streaming, and background invocations through one HTTP contract and Runtime Store. Pass --server custom for a minimal foreground-only FastAPI server.

```
mason init [DIRECTORY] [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `DIRECTORY` | no | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--framework <langgraph|openai>` | `langgraph` \| `openai` | - | no | Agent framework to scaffold (defaults to langgraph). |
| `--server <mason|custom>` | `mason` \| `custom` | `mason` | no | Use Mason's invocation server or a minimal custom FastAPI server. |
| `--profile <PROFILE>` | string | - | no | Seed a local .env with this DATABRICKS_CONFIG_PROFILE so `mason dev` works immediately (defaults to the profile from -p / `mason login`). |
| `--disable-chat-app` | flag | - | no | Scaffold the API-only backend, without the browser chat app. |
| `--enable-chat-app` | flag | - | no | Deprecated: the chat app is included by default; this flag is a no-op. |
| `--memory-store <MEMORY_STORE>` | string | - | no | Name for the declared memory store (default: derived from the directory, <dir>-memory). Only --server mason declares stores by default. |
| `--session-store <SESSION_STORE>` | string | - | no | Name for the declared session store (default: derived from the directory, <dir>-session). |
| `--existing` | flag | - | no | Prepare a coding-agent migration bundle for an existing LangGraph or OpenAI Agents SDK project (defaults to `.`). Requires `--server mason`. |

### `mason dev`

Run your agent locally so you can try it before deploying.

Starts the agent on a local server - by default http://localhost:8000 - and prints where to reach it: the chat UI if the project has one, otherwise a sample request against the agent's API.

Auth uses your Databricks profile (`-p` / `mason login`), and the agent reaches Databricks model serving through the AI Gateway on that profile - so there are no model keys to set up.

Under the hood this wraps `databricks apps run-local`: it reads the command + env from `app.yaml` and runs the app the way the Apps runtime would, so local behavior matches a deployment. The environment is built on the first run and reused after; pass `--prepare-environment` to force a rebuild (e.g. after changing dependencies).

Everything runs locally: `mason dev` is a local deployment that does not depend on a Databricks workspace for its resources. Tracing goes to a local MLflow tracking server (sqlite-backed, under `.mason/`) so traces are recorded on your machine with no workspace experiment or setup - open the printed Traces URL to view them (`mason tracing unbind` doesn't affect dev; it only stops the deployed agent's tracing). Long-term memory is off and conversation history is in-process (not durable): the memory/session stores bound with `mason memory/sessions bind` are created and used only when you `mason deploy`, not here. So there's nothing to provision and no service-principal grant to make; that all happens at `mason deploy` time.

```
mason dev [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Local source directory to run (containing app.yaml). Defaults to the current directory. |
| `--prepare-environment`, `--no-prepare-environment` | flag | - | no | Build the app's environment with uv before running. Default: build only if no .venv exists yet, and reuse it otherwise. Requires uv. |
| `--app-port <APP_PORT>` | integer | - | no | Port to run the app on (default 8000). |

### `mason memory`

Manage an agent's long-term memory: memory stores and their entries.

Memory is what an agent remembers across separate conversations - durable facts and preferences (for example "prefers concise answers", or a saved profile detail), as opposed to the turn-by-turn history of a single conversation (that is `mason sessions`).

A memory store is the managed store that holds this memory; each entry is a small document (a path plus its content) partitioned by actor, so one store keeps every user's memories separate.

| Subcommand | Description |
| --- | --- |
| [`memory stores`](#mason-memory-stores) | Workspace-scoped managed memory stores. |
| [`memory entries`](#mason-memory-entries) | Memory entries within a store, partitioned by actor. |
| [`memory bind`](#mason-memory-bind) | Bind memory STORE to the agent by declaring it in agent.toml. |
| [`memory unbind`](#mason-memory-unbind) | Remove the memory store binding from the agent's agent.toml. |

#### `mason memory stores`

Workspace-scoped managed memory stores.

| Subcommand | Description |
| --- | --- |
| [`memory stores create`](#mason-memory-stores-create) | Create a memory store. |
| [`memory stores list`](#mason-memory-stores-list) | List memory stores in the workspace (25 per page; paginates interactively on a terminal). |
| [`memory stores get`](#mason-memory-stores-get) | Get a memory store by id or resource name. |
| [`memory stores update`](#mason-memory-stores-update) | Update a store's display name and/or description. |
| [`memory stores delete`](#mason-memory-stores-delete) | Delete (soft-delete) a memory store. |

##### `mason memory stores create`

Create a memory store.

```
mason memory stores create [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--display-name <DISPLAY_NAME>` (`--name`) | string | - | yes | Workspace-unique display name (--name is accepted as an alias). |
| `--description <DESCRIPTION>` | string | - | no | Optional human-readable description. |

##### `mason memory stores list`

List memory stores in the workspace (25 per page; paginates interactively on a terminal).

```
mason memory stores list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--page-size <PAGE_SIZE>` | integer | `25` | no | - |
| `--page-token <PAGE_TOKEN>` | string | - | no | - |

##### `mason memory stores get`

Get a memory store by id or resource name.

```
mason memory stores get NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

##### `mason memory stores update`

Update a store's display name and/or description.

```
mason memory stores update NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--display-name <DISPLAY_NAME>` | string | - | no | - |
| `--description <DESCRIPTION>` | string | - | no | - |

##### `mason memory stores delete`

Delete (soft-delete) a memory store.

```
mason memory stores delete NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

#### `mason memory entries`

Memory entries within a store, partitioned by actor.

| Subcommand | Description |
| --- | --- |
| [`memory entries create`](#mason-memory-entries-create) | Create a memory entry. |
| [`memory entries get`](#mason-memory-entries-get) | Get an entry by id or resource name (includes content). |
| [`memory entries list`](#mason-memory-entries-list) | List entries for an actor. |
| [`memory entries search`](#mason-memory-entries-search) | Full-text search an actor's entries, ranked (includes content). |
| [`memory entries update`](#mason-memory-entries-update) | Update an entry's content and/or description. |
| [`memory entries delete`](#mason-memory-entries-delete) | Delete a memory entry. |

##### `mason memory entries create`

Create a memory entry.

```
mason memory entries create [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | Store id or resource name. |
| `--actor-id <ACTOR_ID>` | string | - | yes | Actor (partition) this entry belongs to. |
| `--path <PATH>` | string | - | yes | Absolute path, e.g. /preferences/style.md. |
| `--content <CONTENT>` | string | - | no | Entry content (inline). Use --content-file for large content. |
| `--content-file <CONTENT_FILE>` | path | - | no | Read entry content from a file (avoids shell arg-length limits on large content). |
| `--description <DESCRIPTION>` | string | - | no | Optional human-readable description. |
| `--session-id <SESSION_ID>` | string | - | no | Optional session id to associate the entry with. |
| `--source-type <SOURCE_TYPE>` | string | - | no | Origin of the entry: 'agent' or 'unspecified'. |

##### `mason memory entries get`

Get an entry by id or resource name (includes content).

```
mason memory entries get ENTRY [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `ENTRY` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | no | Store id/name (optional if ENTRY is a full resource name). |

##### `mason memory entries list`

List entries for an actor. The text view omits content; `-o json` includes it.

```
mason memory entries list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--actor-id <ACTOR_ID>` | string | - | yes | Required partition key. |
| `--path-prefix <PATH_PREFIX>` | string | - | no | - |
| `--session-id <SESSION_ID>` | string | - | no | - |
| `--page-size <PAGE_SIZE>` | integer | - | no | - |
| `--page-token <PAGE_TOKEN>` | string | - | no | - |

##### `mason memory entries search`

Full-text search an actor's entries, ranked (includes content).

```
mason memory entries search [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--actor-id <ACTOR_ID>` | string | - | yes | - |
| `--query <QUERY>` | string | - | yes | - |
| `--page-size <PAGE_SIZE>` | integer | - | no | - |

##### `mason memory entries update`

Update an entry's content and/or description.

```
mason memory entries update ENTRY [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `ENTRY` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | no | Store id/name (optional if ENTRY is a full resource name). |
| `--content <CONTENT>` | string | - | no | New entry content. |
| `--description <DESCRIPTION>` | string | - | no | New description. |

##### `mason memory entries delete`

Delete a memory entry.

```
mason memory entries delete ENTRY [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `ENTRY` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | no | Store id/name (optional if ENTRY is a full resource name). |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

#### `mason memory bind`

Bind memory STORE to the agent by declaring it in agent.toml.

This only edits agent.toml - it does not create the store. `mason deploy` creates any declared store that doesn't exist yet and grants the deployed app's service principal access to it.

```
mason memory bind STORE [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |

#### `mason memory unbind`

Remove the memory store binding from the agent's agent.toml.

Only edits agent.toml; the managed store itself is untouched (delete it with `mason memory stores delete`).

```
mason memory unbind [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |

### `mason mcp`

Discover managed MCP Services available through Unity Catalog.

| Subcommand | Description |
| --- | --- |
| [`mcp list`](#mason-mcp-list) | List MCP Services that can be added with ``mason tools add mcp``. |

#### `mason mcp list`

List MCP Services that can be added with ``mason tools add mcp``.

```
mason mcp list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--schema <SCHEMA>` | string | `system.ai` | no | Two-part Unity Catalog schema containing MCP Services. |

### `mason sessions`

Manage an agent's conversations: session stores, the sessions in them, and their items.

A session is a single conversation between an actor (a user) and the agent. It holds that conversation's ordered transcript of items - the messages, tool calls, and results that make up its running state. A session store is the managed store that holds an agent's sessions and their items, giving it durable conversation history it can list, resume, fork, or delete.

This is the short-term, per-conversation counterpart to the cross-conversation memory in `mason memory`.

| Subcommand | Description |
| --- | --- |
| [`sessions stores`](#mason-sessions-stores) | Workspace-scoped session stores. |
| [`sessions items`](#mason-sessions-items) | Transcript items within a session. |
| [`sessions bind`](#mason-sessions-bind) | Bind session STORE to the agent by declaring it in agent.toml. |
| [`sessions unbind`](#mason-sessions-unbind) | Remove the session store binding from the agent's agent.toml. |
| [`sessions create`](#mason-sessions-create) | Create a session in a store. |
| [`sessions list`](#mason-sessions-list) | List sessions in a store. |
| [`sessions get`](#mason-sessions-get) | Get a session by id. |
| [`sessions update`](#mason-sessions-update) | Update a session's metadata. |
| [`sessions delete`](#mason-sessions-delete) | Delete a session. |
| [`sessions fork`](#mason-sessions-fork) | Fork a session into a new independent top-level session. |

#### `mason sessions stores`

Workspace-scoped session stores.

| Subcommand | Description |
| --- | --- |
| [`sessions stores create`](#mason-sessions-stores-create) | Create a session store. |
| [`sessions stores list`](#mason-sessions-stores-list) | List session stores in the workspace (25 per page; paginates interactively on a terminal). |
| [`sessions stores get`](#mason-sessions-stores-get) | Get a session store by name. |
| [`sessions stores update`](#mason-sessions-stores-update) | Update a store's description and/or metadata. |
| [`sessions stores delete`](#mason-sessions-stores-delete) | Delete a session store. |

##### `mason sessions stores create`

Create a session store.

```
mason sessions stores create [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <NAME>` (`--display-name`) | string | - | yes | Workspace-unique store name, 3-63 chars (--display-name is accepted as an alias). |
| `--description <DESCRIPTION>` | string | - | no | - |
| `--metadata <METADATA>` | string | - | no | JSON object of string labels. |

##### `mason sessions stores list`

List session stores in the workspace (25 per page; paginates interactively on a terminal).

```
mason sessions stores list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--page-size <PAGE_SIZE>` | integer | `25` | no | - |
| `--page-token <PAGE_TOKEN>` | string | - | no | - |

##### `mason sessions stores get`

Get a session store by name.

```
mason sessions stores get NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

##### `mason sessions stores update`

Update a store's description and/or metadata.

```
mason sessions stores update NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--description <DESCRIPTION>` | string | - | no | - |
| `--metadata <METADATA>` | string | - | no | JSON object of string labels. |

##### `mason sessions stores delete`

Delete a session store.

```
mason sessions stores delete NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

#### `mason sessions items`

Transcript items within a session.

| Subcommand | Description |
| --- | --- |
| [`sessions items list`](#mason-sessions-items-list) | List transcript items in a session. |
| [`sessions items append`](#mason-sessions-items-append) | Append one or more items to a session (atomic, in order). |
| [`sessions items pop`](#mason-sessions-items-pop) | Remove and return the most recent item. |
| [`sessions items clear`](#mason-sessions-items-clear) | Remove all items from a session. |

##### `mason sessions items list`

List transcript items in a session.

```
mason sessions items list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--session-id <SESSION_ID>` | string | - | yes | - |
| `--order-by <ORDER_BY>` | string | - | no | 'create_time asc' or 'create_time desc'. |
| `--page-size <PAGE_SIZE>` | integer | - | no | - |
| `--page-token <PAGE_TOKEN>` | string | - | no | - |

##### `mason sessions items append`

Append one or more items to a session (atomic, in order).

```
mason sessions items append [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--session-id <SESSION_ID>` | string | - | yes | - |
| `--data <DATA>` | string | - | no | One item's JSON data (repeatable). |
| `--file <FILE>` | path | - | no | JSON array of item data values. |

##### `mason sessions items pop`

Remove and return the most recent item.

```
mason sessions items pop [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--session-id <SESSION_ID>` | string | - | yes | - |

##### `mason sessions items clear`

Remove all items from a session.

```
mason sessions items clear [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--session-id <SESSION_ID>` | string | - | yes | - |

#### `mason sessions bind`

Bind session STORE to the agent by declaring it in agent.toml.

This only edits agent.toml - it does not create the store. `mason deploy` creates any declared store that doesn't exist yet and grants the deployed app's service principal access to it.

```
mason sessions bind STORE [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |

#### `mason sessions unbind`

Remove the session store binding from the agent's agent.toml.

Only edits agent.toml; the managed store itself is untouched (delete it with `mason sessions stores delete`).

```
mason sessions unbind [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |

#### `mason sessions create`

Create a session in a store.

```
mason sessions create [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--actor-id <ACTOR_ID>` | string | - | yes | Application actor id (child must match parent). |
| `--session-id <SESSION_ID>` | string | - | no | Optional caller-chosen id. |
| `--parent-session-id <PARENT_SESSION_ID>` | string | - | no | - |
| `--metadata <METADATA>` | string | - | no | JSON object of string labels. |

#### `mason sessions list`

List sessions in a store.

```
mason sessions list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--filter <FILTER>` | string | - | no | e.g. actor_id = "support-123". |
| `--order-by <ORDER_BY>` | string | - | no | e.g. 'last_activity_time desc'. |
| `--page-size <PAGE_SIZE>` | integer | - | no | - |
| `--page-token <PAGE_TOKEN>` | string | - | no | - |

#### `mason sessions get`

Get a session by id.

```
mason sessions get SESSION_ID [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `SESSION_ID` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | no | Session store name (required in this preview). |

#### `mason sessions update`

Update a session's metadata.

```
mason sessions update SESSION_ID [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `SESSION_ID` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--metadata <METADATA>` | string | - | yes | JSON object of string labels (only mutable field). |

#### `mason sessions delete`

Delete a session.

```
mason sessions delete SESSION_ID [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `SESSION_ID` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--force` | flag | - | no | Cascade-delete descendant sessions. |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

#### `mason sessions fork`

Fork a session into a new independent top-level session.

```
mason sessions fork [SOURCE_SESSION_ID] [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `SOURCE_SESSION_ID` | no | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--source-session-id <SOURCE_SESSION_ID>` | string | - | no | Source session to fork (or pass it as the positional argument). |
| `--actor-id <ACTOR_ID>` | string | - | yes | - |
| `--up-to-item-id <UP_TO_ITEM_ID>` | string | - | no | Copy through this item id inclusively. |
| `--session-id <SESSION_ID>` | string | - | no | Optional id for the fork. |
| `--metadata <METADATA>` | string | - | no | - |

### `mason tracing`

Configure MLflow tracing for your deployed agents, and inspect the traces.

| Subcommand | Description |
| --- | --- |
| [`tracing bind`](#mason-tracing-bind) | Bind tracing to an experiment, by name or id (one required). |
| [`tracing unbind`](#mason-tracing-unbind) | Unbind tracing (remove the binding), turning tracing off for the deployed agent (deploy-only; `mason dev` still traces locally). |
| [`tracing list`](#mason-tracing-list) | List recent agent traces in an experiment. |
| [`tracing get`](#mason-tracing-get) | Get a single trace by id (status, latency, span count, previews). |

#### `mason tracing bind`

Bind tracing to an experiment, by name or id. Requires one of them (like `mason memory/sessions bind`); the binding's presence is what turns tracing on.

The experiment is stored as a NAME, not an id, so the binding stays valid across workspaces/profiles - mason get-or-creates it in the active workspace at deploy. ``--experiment-id`` (e.g. from the experiment's URL) is a convenience: it's resolved to the experiment's name and stored as a name, never as an id.

```
mason tracing bind [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--experiment-name <EXPERIMENT_NAME>` | string | - | no | MLflow experiment name to trace to - an absolute workspace path, e.g. /Shared/mason_traces/&lt;agent&gt; or /Users/&lt;you&gt;/mason_traces/&lt;agent&gt;. mason get-or-creates it at deploy. Mutually exclusive with --experiment-id. |
| `--experiment-id <EXPERIMENT_ID>` | string | - | no | MLflow experiment id (e.g. copied from the experiment's workspace URL) to trace to. Resolved to the experiment's name and stored as a name - mason persists names, not ids, so the binding stays valid across workspaces. Mutually exclusive with --experiment-name. |
| `--source <SOURCE>` | path | `.` | no | Project directory containing agent.toml. Defaults to the current directory. |

#### `mason tracing unbind`

Unbind tracing: remove the experiment binding from agent.toml, turning tracing off for the DEPLOYED agent (`mason deploy` then wires no MLflow env).

Deploy-only: `mason dev` still traces locally to its own MLflow server, so you keep local traces while the deployed agent stays untraced.

```
mason tracing unbind [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Project directory containing agent.toml. Defaults to the current directory. |

#### `mason tracing list`

List recent agent traces in an experiment.

An explicit ``--experiment-name`` / ``--experiment-id`` reads that workspace experiment and must name one that exists (errors otherwise, so a typo isn't mistaken for an empty experiment). With neither, this project's experiment is read: the workspace one if it's been provisioned (by `mason deploy`), otherwise the local `mason dev` store (`.mason/mlflow.db`), so a not-yet-deployed dev run's traces still show up here (tagged "(local dev)"). Nothing traced anywhere yet lists nothing. A UC-backed experiment is read through a SQL warehouse (``--warehouse``).

```
mason tracing list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--experiment-name <EXPERIMENT_NAME>` | string | - | no | MLflow experiment name to read (an absolute workspace path). Default: this project's experiment. |
| `--experiment-id <EXPERIMENT_ID>` | string | - | no | MLflow experiment id to read (e.g. from the experiment URL). Mutually exclusive with --experiment-name. |
| `--warehouse <WAREHOUSE_ID>` | string | - | no | SQL warehouse id used to read traces from a UC-backed experiment (required for UC experiments; ignored for managed). Falls back to the MLFLOW_TRACING_SQL_WAREHOUSE_ID env var. |
| `--limit <LIMIT>` | integer | `20` | no | - |
| `--source <SOURCE>` | path | `.` | no | Project directory to resolve the default experiment from (default: current dir). |

#### `mason tracing get`

Get a single trace by id (status, latency, span count, previews).

Reads from the same place as `mason tracing list`: an explicit ``--experiment-name`` / ``--experiment-id`` targets that workspace store and must name one that exists (errors otherwise); otherwise this project's workspace experiment if provisioned, else its local `mason dev` store. A UC-backed experiment is read through a SQL warehouse (``--warehouse``).

```
mason tracing get TRACE_ID [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `TRACE_ID` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--experiment-name <EXPERIMENT_NAME>` | string | - | no | MLflow experiment name to read (an absolute workspace path). Default: this project's experiment. |
| `--experiment-id <EXPERIMENT_ID>` | string | - | no | MLflow experiment id to read (e.g. from the experiment URL). Mutually exclusive with --experiment-name. |
| `--warehouse <WAREHOUSE_ID>` | string | - | no | SQL warehouse id used to read traces from a UC-backed experiment (required for UC experiments; ignored for managed). Falls back to the MLFLOW_TRACING_SQL_WAREHOUSE_ID env var. |
| `--source <SOURCE>` | path | `.` | no | Project directory to resolve the experiment from (default: current dir). |

### `mason deploy`

Deploy your agent to Databricks Apps and get back a hosted URL to try it.

Rolls the agent out to Databricks Apps and prints the URL where you (or anyone you share it with) can use it. The deployed agent reaches Databricks model serving through the AI Gateway using the app's own identity - no model keys to configure - and `deploy` also reconciles the stores declared in agent.toml and wires in any tracing.

NAME is recorded in agent.toml on the first deploy, so a later `mason deploy` from the project directory can omit it (passing NAME again updates the recorded name). The deployed app is named `agent-mason-<name>` (Mason adds the prefix if absent); use that full name with the `mason deployments` commands. `deployments list` shows only apps carrying this prefix.

Any memory/session store declared in agent.toml (for example, by `mason memory/sessions bind`) is created if it doesn't exist yet; agent.toml itself is never modified for stores.

Scaling to multiple instances (--instances) uses best-effort sticky routing, so a browser session automatically stays on one instance.

API clients that need it must resend a stable UUID in this cookie every request: __Host-databricks-app-router=<uuid>

```
mason deploy [NAME] [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | no | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Local source directory for the deployment (containing app.yaml). Defaults to the current directory. |
| `--pip-index-url <PIP_INDEX_URL>` | string | `https://pypi.org/simple/` | no | Base URL of the Python Package Index. Defaults to public PyPI. |
| `--workspace-path <WORKSPACE_PATH>` | string | - | no | Workspace destination for the synced source (defaults to a per-user path). |
| `--instances <INSTANCES>` | integer range | - | no | Number of deployment instances. |
| `--allow-user-scope-update` | flag | - | no | Allow Mason to add missing user API scopes to an existing App for tools configured with `auth = 'user'`. Once added, later deploys do not need this flag. |

### `mason deployments`

Inspect and manage deployed agents: list, get, stream logs, start, stop, or delete.

| Subcommand | Description |
| --- | --- |
| [`deployments list`](#mason-deployments-list) | List Mason agent deployments (apps named `agent-mason-*`) in the workspace. |
| [`deployments get`](#mason-deployments-get) | Get an agent deployment's details. |
| [`deployments logs`](#mason-deployments-logs) | Stream a deployment's logs. |
| [`deployments start`](#mason-deployments-start) | Start a deployment. |
| [`deployments stop`](#mason-deployments-stop) | Stop a deployment. |
| [`deployments delete`](#mason-deployments-delete) | Delete a deployment. |

#### `mason deployments list`

List Mason agent deployments (apps named `agent-mason-*`) in the workspace.

```
mason deployments list
```

#### `mason deployments get`

Get an agent deployment's details.

```
mason deployments get NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

#### `mason deployments logs`

Stream a deployment's logs.

```
mason deployments logs NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

#### `mason deployments start`

Start a deployment.

```
mason deployments start NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

#### `mason deployments stop`

Stop a deployment.

```
mason deployments stop NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

#### `mason deployments delete`

Delete a deployment.

```
mason deployments delete NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

### `mason endpoint`

Invoke arbitrary HTTP endpoints.

| Subcommand | Description |
| --- | --- |
| [`endpoint invoke`](#mason-endpoint-invoke) | Send one HTTP request to a Databricks App or arbitrary URL. |

#### `mason endpoint invoke`

Send one HTTP request to a Databricks App or arbitrary URL.

```
mason endpoint invoke [APP] [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `APP` | no | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--url <URL>` | string | - | no | Base URL for localhost or an arbitrary HTTP server. |
| `--method <METHOD>` | string | `POST` | no | - |
| `--path <PATH>` | string | - | yes | Request path, such as /api/invocations. |
| `--query <QUERY>` | string | - | no | Query parameter as 'name=value'. |
| `--json <JSON_VALUE>` | string | - | no | Complete JSON request body. |
| `--sse` | flag | - | no | Consume the response as Server-Sent Events. |
| `--session-id <SESSION_ID>` | string | - | no | Application session id (default: generated for a Databricks App). |
| `--timeout <TIMEOUT>` | float range | `300.0` | no | - |
| `--auth`, `--no-auth` | flag | - | no | Inject Databricks OAuth authentication. |

### `mason tools`

Discover available integrations and manage an agent's tool bindings.

Tools are what let an agent act beyond the language model itself - query governed data, call a service, or run a function - and each one is recorded in agent.toml so `mason dev` / `mason deploy` wire it in. `mason tools add` manages these Databricks-managed tool types:

sandbox Query Unity Catalog data via system.ai.sandbox, scoped to the tables, volumes, or paths you choose. mcp A Databricks-managed MCP service (see `mason tools list --kind mcp`), e.g. system.ai.web_search. uc-function An existing Unity Catalog function (catalog.schema.function). genie-one Workspace-wide Genie One MCP tools. genie-agent Native Genie conversation tools for a configured space ID.

Browse available integrations with `mason tools list`, add one with `mason tools add <type>`, and drop a binding with `mason tools remove`. Review agent.toml for configured managed tools and MCP bindings. The list shows addable integrations, not configured bindings or individual operations inside an MCP service. Custom Python tools are code-first - write them directly in your project's code rather than through the CLI.

| Subcommand | Description |
| --- | --- |
| [`tools add`](#mason-tools-add) | Add a managed sandbox, MCP service, UC function, or Genie tool binding. |
| [`tools list`](#mason-tools-list) | List available integrations to add, not configured agent bindings. |
| [`tools remove`](#mason-tools-remove) | Remove a managed tool binding from this agent. |

#### `mason tools add`

Add a managed sandbox, MCP service, UC function, or Genie tool binding.

Subcommands target the current directory by default.

Pass --source PATH to target another project.

Review that project's agent.toml to check configured managed tools and MCP bindings.

| Subcommand | Description |
| --- | --- |
| [`tools add sandbox`](#mason-tools-add-sandbox) | Add a data sandbox tool (system.ai.sandbox), scoped to specific Unity Catalog resources. |
| [`tools add mcp`](#mason-tools-add-mcp) | Validate and add a Databricks-managed MCP service as a tool. |
| [`tools add uc-function`](#mason-tools-add-uc-function) | Add an existing Unity Catalog function (catalog.schema.function) as a tool. |
| [`tools add genie-one`](#mason-tools-add-genie-one) | Add workspace-wide Genie One MCP tools. |
| [`tools add genie-agent`](#mason-tools-add-genie-agent) | Add native Genie conversation tools for a 32-character lowercase hexadecimal SPACE_ID. |

##### `mason tools add sandbox`

Add a data sandbox tool (system.ai.sandbox), scoped to specific Unity Catalog resources.

Review the target project's agent.toml to check configured managed tools and MCP bindings.

```
mason tools add sandbox [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--scope <SCOPES>` | string | - | yes | Allowed table:, volume:, or workspace: resource. Repeat for multiple scopes. |
| `--permission <read_only|read_write>` | `read_only` \| `read_write` | `read_only` | no | - |
| `--name <TOOL_ID>` | string | `sandbox` | no | - |
| `--auth <user|app>` | `user` \| `app` | `user` | no | - |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |

##### `mason tools add mcp`

Validate and add a Databricks-managed MCP service as a tool.

Use `mason tools list --kind mcp` for available services. Review the target project's agent.toml to check configured managed tools and MCP bindings.

```
mason tools add mcp SERVICE [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `SERVICE` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <TOOL_ID>` | string | - | no | - |
| `--auth <user|app>` | `user` \| `app` | `user` | no | - |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |

##### `mason tools add uc-function`

Add an existing Unity Catalog function (catalog.schema.function) as a tool.

```
mason tools add uc-function FUNCTION_NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `FUNCTION_NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <TOOL_ID>` | string | - | no | - |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |

##### `mason tools add genie-one`

Add workspace-wide Genie One MCP tools.

```
mason tools add genie-one [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <TOOL_ID>` | string | `genie_one` | no | - |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |

##### `mason tools add genie-agent`

Add native Genie conversation tools for a 32-character lowercase hexadecimal SPACE_ID.

```
mason tools add genie-agent SPACE_ID [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `SPACE_ID` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <TOOL_ID>` | string | `genie_agent` | no | - |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |

#### `mason tools list`

List available integrations to add, not configured agent bindings.

By default, show built-in add recipes plus caller-visible MCP Services in system.ai. `--kind mcp` limits discovery to MCP Services; `--schema catalog.schema` replaces system.ai. Sandbox recipes require scopes; UC-function and Genie Agent recipes require concrete resource identifiers. Genie One needs no additional argument.

No agent project is required. MCP discovery uses your Databricks profile; local recipes do not authenticate. This does not scan every workspace schema or list individual MCP operations. API failures return a nonzero exit status and mark discovery incomplete, not empty.

Review agent.toml to check configured managed tools and MCP bindings. The former configured list and `--source` option are removed. JSON discovery uses schema_version 2 and available_tools.

```
mason tools list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--kind <sandbox|mcp|uc-function|genie-one|genie-agent>` | choice | - | no | Show one integration kind. Sandbox, uc-function, genie-one, and genie-agent show local add recipes only. |
| `--schema <SCHEMA>` | string | - | no | Two-part UC schema: catalog.schema (default: system.ai). Requires --kind mcp. |

#### `mason tools remove`

Remove a managed tool binding from this agent.

```
mason tools remove TOOL_ID [MCP_SERVICE] [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `TOOL_ID` | yes | - |
| `MCP_SERVICE` | no | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Mason agent project containing agent.toml. |
