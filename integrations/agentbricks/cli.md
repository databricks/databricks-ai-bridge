# Agent Bricks CLI command reference

`agentbricks` is the Agent Bricks command-line interface for building and deploying custom AI agents on Databricks. It
scaffolds an agent project from a template, runs it locally with a chat UI, deploys it to Databricks
Apps, and manages the tools, memory, sessions, and tracing behind it - all from one authenticated
command.

This page is the full command reference: every command, subcommand, argument, and option. For
concepts, guides, and the Python SDK, see the [README](README.md). Every command also has built-in
help - append `--help` (or `-h`) at any level, for example `agentbricks deploy --help` or
`agentbricks sessions items append --help`.

> **Preview:** Agent Bricks CLI is experimental. The CLI, its commands, and the underlying agent APIs are all in
> preview, may need to be enabled for your workspace, and are likely to change in
> backward-incompatible ways.

## Installation

```sh
pip install databricks-agentbricks
```

The `databricks-agentbricks` distribution provides the `agentbricks` command and AgentKit SDK.

See [Installation](README.md#installation) for installation details and shell completion.

## Authentication

Agent Bricks CLI authenticates with a [Databricks configuration profile](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
Run `agentbricks login` once to save a default profile, or pass `--profile` / `-p` on any command. Without
a profile, the Databricks SDK's default authentication resolution is used. See
[Authentication](README.md#authentication) for details.

## Global options

These options apply to every command. Pass them before the command name, for example
`agentbricks -p my-profile -o json sessions stores list`.

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| `--profile <PROFILE>` (`-p`) | string | - | Name of a saved Databricks workspace sign-in. Run `databricks auth profiles` to list yours. |
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
| [`login`](#agentbricks-login) | Authenticate and save a default profile |
| [`logout`](#agentbricks-logout) | Forget the saved default profile |
| [`init`](#agentbricks-init) | Scaffold a new agent project |
| [`doctor`](#agentbricks-doctor) | Check an existing agent's Agent Bricks onboarding |
| [`dev`](#agentbricks-dev) | Run the agent locally with a chat UI |
| [`memory`](#agentbricks-memory) | Manage an agent's long-term memory |
| [`mcp`](#agentbricks-mcp) | Discover managed MCP services |
| [`sessions`](#agentbricks-sessions) | Manage an agent's conversation sessions |
| [`tracing`](#agentbricks-tracing) | Set up and inspect agent tracing |
| [`deploy`](#agentbricks-deploy) | Deploy an agent to Databricks Apps |
| [`deployments`](#agentbricks-deployments) | Manage deployed agents |
| [`endpoint`](#agentbricks-endpoint) | Invoke arbitrary HTTP endpoints. |
| [`tools`](#agentbricks-tools) | Manage an agent's tools |
| [`experimental`](#agentbricks-experimental) | Commands whose interface is still being designed: `experimental models`, `experimental prompts` |

## Commands

### `agentbricks login`

Save a Databricks workspace sign-in as your default, so later commands can omit -p.

Create the sign-in first with `databricks auth login --profile <name>`.

```
agentbricks login [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--profile <PROFILE>` (`-p`) | string | - | no | Name of a saved Databricks workspace sign-in to use by default. Run `databricks auth profiles` to list yours. |

### `agentbricks logout`

Forget the saved profile selection without deleting its credentials.

```
agentbricks logout
```

### `agentbricks init`

Scaffold a local agent project from an Agent Bricks CLI template.

DIRECTORY is the target path to create (defaults to the template's own name). The directory must not already exist. init records a deployment name that fits the Databricks Apps limit (--name, or the directory name shortened to 17 characters), so `agentbricks deploy` from the project directory needs no NAME.

Databricks App names cap at 30 characters and deploy adds the `agent-bricks-` prefix, so a deployment name can be up to 17 lowercase letters, digits, and hyphens. init records `[agent].deployment_name` in `agent.toml` and shows it as `Deploy name`. A directory name that is valid and fits is recorded unchanged. Otherwise init sanitizes it and, if it is still over 17 characters, keeps its first 10 characters plus `-<token>`, the same 6-letter token as the default stores (`caj-langgraph-agent` becomes `caj-langgr-<token>`). `--name` is validated before any file is written; an invalid name fails with a suggestion. JSON results include `deployment_name` and `deployment_name_truncated` (true when init changed the directory name to fit).

Pass --profile (or set a default via `agentbricks login` / -p) to seed a local `.env` so the scaffolded project runs with `agentbricks dev` right away.

The scaffold is preconfigured to call Databricks model serving through the AI Gateway using that profile, so it can talk to a model with no separate endpoint or API key to set up.

New projects include the `agent-bricks-workflow` skill in `.agents/skills/`, with discovery pointers in
`.claude/skills/` and `.agent/skills/`. Local `dev` is optional when the goal is deployment.
JSON results include `workflow_skill` and `skill_pointers`.

With `--existing`, prepare the separate `agent-bricks-migrate` bundle and pointers in all three
discovery roots without changing existing application source or configuration. Coding agents in
existing projects can read the bundled workflow skill at the path printed by the CLI's automatic
stderr hint, enabled by default when a coding-agent environment is detected. Ordinary human
invocations do not emit this hint. Set `AGENTBRICKS_DISABLE_AGENT_HINT=1` to disable hints without
disabling skill installation or discovery.

`--server agentbricks` selects the managed server, which supports foreground, streaming, and background invocations through one HTTP contract and Runtime Store. This existing server value is recorded in `agent.toml`. Pass `--server custom` for a minimal foreground-only FastAPI server.

```
agentbricks init [DIRECTORY] [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `DIRECTORY` | no | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--framework <langgraph|openai>` | `langgraph` \| `openai` | `langgraph` | no | Agent framework: langgraph (LangGraph, default) or openai (OpenAI Agents SDK). |
| `--server <agentbricks|custom>` | `agentbricks` \| `custom` | `agentbricks` | no | Use the managed invocation server (`agentbricks` is the existing `agent.toml` value) or a minimal custom FastAPI server. |
| `--profile <PROFILE>` | string | - | no | Seed a local .env with this DATABRICKS_CONFIG_PROFILE so `agentbricks dev` works immediately (defaults to the profile from -p / `agentbricks login`). |
| `--disable-chat-app` | flag | - | no | Scaffold the API-only backend, without the browser chat app. |
| `--enable-chat-app` | flag | - | no | Deprecated: the chat app is included by default; this flag is a no-op. |
| `--memory-store <MEMORY_STORE>` | string | - | no | Name for the declared memory store (default: a unique name derived from --name or the directory, <name>-<token>-memory). Only --server agentbricks declares stores by default. |
| `--session-store <SESSION_STORE>` | string | - | no | Name for the declared session store (default: a unique name derived from --name or the directory, <name>-<token>-sessions). |
| `--name <NAME>` | string | - | no | Deployment name, recorded in agent.toml as [agent].deployment_name. Deploys as agent-bricks-<name>, so it can be up to 17 lowercase letters, digits, and hyphens. Also names the default stores and tracing experiment (default: derived from the directory, shortened to fit). |
| `--existing` | flag | - | no | Prepare a coding-agent migration bundle for an existing LangGraph or OpenAI Agents SDK project (defaults to `.`). Requires `--server agentbricks`. |

### `agentbricks doctor`

Check whether an existing agent repository is onboarded to Agent Bricks. DIRECTORY defaults to the
current directory.

Doctor is read-only and offline: it does not import application source, contact Databricks, or
change files. It checks `agent.toml`, `.agentbricks/project.toml`, the framework-specific
`databricks-agentbricks` dependency extra, `app.yaml`, and production Python source for a
`DurableAgentServer` instance with an `invoke` hook plus a recognized framework adapter call; test,
example, and old/stale directories are excluded. A bounded source scan that exceeds a limit is
reported while the evidence it already found still counts. These checks are static repository
evidence, not proof that the configured startup command executes the files found. The
`deployment_name` check confirms that `[agent].deployment_name` (or the directory name when none is
recorded) fits the Databricks Apps limit: up to 17 lowercase letters, digits, and hyphens after the
`agent-bricks-` prefix. On failure its detail gives `agentbricks deploy <name>` with a name that fits;
when it is the only failing check, that is also the report's `next_step`. Doctor exits 0
only when every check passes, and exits 1 after printing a normal report otherwise. If the framework
is unknown, the remediation requires an explicit `--framework <framework_name>`. Use global
`-o json` for a structured report.

```
agentbricks doctor [DIRECTORY]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `DIRECTORY` | no | Existing agent repository to inspect (default: `.`). |

### `agentbricks dev`

Run your agent locally so you can try it before deploying.

Starts the agent on a local server - by default http://localhost:8000 - and prints where to reach it: the chat UI if the project has one, otherwise a sample request against the agent's API.

Auth uses your Databricks profile (`-p` / `agentbricks login`), and the agent reaches Databricks model serving through the AI Gateway on that profile - so there are no model keys to set up.

Under the hood this wraps `databricks apps run-local`: it reads the command + env from `app.yaml` and runs the app the way the Apps runtime would, so local behavior matches a deployment. The environment is built on the first run and reused after; pass `--prepare-environment` to force a rebuild (e.g. after changing dependencies).

Everything runs locally: `agentbricks dev` is a local deployment that does not depend on a Databricks workspace for its resources. Tracing goes to a local MLflow tracking server (sqlite-backed, under the existing `.agentbricks/` state directory) so traces are recorded on your machine with no workspace experiment or setup - open the printed Traces URL to view them (`agentbricks tracing unbind` doesn't affect dev; it only stops the deployed agent's tracing). Long-term memory is off and conversation history is in-process (not durable): the memory/session stores bound with `agentbricks memory/sessions bind` are created and used only when you `agentbricks deploy`, not here. So there's nothing to provision and no service-principal grant to make; that all happens at `agentbricks deploy` time.

```
agentbricks dev [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Local source directory to run (containing app.yaml). Defaults to the current directory. |
| `--prepare-environment`, `--no-prepare-environment` | flag | - | no | Build the app's environment with uv before running. Default: build only if no .venv exists yet, and reuse it otherwise. Requires uv. |
| `--app-port <APP_PORT>` | integer | - | no | Port to run the agent on (default 8000). |

### `agentbricks memory`

Manage an agent's long-term memory: memory stores and their entries.

Memory is what an agent remembers across separate conversations - durable facts and preferences (for example "prefers concise answers", or a saved profile detail), as opposed to the turn-by-turn history of a single conversation (that is `agentbricks sessions`).

A memory store is the managed store that holds this memory; each entry is a small document (a path plus its content) partitioned by actor, so one store keeps every user's memories separate.

| Subcommand | Description |
| --- | --- |
| [`memory stores`](#agentbricks-memory-stores) | Workspace-scoped managed memory stores. |
| [`memory entries`](#agentbricks-memory-entries) | Memory entries within a store, partitioned by actor. |
| [`memory pipeline`](#agentbricks-memory-pipeline) | Distill session history into long-term memory. |
| [`memory bind`](#agentbricks-memory-bind) | Bind memory STORE to the agent by declaring it in agent.toml. |
| [`memory unbind`](#agentbricks-memory-unbind) | Remove the memory store binding from the agent's agent.toml. |

#### `agentbricks memory stores`

Workspace-scoped managed memory stores.

| Subcommand | Description |
| --- | --- |
| [`memory stores create`](#agentbricks-memory-stores-create) | Create a memory store. |
| [`memory stores list`](#agentbricks-memory-stores-list) | List memory stores in the workspace. |
| [`memory stores get`](#agentbricks-memory-stores-get) | Get a memory store by id, resource name (memory-stores/<id>), or display name. |
| [`memory stores update`](#agentbricks-memory-stores-update) | Update a store's display name and/or description. |
| [`memory stores delete`](#agentbricks-memory-stores-delete) | Delete (soft-delete) a memory store. |
| [`memory stores permissions`](#agentbricks-memory-stores-permissions) | Inspect and manage store access. |

##### `agentbricks memory stores create`

Create a memory store.

```
agentbricks memory stores create [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--display-name <DISPLAY_NAME>` (`--name`) | string | - | yes | Workspace-unique display name (--name is accepted as an alias). |
| `--description <DESCRIPTION>` | string | - | no | Optional human-readable description. |

##### `agentbricks memory stores list`

List memory stores in the workspace.

Without --name or --all, shows 25 per page: it paginates interactively on a terminal and otherwise
prints the --page-token for the next page. --name and --all page through every store and print one
complete list. The API has no server-side filter, so --name reads every page.

```
agentbricks memory stores list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <NAME>` | string | - | no | Only stores whose display name contains TEXT (case-insensitive). Reads every page. |
| `--all` | flag | - | no | Page through every store. |
| `--page-size <PAGE_SIZE>` | integer | `25` | no | Stores per page without --name or --all. |
| `--page-token <PAGE_TOKEN>` | string | - | no | Resume from a previous page's token (not with --name or --all). |

##### `agentbricks memory stores get`

Get a memory store by id, resource name (memory-stores/<id>), or display name.

A display name is resolved by listing every store; if several stores share it, the command fails and
lists their resource names.

```
agentbricks memory stores get NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

##### `agentbricks memory stores update`

Update a store's display name and/or description.

```
agentbricks memory stores update NAME [options]
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

##### `agentbricks memory stores delete`

Delete (soft-delete) a memory store.

```
agentbricks memory stores delete NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

##### `agentbricks memory stores permissions`

Inspect and manage store access.

Get returns your effective permission. List shows direct grants, excluding implicit
ownership and inherited access. Grant, List, and Revoke require the creator or a
workspace admin. STORE uses the same identifiers as the store commands.

| Subcommand | Description |
| --- | --- |
| [`memory stores permissions get`](#agentbricks-memory-stores-permissions-get) | Show the authenticated caller's effective permission on STORE. |
| [`memory stores permissions list`](#agentbricks-memory-stores-permissions-list) | List explicit grants on STORE as its creator or a workspace admin. |
| [`memory stores permissions grant`](#agentbricks-memory-stores-permissions-grant) | Grant access on STORE as its creator or a workspace admin. |
| [`memory stores permissions revoke`](#agentbricks-memory-stores-permissions-revoke) | Revoke both READ and WRITE on STORE as its creator or a workspace admin. |

###### `agentbricks memory stores permissions get`

Show the authenticated caller's effective permission on STORE.

Session stores accept a store name or resource name. Memory stores accept an ID,
resource name, or display name; ambiguous display names fail. No access is reported as
PERMISSION_LEVEL_UNSPECIFIED in JSON.

```
agentbricks memory stores permissions get STORE
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | Store identifier; see command description. |

###### `agentbricks memory stores permissions list`

List explicit grants on STORE as its creator or a workspace admin.

Accepts a session store name or memory store ID, or a resource name. Implicit ownership
and inherited access are excluded. Text output offers subsequent pages on a terminal;
JSON and piped output return one page unless --all is supplied.

```
agentbricks memory stores permissions list STORE [options]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | Store identifier; see command description. |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--page-size <PAGE_SIZE>` | integer [1–1000] | `25` | no | Grants per page. |
| `--page-token <PAGE_TOKEN>` | string | - | no | Resume from a previous page (not with --all). |
| `--all` | flag | - | no | Fetch every page of direct grants. |

###### `agentbricks memory stores permissions grant`

Grant access on STORE as its creator or a workspace admin.

Specify exactly one of --user or --service-principal.
Accepts a session store name or memory store ID, or a resource name.

```
agentbricks memory stores permissions grant STORE [options]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | Store identifier; see command description. |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--service-principal <SERVICE_PRINCIPAL>` | string | - | no | Service principal application/client ID, not a numeric SCIM ID (not with --user). |
| `--user <USER>` | string | - | no | User email (not with --service-principal). |
| `--permission <READ\|WRITE>` | `READ` \| `WRITE` | - | yes | Access to grant. WRITE includes READ; READ preserves an existing WRITE grant. |

###### `agentbricks memory stores permissions revoke`

Revoke both READ and WRITE on STORE as its creator or a workspace admin.

Specify exactly one of --user or --service-principal.
Accepts a session store name or memory store ID, or a resource name. Repeating a revoke
succeeds. Ownership cannot be revoked; inherited access is not removed.

```
agentbricks memory stores permissions revoke STORE [options]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | Store identifier; see command description. |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--service-principal <SERVICE_PRINCIPAL>` | string | - | no | Service principal application/client ID, not a numeric SCIM ID (not with --user). |
| `--user <USER>` | string | - | no | User email (not with --service-principal). |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

#### `agentbricks memory entries`

Memory entries within a store, partitioned by actor.

| Subcommand | Description |
| --- | --- |
| [`memory entries create`](#agentbricks-memory-entries-create) | Create a memory entry. |
| [`memory entries get`](#agentbricks-memory-entries-get) | Get an entry by id or resource name (includes content). |
| [`memory entries list`](#agentbricks-memory-entries-list) | List entries for an actor. |
| [`memory entries search`](#agentbricks-memory-entries-search) | Full-text search an actor's entries, ranked (includes content). |
| [`memory entries update`](#agentbricks-memory-entries-update) | Update an entry's content and/or description. |
| [`memory entries delete`](#agentbricks-memory-entries-delete) | Delete a memory entry. |

##### `agentbricks memory entries create`

Create a memory entry.

```
agentbricks memory entries create [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | Store id or resource name. |
| `--actor-id <ACTOR_ID>` | string | - | yes | Who the memory belongs to, usually a user. The deployed chat app uses the signed-in user's email. |
| `--path <PATH>` | string | - | yes | Key for the entry inside the store, written like a path (e.g. /preferences/style.md). Not a file on your machine. |
| `--content <CONTENT>` | string | - | no | Entry content (inline). Use --content-file for large content. |
| `--content-file <CONTENT_FILE>` | path | - | no | Read entry content from a file (avoids shell arg-length limits on large content). |
| `--description <DESCRIPTION>` | string | - | no | Optional human-readable description. |
| `--session-id <SESSION_ID>` | string | - | no | Optional session id to associate the entry with. |
| `--source-type <SOURCE_TYPE>` | string | - | no | Origin of the entry: 'agent' or 'unspecified'. |

##### `agentbricks memory entries get`

Get an entry by id or resource name (includes content).

```
agentbricks memory entries get ENTRY [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `ENTRY` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | no | Store id/name (optional if ENTRY is a full resource name). |

##### `agentbricks memory entries list`

List entries for an actor. The text view omits content; `-o json` includes it.

```
agentbricks memory entries list [options]
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

##### `agentbricks memory entries search`

Full-text search an actor's entries, ranked (includes content).

```
agentbricks memory entries search [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--actor-id <ACTOR_ID>` | string | - | yes | - |
| `--query <QUERY>` | string | - | yes | - |
| `--page-size <PAGE_SIZE>` | integer | - | no | - |

##### `agentbricks memory entries update`

Update an entry's content and/or description.

```
agentbricks memory entries update ENTRY [options]
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

##### `agentbricks memory entries delete`

Delete a memory entry.

```
agentbricks memory entries delete ENTRY [options]
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

#### `agentbricks memory bind`

Bind memory STORE to the agent by declaring it in agent.toml.

This only edits agent.toml - it does not create the store. `agentbricks deploy` creates any declared store that doesn't exist yet and grants the deployed app's service principal access to it.

```
agentbricks memory bind STORE [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |

#### `agentbricks memory unbind`

Remove the memory store binding from the agent's agent.toml.

This only edits agent.toml. The next `agentbricks deploy` removes the managed environment reference from app.yaml; previously granted managed-store access is not revoked.

```
agentbricks memory unbind [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |

#### `agentbricks memory pipeline`

Manage pipelines that distill session history into long-term memory.

| Subcommand | Description |
| --- | --- |
| [`memory pipeline create`](#agentbricks-memory-pipeline-create) | Create a Dreamer memory pipeline. |
| [`memory pipeline list`](#agentbricks-memory-pipeline-list) | List Dreamer memory pipelines in the workspace. |
| [`memory pipeline get`](#agentbricks-memory-pipeline-get) | Get a Dreamer memory pipeline by id or resource name. |
| [`memory pipeline update`](#agentbricks-memory-pipeline-update) | Update a pipeline's display name, instructions, model, or trigger. |
| [`memory pipeline delete`](#agentbricks-memory-pipeline-delete) | Delete a Dreamer memory pipeline and its backing job. |
| [`memory pipeline run`](#agentbricks-memory-pipeline-run) | Manually run a Dreamer memory pipeline. |
| [`memory pipeline pause`](#agentbricks-memory-pipeline-pause) | Pause scheduled runs; active runs continue and manual runs remain available. |
| [`memory pipeline resume`](#agentbricks-memory-pipeline-resume) | Enable scheduled runs, automatically about every 24 hours. |
| [`memory pipeline runs`](#agentbricks-memory-pipeline-runs) | Inspect memory pipeline runs. |

##### `agentbricks memory pipeline create`

```text
agentbricks memory pipeline create --memory-store TEXT --session-store TEXT [options]
```

| Option | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--memory-store` | string | - | yes | Memory store name or resource name. |
| `--session-store` | string | - | yes | Session store name or resource name. |
| `--model` | string | - | no | Model service used for Dreamer distillation. |
| `--display-name` | string | - | no | Optional human-readable pipeline name. |
| `--instructions` | string | - | no | Instructions steering distillation: inline text or @path to a UTF-8 file. |
| `--trigger <manual\|scheduled>` | `manual` \| `scheduled` | `manual` | no | How runs start: only when you start one, or also automatically about every 24 hours. |

```bash
agentbricks memory pipeline create --memory-store agent-memory --session-store agent-sessions \
  --model system.ai.gpt-5-6-sol
```

Create a pipeline that also runs automatically about every 24 hours:

```bash
agentbricks memory pipeline create --memory-store agent-memory --session-store agent-sessions \
  --trigger scheduled
```

##### `agentbricks memory pipeline list`

```text
agentbricks memory pipeline list [options]
```

| Option | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--page-size` | integer | 25 | no | Maximum number of pipelines to return. |
| `--page-token` | string | - | no | Token from a previous page. |

##### `agentbricks memory pipeline get`

```text
agentbricks memory pipeline get NAME
```

`NAME` is a pipeline id or full `memory-pipelines/<id>` resource name.

##### `agentbricks memory pipeline update`

```text
agentbricks memory pipeline update NAME [options]
```

| Option | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--display-name` | string | - | no | New human-readable pipeline name. |
| `--model` | string | - | no | Model service for distillation; empty string clears the override. |
| `--trigger <manual\|scheduled>` | `manual` \| `scheduled` | - | no | How runs start: only when you start one, or also automatically about every 24 hours. |
| `--instructions` | string | - | no | Instructions steering distillation: inline text or @path to a UTF-8 file. |

Load complex instructions from a UTF-8 file (also supported by `create`):

```sh
agentbricks memory pipeline update p-123 --instructions @/path/to/instructions.md \
  --model system.ai.gpt-5-6-sol
```

Instructions are sent as `dreamer_policy.instructions` with that leaf field in the update mask.
`--model ""` clears the override. Create and update send only `dreamer_policy.model`;
updates use the `dreamer_policy.model` field mask. There is no legacy top-level model fallback.
List and detail views accept either model location, preferring the nested value when present.
`--trigger` updates `dreamer_policy.trigger`: `manual` maps to `MANUAL_ONLY`, and
`scheduled` maps to `SCHEDULED`. Omit the flag to preserve the current trigger.

##### `agentbricks memory pipeline pause`

```sh
agentbricks memory pipeline pause NAME
```

Set `dreamer_policy.trigger` to `MANUAL_ONLY` to stop scheduled runs. Active runs continue;
manual `run` calls remain available. `NAME` accepts a pipeline id or full resource name.
Returns an error if the pipeline is already manual-only (paused).

##### `agentbricks memory pipeline resume`

```sh
agentbricks memory pipeline resume NAME
```

Set `dreamer_policy.trigger` to `SCHEDULED` to enable automatic runs about every 24 hours.
This also enables scheduling for a pipeline originally created as manual-only.
Returns an error if the pipeline is already scheduled (resumed).
Both commands update only the trigger and support the global `--output json` option.

##### `agentbricks memory pipeline delete`

```text
agentbricks memory pipeline delete NAME [options]
```

| Option | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--yes`, `-y` | flag | false | no | Skip the confirmation prompt. |

##### `agentbricks memory pipeline runs`

| Subcommand | Description |
| --- | --- |
| [`memory pipeline runs get`](#agentbricks-memory-pipeline-runs-get) | Get run details through the memory pipeline API. |
| [`memory pipeline runs list`](#agentbricks-memory-pipeline-runs-list) | List recorded pipeline runs, most recently started first. |

###### `agentbricks memory pipeline runs list`

```sh
agentbricks --profile dogfood memory pipeline runs list memory-pipelines/<id> --page-size 50
```

`NAME` accepts a pipeline id or full `memory-pipelines/<id>` resource name.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--page-size` | nonnegative integer | 50 | Maximum results; zero uses the server default of 50, values above 1000 are capped by the server. |
| `--page-token` | string | - | Continuation token from the previous page for this pipeline. |

Lists persisted history from the memory-pipeline runs API, ordered by descending start time
and then run ID. Text output shows resource name, state, start time, and a next-page hint.
Global `--output json` returns the API response, including `pipeline_runs` and `next_page_token`.
History may be missing or stale after bookkeeping failures; use `runs get` for current Jobs status.

###### `agentbricks memory pipeline runs get`

```sh
agentbricks --profile dogfood memory pipeline runs get \
  memory-pipelines/2043d0c0-7d15-496e-83af-b6893165038f/runs/418898433707211
```

`NAME` must be `memory-pipelines/<pipeline-id>/runs/<numeric-run-id>`. This command calls
`GET /api/2.0/agents/{NAME}` using the selected profile and the complete run resource name.
Text output renders pipeline run details; global `--output json` returns the native
`PipelineRun` response. It no longer invokes `databricks jobs get-run` or returns the Jobs JSON schema.

##### `agentbricks memory pipeline run`

Trigger the pipeline's Dreamer engine and return the newly created run.

```text
agentbricks memory pipeline run NAME
```

`NAME` is a pipeline id or full `memory-pipelines/<id>` resource name.

### `agentbricks mcp`

Discover managed MCP Services available through Unity Catalog.

| Subcommand | Description |
| --- | --- |
| [`mcp list`](#agentbricks-mcp-list) | List MCP Services that can be added with ``agentbricks tools add mcp``. |

#### `agentbricks mcp list`

List MCP Services that can be added with ``agentbricks tools add mcp``.

```
agentbricks mcp list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--schema <SCHEMA>` | string | `system.ai` | no | Two-part Unity Catalog schema containing MCP Services. |

### `agentbricks sessions`

Manage an agent's conversations: session stores, the sessions in them, and their items.

A session is a single conversation between an actor (a user) and the agent. It holds that conversation's ordered transcript of items - the messages, tool calls, and results that make up its running state. A session store is the managed store that holds an agent's sessions and their items, giving it durable conversation history it can list, resume, fork, or delete.

This is the short-term, per-conversation counterpart to the cross-conversation memory in `agentbricks memory`.

| Subcommand | Description |
| --- | --- |
| [`sessions stores`](#agentbricks-sessions-stores) | Workspace-scoped session stores. |
| [`sessions items`](#agentbricks-sessions-items) | Transcript items within a session. |
| [`sessions bind`](#agentbricks-sessions-bind) | Bind session STORE to the agent by declaring it in agent.toml. |
| [`sessions unbind`](#agentbricks-sessions-unbind) | Remove the session store binding from the agent's agent.toml. |
| [`sessions create`](#agentbricks-sessions-create) | Create a session in a store. |
| [`sessions list`](#agentbricks-sessions-list) | List sessions in a store. |
| [`sessions get`](#agentbricks-sessions-get) | Get a session by id. |
| [`sessions update`](#agentbricks-sessions-update) | Update a session's metadata. |
| [`sessions delete`](#agentbricks-sessions-delete) | Delete a session. |
| [`sessions fork`](#agentbricks-sessions-fork) | Fork a session into a new independent top-level session. |

#### `agentbricks sessions stores`

Workspace-scoped session stores.

| Subcommand | Description |
| --- | --- |
| [`sessions stores create`](#agentbricks-sessions-stores-create) | Create a session store. |
| [`sessions stores list`](#agentbricks-sessions-stores-list) | List session stores in the workspace. |
| [`sessions stores get`](#agentbricks-sessions-stores-get) | Get a session store by name (the name in agent.toml) or resource name (session-stores/<name>). |
| [`sessions stores update`](#agentbricks-sessions-stores-update) | Update a store's description and/or metadata. |
| [`sessions stores delete`](#agentbricks-sessions-stores-delete) | Delete a session store. |
| [`sessions stores permissions`](#agentbricks-sessions-stores-permissions) | Inspect and manage store access. |

##### `agentbricks sessions stores create`

Create a session store.

```
agentbricks sessions stores create [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <NAME>` (`--display-name`) | string | - | yes | Workspace-unique store name, 3-63 chars (--display-name is accepted as an alias). |
| `--description <DESCRIPTION>` | string | - | no | - |
| `--metadata <METADATA>` | string | - | no | JSON object of string labels. |

##### `agentbricks sessions stores list`

List session stores in the workspace.

Without --name or --all, shows 25 per page: it paginates interactively on a terminal and otherwise
prints the --page-token for the next page. --name and --all page through every store and print one
complete list. The API has no server-side filter, so --name reads every page.

```
agentbricks sessions stores list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <NAME>` | string | - | no | Only stores whose name contains TEXT (case-insensitive). Reads every page. |
| `--all` | flag | - | no | Page through every store. |
| `--page-size <PAGE_SIZE>` | integer | `25` | no | Stores per page without --name or --all. |
| `--page-token <PAGE_TOKEN>` | string | - | no | Resume from a previous page's token (not with --name or --all). |

##### `agentbricks sessions stores get`

Get a session store by name (the name in agent.toml) or resource name (session-stores/<name>).

```
agentbricks sessions stores get NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

##### `agentbricks sessions stores update`

Update a store's description and/or metadata.

```
agentbricks sessions stores update NAME [options]
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

##### `agentbricks sessions stores delete`

Delete a session store.

```
agentbricks sessions stores delete NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

##### `agentbricks sessions stores permissions`

Inspect and manage store access.

Get returns your effective permission. List shows direct grants, excluding implicit
ownership and inherited access. Grant, List, and Revoke require the creator or a
workspace admin. STORE uses the same identifiers as the store commands.

| Subcommand | Description |
| --- | --- |
| [`sessions stores permissions get`](#agentbricks-sessions-stores-permissions-get) | Show the authenticated caller's effective permission on STORE. |
| [`sessions stores permissions list`](#agentbricks-sessions-stores-permissions-list) | List explicit grants on STORE as its creator or a workspace admin. |
| [`sessions stores permissions grant`](#agentbricks-sessions-stores-permissions-grant) | Grant access on STORE as its creator or a workspace admin. |
| [`sessions stores permissions revoke`](#agentbricks-sessions-stores-permissions-revoke) | Revoke both READ and WRITE on STORE as its creator or a workspace admin. |

###### `agentbricks sessions stores permissions get`

Show the authenticated caller's effective permission on STORE.

Session stores accept a store name or resource name. Memory stores accept an ID,
resource name, or display name; ambiguous display names fail. No access is reported as
PERMISSION_LEVEL_UNSPECIFIED in JSON.

```
agentbricks sessions stores permissions get STORE
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | Store identifier; see command description. |

###### `agentbricks sessions stores permissions list`

List explicit grants on STORE as its creator or a workspace admin.

Accepts a session store name or memory store ID, or a resource name. Implicit ownership
and inherited access are excluded. Text output offers subsequent pages on a terminal;
JSON and piped output return one page unless --all is supplied.

```
agentbricks sessions stores permissions list STORE [options]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | Store identifier; see command description. |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--page-size <PAGE_SIZE>` | integer [1–1000] | `25` | no | Grants per page. |
| `--page-token <PAGE_TOKEN>` | string | - | no | Resume from a previous page (not with --all). |
| `--all` | flag | - | no | Fetch every page of direct grants. |

###### `agentbricks sessions stores permissions grant`

Grant access on STORE as its creator or a workspace admin.

Specify exactly one of --user or --service-principal.
Accepts a session store name or memory store ID, or a resource name.

```
agentbricks sessions stores permissions grant STORE [options]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | Store identifier; see command description. |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--service-principal <SERVICE_PRINCIPAL>` | string | - | no | Service principal application/client ID, not a numeric SCIM ID (not with --user). |
| `--user <USER>` | string | - | no | User email (not with --service-principal). |
| `--permission <READ\|WRITE>` | `READ` \| `WRITE` | - | yes | Access to grant. WRITE includes READ; READ preserves an existing WRITE grant. |

###### `agentbricks sessions stores permissions revoke`

Revoke both READ and WRITE on STORE as its creator or a workspace admin.

Specify exactly one of --user or --service-principal.
Accepts a session store name or memory store ID, or a resource name. Repeating a revoke
succeeds. Ownership cannot be revoked; inherited access is not removed.

```
agentbricks sessions stores permissions revoke STORE [options]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | Store identifier; see command description. |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--service-principal <SERVICE_PRINCIPAL>` | string | - | no | Service principal application/client ID, not a numeric SCIM ID (not with --user). |
| `--user <USER>` | string | - | no | User email (not with --service-principal). |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

#### `agentbricks sessions items`

Transcript items within a session.

| Subcommand | Description |
| --- | --- |
| [`sessions items list`](#agentbricks-sessions-items-list) | List transcript items in a session. |
| [`sessions items append`](#agentbricks-sessions-items-append) | Append one or more items to a session (atomic, in order). |
| [`sessions items pop`](#agentbricks-sessions-items-pop) | Remove and return the most recent item. |
| [`sessions items clear`](#agentbricks-sessions-items-clear) | Remove all items from a session. |

##### `agentbricks sessions items list`

List transcript items in a session.

```
agentbricks sessions items list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--session-id <SESSION_ID>` | string | - | yes | - |
| `--order-by <ORDER_BY>` | string | - | no | 'create_time asc' or 'create_time desc'. |
| `--page-size <PAGE_SIZE>` | integer | - | no | - |
| `--page-token <PAGE_TOKEN>` | string | - | no | - |

##### `agentbricks sessions items append`

Append one or more items to a session (atomic, in order).

```
agentbricks sessions items append [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--session-id <SESSION_ID>` | string | - | yes | - |
| `--data <DATA>` | string | - | no | One item's JSON data (repeatable). |
| `--file <FILE>` | path | - | no | JSON array of item data values. |

##### `agentbricks sessions items pop`

Remove and return the most recent item.

```
agentbricks sessions items pop [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--session-id <SESSION_ID>` | string | - | yes | - |

##### `agentbricks sessions items clear`

Remove all items from a session.

```
agentbricks sessions items clear [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--session-id <SESSION_ID>` | string | - | yes | - |

#### `agentbricks sessions bind`

Bind session STORE to the agent by declaring it in agent.toml.

This only edits agent.toml - it does not create the store. `agentbricks deploy` creates any declared store that doesn't exist yet and grants the deployed app's service principal access to it.

```
agentbricks sessions bind STORE [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `STORE` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |

#### `agentbricks sessions unbind`

Remove the session store binding from the agent's agent.toml.

This only edits agent.toml. The next `agentbricks deploy` removes the managed environment reference from app.yaml; previously granted managed-store access is not revoked.

```
agentbricks sessions unbind [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |

#### `agentbricks sessions create`

Create a session in a store.

```
agentbricks sessions create [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--actor-id <ACTOR_ID>` | string | - | yes | Application actor id (child must match parent). |
| `--session-id <SESSION_ID>` | string | - | no | Optional caller-chosen id. |
| `--parent-session-id <PARENT_SESSION_ID>` | string | - | no | - |
| `--metadata <METADATA>` | string | - | no | JSON object of string labels. |

#### `agentbricks sessions list`

List sessions in a store.

```
agentbricks sessions list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | yes | - |
| `--filter <FILTER>` | string | - | no | e.g. actor_id = "support-123". |
| `--order-by <ORDER_BY>` | string | - | no | e.g. 'last_activity_time desc'. |
| `--page-size <PAGE_SIZE>` | integer | - | no | - |
| `--page-token <PAGE_TOKEN>` | string | - | no | - |

#### `agentbricks sessions get`

Get a session by id.

```
agentbricks sessions get SESSION_ID [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `SESSION_ID` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--store <STORE>` | string | - | no | Session store name (required in this preview). |

#### `agentbricks sessions update`

Update a session's metadata.

```
agentbricks sessions update SESSION_ID [options]
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

#### `agentbricks sessions delete`

Delete a session.

```
agentbricks sessions delete SESSION_ID [options]
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

#### `agentbricks sessions fork`

Fork a session into a new independent top-level session.

```
agentbricks sessions fork [SOURCE_SESSION_ID] [options]
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

### `agentbricks tracing`

Configure MLflow tracing for your deployed agents, and inspect the traces.

| Subcommand | Description |
| --- | --- |
| [`tracing bind`](#agentbricks-tracing-bind) | Bind tracing to an experiment, by name or id (one required). |
| [`tracing unbind`](#agentbricks-tracing-unbind) | Unbind tracing (remove the binding), turning tracing off for the deployed agent (deploy-only; `agentbricks dev` still traces locally). |
| [`tracing list`](#agentbricks-tracing-list) | List recent agent traces in an experiment. |
| [`tracing get`](#agentbricks-tracing-get) | Get a single trace by id (status, latency, span count, previews). |

#### `agentbricks tracing bind`

Bind tracing to an experiment, by name or id. Requires one of them (like `agentbricks memory/sessions bind`); the binding's presence is what turns tracing on.

The experiment is stored as a NAME, not an id, so the binding stays valid across workspaces/profiles - Agent Bricks CLI creates or reuses it in the active workspace at deploy. ``--experiment-id`` (e.g. from the experiment's URL) is a convenience: it's resolved to the experiment's name and stored as a name, never as an id.

```
agentbricks tracing bind [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--experiment-name <EXPERIMENT_NAME>` | string | - | no | MLflow experiment name to trace to - an absolute workspace path, e.g. /Shared/agentbricks_traces/&lt;agent&gt; or /Users/&lt;you&gt;/agentbricks_traces/&lt;agent&gt;. Agent Bricks CLI creates or reuses it at deploy. Mutually exclusive with --experiment-id. |
| `--experiment-id <EXPERIMENT_ID>` | string | - | no | MLflow experiment id (e.g. copied from the experiment's workspace URL) to trace to. Resolved to the experiment's name and stored as a name - Agent Bricks CLI stores names, not ids, so the binding stays valid across workspaces. Mutually exclusive with --experiment-name. |
| `--source <SOURCE>` | path | `.` | no | Project directory containing agent.toml. Defaults to the current directory. |

#### `agentbricks tracing unbind`

Unbind tracing: remove the experiment binding from agent.toml, turning tracing off for the DEPLOYED agent (`agentbricks deploy` then wires no MLflow env).

Deploy-only: `agentbricks dev` still traces locally to its own MLflow server, so you keep local traces while the deployed agent stays untraced.

```
agentbricks tracing unbind [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Project directory containing agent.toml. Defaults to the current directory. |

#### `agentbricks tracing list`

List recent agent traces in an experiment.

An explicit ``--experiment-name`` / ``--experiment-id`` reads that workspace experiment and must name one that exists (errors otherwise, so a typo isn't mistaken for an empty experiment). With neither, this project's experiment is read: the workspace one if it's been provisioned (by `agentbricks deploy`), otherwise the local `agentbricks dev` store (`.agentbricks/mlflow.db`), so a not-yet-deployed dev run's traces still show up here (tagged "(local dev)"). Nothing traced anywhere yet lists nothing. A UC-backed experiment is read through a SQL warehouse (``--warehouse``).

```
agentbricks tracing list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--experiment-name <EXPERIMENT_NAME>` | string | - | no | MLflow experiment name to read (an absolute workspace path). Default: this project's experiment. |
| `--experiment-id <EXPERIMENT_ID>` | string | - | no | MLflow experiment id to read (e.g. from the experiment URL). Mutually exclusive with --experiment-name. |
| `--warehouse <WAREHOUSE_ID>` | string | - | no | SQL warehouse id used to read traces from a UC-backed experiment (required for UC experiments; ignored for managed). Falls back to the MLFLOW_TRACING_SQL_WAREHOUSE_ID env var. |
| `--limit <LIMIT>` | integer | `20` | no | - |
| `--source <SOURCE>` | path | `.` | no | Project directory to resolve the default experiment from (default: current dir). |

#### `agentbricks tracing get`

Get a single trace by id (status, latency, span count, previews).

Reads from the same place as `agentbricks tracing list`: an explicit ``--experiment-name`` / ``--experiment-id`` targets that workspace store and must name one that exists (errors otherwise); otherwise this project's workspace experiment if provisioned, else its local `agentbricks dev` store. A UC-backed experiment is read through a SQL warehouse (``--warehouse``).

```
agentbricks tracing get TRACE_ID [options]
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

### `agentbricks experimental`

Commands whose interface is still being designed. They work and are tested, but their names, options, and behavior may change between releases without a deprecation period. The agent.toml tables they write live under `[experimental]` for the same reason, so they can graduate to top-level tables later.

| Subcommand | Description |
| --- | --- |
| [`experimental models`](#agentbricks-experimental-models) | Choose, evaluate, and upgrade the models behind your agent |
| [`experimental prompts`](#agentbricks-experimental-prompts) | Declare the Prompt Registry prompts your agent loads |

### `agentbricks experimental models`

Choose, evaluate, and upgrade the models behind your agent. Each LLM call site in the agent (a *role*: for example `router` and `writer` in a compound agent, or the single `agent` role in a one-model agent) calls a Unity Catalog AI Gateway model service you own (`catalog.schema.name`, declared under `[experimental.model_services.<role>]` in agent.toml) instead of a hardcoded `system.ai.*` model:

```toml
[experimental.model_services.router]
name = "main.my_agent.router_llm"
default = "system.ai.claude-haiku-4-5"

[experimental.model_services.writer]
name = "main.my_agent.writer_llm"
default = "system.ai.claude-sonnet-4-5"
```

`agentbricks deploy` creates each service if it's missing (routed to its binding's default model), grants the app's service principal EXECUTE on it, and wires it in as `AGENT_MODEL_SERVICE_<ROLE>`; the agent reads it with `resolve_model_service("<role>")`. After that, switching the model behind a service takes effect without a code change or redeploy, and `models upgrade` searches every role's model, plus the prompts bound with [`experimental prompts`](#agentbricks-experimental-prompts), together. Upgrade runs and switches are recorded in `.agentbricks/model_upgrades.json`; switches can be rolled back.

| Subcommand | Description |
| --- | --- |
| [`models bind`](#agentbricks-experimental-models-bind) | Declare the model service for one of the agent's LLM calls in agent.toml (creates nothing; deploy provisions it). |
| [`models unbind`](#agentbricks-experimental-models-unbind) | Remove a model-service binding from agent.toml (the service is left in place). |
| [`models list`](#agentbricks-experimental-models-list) | List the `system.ai.*` chat models you can route the agent to. |
| [`models status`](#agentbricks-experimental-models-status) | Show each bound model service, the model behind it now, and the latest upgrade run with its recommendation. |
| [`models set`](#agentbricks-experimental-models-set) | Switch one model service to a named model. |
| [`models rollback`](#agentbricks-experimental-models-rollback) | Undo the last apply or set: every model it switched and every prompt alias it moved. |
| [`models upgrade`](#agentbricks-experimental-models-upgrade) | Search models for every role, and optionally rewritten prompts, together, as a Databricks job. |
| [`models apply`](#agentbricks-experimental-models-apply) | Apply the latest finished upgrade run's recommendation: every model and prompt it changed. |

#### `agentbricks experimental models bind`

Bind model service SERVICE to one of the agent's LLM calls by declaring it in agent.toml. Bind one service per call site in a compound agent, each under its own `--role`. This only edits agent.toml; `agentbricks deploy` creates the service if it doesn't exist, routed to `--default`. Deploy never repoints an existing service, so a redeploy can't undo an upgrade.

```
agentbricks experimental models bind SERVICE [options]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `SERVICE` | yes | Three-part UC name: `catalog.schema.name`. |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--role <ROLE>` | string | `agent` | no | Name for the LLM call site this service backs (lowercase identifier, e.g. `router`). |
| `--default <MODEL>` | string | - | no | `system.ai.*` model deploy routes the service to when it creates it (the `system.ai.` prefix is optional). Omit to keep a recorded default. |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

#### `agentbricks experimental models unbind`

Remove a model-service binding from agent.toml. The next deploy removes that role's `AGENT_MODEL_SERVICE_<ROLE>` entry from app.yaml, so the LLM call uses its own default model directly again. The service itself is left in place.

```
agentbricks experimental models unbind [options]
```

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--role <ROLE>` | string | only bound role | no | Which bound LLM call site. Required when more than one role is bound. |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

#### `agentbricks experimental models list`

List the chat-capable `system.ai.*` models in the workspace.

```
agentbricks experimental models list
```

#### `agentbricks experimental models status`

Show each bound model service and the model it routes to now, the last recorded switch, and the latest upgrade run: the job's state and, once it has finished, each role's per-model scores and recommendation.

```
agentbricks experimental models status [options]
```

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

#### `agentbricks experimental models set`

Switch one bound model service to MODEL. Prompts for confirmation; `-o json` switches only with `--yes`.

```
agentbricks experimental models set MODEL [options]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `MODEL` | yes | A `system.ai.*` model (the prefix is optional). |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--role <ROLE>` | string | only bound role | no | Which bound LLM call site. Required when more than one role is bound. |
| `--yes` (`-y`) | flag | - | no | Switch without asking for confirmation. |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

#### `agentbricks experimental models rollback`

Undo the last `models apply` or `models set`. Every model service the action switched goes back to its previous model, and every prompt alias it moved (for example `@production`) goes back to the version it pointed at before, the one `@production_previous` marks. If a step fails, run it again to finish the remaining changes; completed steps are skipped. Once the action is fully undone, another rollback undoes the action before that. Refuses if a model or prompt alias moved to an unrelated value after the action. A rollback with `--role` records that model as restored and leaves the action's other models and prompts available for a later rollback. Moving prompt aliases needs the `upgrade` extra installed locally.

```
agentbricks experimental models rollback [options]
```

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--role <ROLE>` | string | - | no | Roll back only this role's last model switch; prompts are left alone. |
| `--yes` (`-y`) | flag | - | no | Switch without asking for confirmation. |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

#### `agentbricks experimental models upgrade`

Search candidate models for every bound role together, and with `--prompt` rewritten versions of the agent's MLflow Prompt Registry prompts, as a serverless Databricks job. Every combination is scored on the whole agent, so a cheaper router that hurts the writer's answers loses. The search never runs on your machine (it can take hours), so this command uploads the project to `/Workspace/Users/<you>/agentbricks_model_upgrades/<name>` (its own folder, separate from the deployed app's source), submits a one-time run, and returns. The job's environment installs the uploaded project from its `pyproject.toml` plus `databricks-agentbricks[upgrade]`.

The job imports the `predict_fn`, eval data, and scorers you name from the project (`module:attr` references, the same inputs `optimize_prompts_and_models` takes) and runs the whole agent on each eval record per candidate combination, against temporary `<service>_exp_<id>` copies of the model services (production traffic is untouched). Each search uses its own copies and deletes only those copies when it exits, so overlapping jobs can evaluate independently. It sets each `AGENT_MODEL_SERVICE_<ROLE>` before importing your code, so every LLM call that uses `resolve_model_service("<role>")` hits its copy. The winner balances quality, latency, and cost (`--quality-weight`, `--latency-weight`, `--cost-weight`); at equal quality the cheaper models win. The run logs to the MLflow experiment `<trace experiment>-model-upgrades`, never to the agent's own trace experiment.

Nothing changes until the recommendation is applied. By default the command returns once the job is submitted: check on it with `agentbricks experimental models status`, then run `agentbricks experimental models apply`. `--apply ask` waits for the result and asks before changing anything, and `--apply auto` applies it without asking. `--run-on local` runs the search in this process instead of a job, which needs the `upgrade` extra and the project's dependencies installed locally.

When the run finishes, the command prints a summary: a table of each role's current and recommended model, each role's candidate scores, the change in the combined quality, latency, and cost score, the prompts that were rewritten, and the MLflow run that recorded it.

```
agentbricks experimental models upgrade --candidates MODEL --predict REF --train-data REF --val-data REF --scorer REF [options]
```

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--candidates <ROLE=MODEL,...>` (`-c`) | string | - | for a model search | `system.ai.*` models to evaluate for one role, e.g. `router=claude-haiku-4-5,gpt-5-4-nano`. Repeat per role; the `ROLE=` prefix is optional when only one role is bound. Roles you don't list keep their current model. Each role's current model is always its baseline. Omit to optimize the agent's prompts only. |
| `--prompt <URI>` | `prompts:/` URI | prompts bound in agent.toml | no | Prompt Registry prompt to optimize alongside the models (e.g. `prompts:/main.my_agent.system@production`). Repeat for more. Defaults to the prompts bound with `prompts bind`, at `@production`. The agent must load it with `load_prompt` and call `.format()` on every call. |
| `--models-only` | flag | - | no | Search models only, leaving the agent's prompts as they are. |
| `--predict <REF>` | `module:attr` | - | yes | Your agent's `predict_fn`: a sync callable taking one eval record's `inputs` dict. |
| `--train-data <REF>` | `module:attr` | - | yes | Training records: a list of `{"inputs": ..., "expectations": ...}`. |
| `--val-data <REF>` | `module:attr` | - | yes | Validation records, same shape as `--train-data`. |
| `--scorer <REF>` | `module:attr` | - | yes | A scorer or list of scorers (MLflow scorers, or `(inputs, expectations, answer) -> float`). Repeat for more. |
| `--budget <N>` | integer >= 1 | 10 x eval records | no | Evaluation budget, in agent runs. |
| `--quality-weight <NUMBER>` | number >= 0 | `0.7` | no | How much the scorers' quality counts toward picking the winner. The three weights are relative: they're scaled to sum to 1. |
| `--latency-weight <NUMBER>` | number >= 0 | `0.2` | no | How much latency counts toward picking the winner. |
| `--cost-weight <NUMBER>` | number >= 0 | `0.1` | no | How much token cost counts toward picking the winner. |
| `--latency-gate <SECONDS>` | number >= 1 | `60` | no | A record slower than this scores 0 on every objective. Raise it for agents that legitimately take longer, or for `--run-on local`, where each call takes longer than on a serverless job. |
| `--timeout-hours <H>` | number >= 0.1 | `6` | no | Cancel the job if it runs longer than this. |
| `--run-on <WHERE>` | `job`, `local` | `job` | no | Where the search runs: a serverless Databricks job, or this process. |
| `--apply <MODE>` | `never`, `ask`, `auto` | `never` | no | What to do with the recommendation: leave it for `models apply`, wait and ask, or wait and apply without asking. Under `-o json`, `ask` applies nothing. |
| `--wait` | flag | - | no | Wait for the job to finish and show its recommendation (Ctrl-C detaches; the job keeps running). Implied by `--apply ask` / `auto`. |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

#### `agentbricks experimental models apply`

Apply the latest finished upgrade run's recommendation with the optimizer's `promote_to_prod`: it repoints every model service whose model changed and registers each rewritten prompt as a new version (its alias moves there; the prior version keeps `@production_previous`), rolling all of it back if any step fails. Needs the `upgrade` extra installed locally. Errors if the run hasn't finished, or if a service or prompt alias changed after the run started. All prompt aliases are checked before any production update, including prompts whose templates are unchanged. Prompts for confirmation; `-o json` switches only with `--yes`.

```
agentbricks experimental models apply [options]
```

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--yes` (`-y`) | flag | - | no | Switch without asking for confirmation. |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

### `agentbricks experimental prompts`

Declare the MLflow Prompt Registry prompts your agent loads. Each one is a binding under `[experimental.prompts]` in agent.toml, keyed by a short name:

```toml
[experimental.prompts]
writer = "main.my_agent.writer"
```

`agentbricks deploy` grants the app's service principal what the Prompt Registry requires to load them: USE SCHEMA, EXECUTE, CREATE FUNCTION, and MANAGE on each prompt's schema, plus USE CATALOG.

Bound prompts are what [`models upgrade`](#agentbricks-experimental-models-upgrade) rewrites by default, and what [`models apply`](#agentbricks-experimental-models-apply) and [`models rollback`](#agentbricks-experimental-models-rollback) move the `@production` alias of.

| Subcommand | Description |
| --- | --- |
| [`prompts bind`](#agentbricks-experimental-prompts-bind) | Declare a Prompt Registry prompt the agent loads, so deploy grants the app access to it. |
| [`prompts unbind`](#agentbricks-experimental-prompts-unbind) | Remove a prompt binding from agent.toml (the prompt is left in place). |
| [`prompts list`](#agentbricks-experimental-prompts-list) | List the bound prompts and the version each one's `@production` alias points at. |

#### `agentbricks experimental prompts bind`

Declare MLflow Prompt Registry prompt PROMPT (`catalog.schema.name`) that the agent loads. This only edits agent.toml. `agentbricks deploy` grants the app's service principal USE SCHEMA, EXECUTE, CREATE FUNCTION, and MANAGE on the prompt's schema, which the Prompt Registry requires even to load a prompt.

```
agentbricks experimental prompts bind PROMPT [options]
```

_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `PROMPT` | yes | Three-part Prompt Registry name: `catalog.schema.name`. |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--key <KEY>` | string | prompt's own name | no | Name for this prompt in agent.toml (lowercase identifier, e.g. `writer`). |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

#### `agentbricks experimental prompts unbind`

Remove prompt binding KEY from agent.toml. The prompt itself, and any grants already applied, are left in place.

```
agentbricks experimental prompts unbind KEY [options]
```

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

#### `agentbricks experimental prompts list`

List the prompts bound in agent.toml and the version each one's `@production` alias points at (`-` when the prompt or alias doesn't exist yet).

```
agentbricks experimental prompts list [options]
```

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Agent Bricks project containing agent.toml. |

### `agentbricks deploy`

Deploy your agent to Databricks Apps and get back a hosted URL to try it.

Rolls the agent out to Databricks Apps and prints the URL where you (or anyone you share it with) can use it. The deployed agent reaches Databricks model serving through the AI Gateway using the app's own identity - no model keys to configure - and `deploy` also reconciles the stores declared in agent.toml and wires in any tracing.

NAME is recorded in agent.toml on the first deploy, so a later `agentbricks deploy` from the project directory can omit it (passing NAME again updates the recorded name). Deployed apps are named `agent-bricks-<name>`. App names cap at 30 characters, so NAME can be up to 17 (`agentbricks init` records one that fits). A longer name fails before anything is provisioned, with a suggested name that fits. A project with a recorded base name reuses that app when NAME is omitted; you can also pass the full app name to update it. Use the full app name with the `agentbricks deployments` commands.

If a first deploy stops partway, the app it created is kept: re-run `agentbricks deploy` to resume, or remove it with `agentbricks deployments delete <app-name>`.

Any memory/session store declared in agent.toml (for example, by `agentbricks memory/sessions bind`) is created if it doesn't exist yet; agent.toml itself is never modified for stores.

Scaling to multiple instances (--instances) uses best-effort sticky routing, so a browser session automatically stays on one instance.

To keep a session on one app replica (sticky routing), API clients must resend a stable UUID (for example, their session id) in the `X-Routing-Key` request header on every request. The header is a routing hint only - non-blank, no more than 128 UTF-8 bytes, used verbatim - not authentication and not the session id itself (send the session id in the request body). Example: `X-Routing-Key: <uuid>`

A first deploy takes a few minutes (mostly starting app compute). Deploy finishes once the agent answers requests. From a coding agent or script with a short command timeout, use --no-wait and then `agentbricks deployments wait`.

```
agentbricks deploy [NAME] [options]
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
| `--allowed-origin <ORIGIN>` | string (repeatable) | - | no | Allow this browser origin to call the Agent Bricks server. Repeat for multiple origins; replaces and records the allowlist in agent.toml. Omit to reuse the recorded origins. |
| `--clear-allowed-origins` | flag | - | no | Clear the recorded browser origin allowlist and disable runtime CORS on this deploy. |
| `--allow-user-scope-update` | flag | - | no | Allow Agent Bricks to add missing user API scopes to an existing App from managed-tool inference and `[auth.user]` additions. Once added, later deploys do not need this flag. |
| `--no-wait` | flag | - | no | Run the quick checks, then finish the deploy in a background process and exit. Follow it with `agentbricks deployments wait`. |

`deploy` refuses to start while another deploy of the same app is still running from the project. It records its progress in `.agentbricks/deploy-<app>.json`, and a `--no-wait` deploy logs to `.agentbricks/deploy-<app>.log`. Long steps print `• <step>… M:SS` every 15 seconds: on stderr when output isn't a terminal or under `-o json` (whose stdout is exactly one JSON document), and in the spinner on a terminal. After rollout, `deploy` waits up to 5 minutes for the app to stop returning 502/503/504 (`• Waiting for the agent to answer…`), then prints `Deployed`; if it never answers, it exits 1 and points at `agentbricks deployments logs <app>`. When the profile can't call the app (a PAT, or 401/403), it prints `Couldn't check the agent answers: <reason>.` on stderr and still succeeds. The `-o json` result includes `"ready"`: `true` once the app answered, `null` when it couldn't be checked. With `--no-wait`, text output is `Deploy of <app> started in the background. Next: agentbricks deployments wait <app>` and `-o json` output is `{"deployment", "status": "started", "log", "state_file"}`.

### `agentbricks deployments`

Inspect and manage deployed agents: list, get, wait, stream logs, start, stop, or delete.

| Subcommand | Description |
| --- | --- |
| [`deployments list`](#agentbricks-deployments-list) | List Agent Bricks deployments (apps named `agent-bricks-*`). |
| [`deployments get`](#agentbricks-deployments-get) | Get an agent deployment's details. |
| [`deployments wait`](#agentbricks-deployments-wait) | Wait for a deployment to finish, for up to --timeout seconds, then report it. |
| [`deployments logs`](#agentbricks-deployments-logs) | Stream a deployment's logs. |
| [`deployments start`](#agentbricks-deployments-start) | Start a deployment. |
| [`deployments stop`](#agentbricks-deployments-stop) | Stop a deployment. |
| [`deployments delete`](#agentbricks-deployments-delete) | Delete a deployment, and its Runtime Store when managed provisioning is enabled. |

#### `agentbricks deployments list`

List Agent Bricks deployments (apps named `agent-bricks-*`).

```
agentbricks deployments list
```

#### `agentbricks deployments get`

Get an agent deployment's details.

```
agentbricks deployments get NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

#### `agentbricks deployments wait`

Wait for a deployment to finish, for up to --timeout seconds, then report it.

Follows the deploy started by `agentbricks deploy` (including --no-wait) in this project, or otherwise the App's state in the workspace. NAME defaults to the deployment recorded in agent.toml.

Exit status: 0 deployed and answering (or, with a note, deployed but not checkable); 1 failed; 75 still deploying at the timeout (run the same command again).

```
agentbricks deployments wait [NAME] [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | no | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--timeout <TIMEOUT>` | integer range | `100` | no | Seconds to wait before returning exit 75. |
| `--source <SOURCE>` | path | `.` | no | Project directory the deploy ran from. |

On exit 0 it prints the URL and the invoke command; on exit 1, the error and the last 20 lines of the deploy log (or the App's status message); on exit 75, `still deploying: <phase>, M:SS elapsed — run the same command again`. A deploy whose app still returns 502/503/504 is in the `Waiting for the agent to answer` phase. Phase changes and heartbeats go to stderr. When the profile can't call the app, exit 0 comes with `Couldn't check the agent answers` on stderr. Under `-o json`, stdout is exactly one object `{deployment, status, phase, elapsed_s, url, error, log_tail, ready}` with the same exit code; `ready` is `true` once the app answered and `null` otherwise.

#### `agentbricks deployments logs`

Stream a deployment's logs.

```
agentbricks deployments logs NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

#### `agentbricks deployments start`

Start a deployment.

```
agentbricks deployments start NAME
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

#### `agentbricks deployments stop`

Stop a deployment.

```
agentbricks deployments stop NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

#### `agentbricks deployments delete`

Delete a deployment, and its Runtime Store when managed provisioning is enabled.

```
agentbricks deployments delete NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--delete-source` | flag | - | no | Also delete the workspace folder the deployment's source was uploaded to, once the deployment is gone. |
| `--yes`, `-y` | flag | - | no | Skip the confirmation prompt. |

### `agentbricks endpoint`

Invoke arbitrary HTTP endpoints.

| Subcommand | Description |
| --- | --- |
| [`endpoint invoke`](#agentbricks-endpoint-invoke) | Send one HTTP request to a Databricks App or arbitrary URL. |

#### `agentbricks endpoint invoke`

Send one HTTP request to a Databricks App or arbitrary URL.

```
agentbricks endpoint invoke [APP] [options]
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
| `--routing-key <ROUTING_KEY>` | string | - | no | Sticky-routing key; set it to your stable session id to keep a session on one app replica. Sent verbatim as the X-Routing-Key header (default: generated for a Databricks App). Routing only: while session_id is a natural routing key, it is recommended to pass the session id in the --json body for session continuity. |
| `--timeout <TIMEOUT>` | float range | `300.0` | no | - |
| `--auth`, `--no-auth` | flag | - | no | Inject Databricks OAuth authentication. |

### `agentbricks tools`

Discover available integrations and manage an agent's tool bindings.

Tools are what let an agent act beyond the language model itself - query governed data, call a service, or run a function - and each one is recorded in agent.toml so `agentbricks dev` / `agentbricks deploy` wire it in. `agentbricks tools add` manages these Databricks-managed tool types:

sandbox Run code in a Databricks sandbox (system.ai.sandbox) with the selected identity's Unity Catalog access. mcp A Databricks-managed MCP service (see `agentbricks tools list --kind mcp`), e.g. system.ai.web_search. uc-function An existing Unity Catalog function (catalog.schema.function). genie-one Workspace-wide Genie One MCP tools. genie-agent Native Genie conversation tools for a configured space ID.

Browse available integrations with `agentbricks tools list`, add one with `agentbricks tools add <type>`, and drop a binding with `agentbricks tools remove`. Review agent.toml for configured managed tools and MCP bindings. The list shows addable integrations, not configured bindings or individual operations inside an MCP service. Custom Python tools are code-first - write them directly in your project's code rather than through the CLI.

| Subcommand | Description |
| --- | --- |
| [`tools add`](#agentbricks-tools-add) | Add a managed sandbox, MCP service, UC function, or Genie tool binding. |
| [`tools list`](#agentbricks-tools-list) | List available integrations to add, not configured agent bindings. |
| [`tools remove`](#agentbricks-tools-remove) | Remove a managed tool binding from this agent. |

#### `agentbricks tools add`

Add a managed sandbox, MCP service, UC function, or Genie tool binding.

Subcommands target the current directory by default.

Pass --source PATH to target another project.

Review that project's agent.toml to check configured managed tools and MCP bindings.

| Subcommand | Description |
| --- | --- |
| [`tools add sandbox`](#agentbricks-tools-add-sandbox) | Add a sandbox tool (system.ai.sandbox) that runs code with the selected identity's access. |
| [`tools add mcp`](#agentbricks-tools-add-mcp) | Validate and add a Databricks-managed MCP service as a tool. |
| [`tools add uc-function`](#agentbricks-tools-add-uc-function) | Add an existing Unity Catalog function (catalog.schema.function) as a tool. |
| [`tools add genie-one`](#agentbricks-tools-add-genie-one) | Add workspace-wide Genie One MCP tools. |
| [`tools add genie-agent`](#agentbricks-tools-add-genie-agent) | Add native Genie conversation tools for a 32-character lowercase hexadecimal SPACE_ID. |

##### `agentbricks tools add sandbox`

Add a sandbox tool (system.ai.sandbox) that runs code with the selected identity's access.

Sandbox code can read and write whatever Unity Catalog data the user (--auth user) or the App service principal (--auth app) can. Review the target project's agent.toml to check configured managed tools and MCP bindings.

```
agentbricks tools add sandbox [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <TOOL_ID>` | string | `sandbox` | no | - |
| `--auth <user|app>` | `user` \| `app` | `user` | no | - |
| `--databricks-access-token-included`, `--no-databricks-access-token-included` | boolean flag | `--databricks-access-token-included` | no | Expose the selected Databricks credential to sandbox code. |
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |

##### `agentbricks tools add mcp`

Validate and add a Databricks-managed MCP service as a tool.

Use `agentbricks tools list --kind mcp` for available services. Review the target project's agent.toml to check configured managed tools and MCP bindings.

```
agentbricks tools add mcp SERVICE [options]
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
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |

##### `agentbricks tools add uc-function`

Add an existing Unity Catalog function (catalog.schema.function) as a tool.

```
agentbricks tools add uc-function FUNCTION_NAME [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `FUNCTION_NAME` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <TOOL_ID>` | string | - | no | - |
| `--auth <user|app>` | `user` \| `app` | `user` | no | - |
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |

##### `agentbricks tools add genie-one`

Add workspace-wide Genie One MCP tools.

```
agentbricks tools add genie-one [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <TOOL_ID>` | string | `genie_one` | no | - |
| `--auth <user|app>` | `user` \| `app` | `user` | no | - |
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |

##### `agentbricks tools add genie-agent`

Add native Genie conversation tools for a 32-character lowercase hexadecimal SPACE_ID.

```
agentbricks tools add genie-agent SPACE_ID [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `SPACE_ID` | yes | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--name <TOOL_ID>` | string | `genie_agent` | no | - |
| `--auth <user|app>` | `user` \| `app` | `user` | no | - |
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |

#### `agentbricks tools list`

List available integrations to add, not configured agent bindings.

By default, show built-in add recipes plus caller-visible MCP Services in system.ai. `--kind mcp` limits discovery to MCP Services; `--schema catalog.schema` replaces system.ai. UC-function and Genie Agent recipes require concrete resource identifiers. Sandbox and Genie One recipes need no additional argument.

No agent project is required. MCP discovery uses your Databricks profile; local recipes do not authenticate. This does not scan every workspace schema or list individual MCP operations. API failures return a nonzero exit status and mark discovery incomplete, not empty.

Review agent.toml to check configured managed tools and MCP bindings. The former configured list and `--source` option are removed. JSON discovery uses schema_version 2 and available_tools.

```
agentbricks tools list [options]
```


_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--kind <sandbox|mcp|uc-function|genie-one|genie-agent>` | choice | - | no | Show one integration kind. Sandbox, uc-function, genie-one, and genie-agent show local add recipes only. |
| `--schema <SCHEMA>` | string | - | no | Two-part UC schema: catalog.schema (default: system.ai). Requires --kind mcp. |

#### `agentbricks tools remove`

Remove a managed tool binding from this agent.

```
agentbricks tools remove TOOL_ID [MCP_SERVICE] [options]
```


_Arguments_

| Argument | Required | Description |
| --- | --- | --- |
| `TOOL_ID` | yes | - |
| `MCP_SERVICE` | no | - |

_Options_

| Option | Values | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `--source <SOURCE>` | path | `.` | no | Agent project containing agent.toml. |
