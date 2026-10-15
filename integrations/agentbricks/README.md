# Agent Bricks CLI (`agentbricks`)

Agent Bricks CLI is an experimental command-line interface for building and deploying custom
agents on Databricks. It manages memory, sessions, tracing, and deployments through one CLI.

> The underlying APIs are in preview and may need workspace enablement.

## Prerequisites

- **Python ≥3.10** to run the CLI; generated agent projects require Python 3.11+.
- **[`uv`](https://docs.astral.sh/uv/)** to scaffold, run, and deploy an agent. Resource
  commands can run without it; reading local traces requires it.
- **[Databricks CLI](https://docs.databricks.com/dev-tools/cli/)** for browser-based
  `agentbricks auth login`. An already-authenticated profile does not require it.

## Installation

The `databricks-agentbricks` Python distribution installs the `agentbricks` command and AgentKit.
Install it from PyPI:

```sh
pip install databricks-agentbricks
```

Install from PyPI rather than from this repository. Templates on `main` can depend on SDK changes that
are not yet released, while generated projects install the released SDK from PyPI, so a CLI installed
from source can scaffold projects that fail to start. To work on Agent Bricks itself, see
[CONTRIBUTING.md](CONTRIBUTING.md).

The base package includes the CLI, store SDK, and `DurableAgentServer` HTTP runtime. Generated projects
declare their framework dependencies automatically.

## Shell completion
Add this to `~/.zshrc`:
```sh
eval "$(_AGENTBRICKS_COMPLETE=zsh_source agentbricks)"
```

## Authentication

Deployment reports the user API scopes being added and the tool binding or
`[auth.user].additional_api_scopes` entry that requires each scope. When the required grants already
exist, it reports that scopes are unchanged. Existing grants are preserved when tools change.

Agent Bricks CLI uses [Databricks authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
Sign in to a profile, then scaffold a project that records it in its `.env`:

```sh
agentbricks auth login <profile>
agentbricks init my-agent --profile <profile>
```

`agentbricks auth login` validates the profile and, in an interactive terminal, runs
`databricks auth login --profile <profile>` if it is not authenticated yet (this needs the Databricks
CLI; in CI or without a terminal, authenticate the profile beforehand). It saves nothing. To change a
project's profile later, run `agentbricks auth switch <profile>` in the project directory, and
`agentbricks auth describe` shows which profile (and workspace) a directory would use.

Inside an Agent Bricks project (a directory with `.agentbricks/project.toml` or `app.yaml`), every
command uses the profile in the project's `.env`. Outside a project, pass the global `--profile/-p`
option before a command, for example `agentbricks --profile <profile> tools list`, or set
`DATABRICKS_CONFIG_PROFILE`. `-p` can go before or after the command —
`agentbricks tools list -p <profile>` works too (the trailing form goes after the full subcommand).
Use `--output json` for scripting.

The profile is resolved in this order, most to least specific: the `--profile/-p` flag, the project
`.env`'s `DATABRICKS_CONFIG_PROFILE` (inside an Agent Bricks project; `dev` and `deploy` resolve
against their source directory so the CLI and the locally running agent use one profile), the
`DATABRICKS_CONFIG_PROFILE` environment variable, and finally the Databricks SDK's default
authentication resolution — which also covers `DATABRICKS_HOST`/`DATABRICKS_TOKEN` environment
authentication, not only `[DEFAULT]` in `~/.databrickscfg`. `dev` passes the resolved profile to
the agent (unless `.env` sets `DATABRICKS_TOKEN` or `DATABRICKS_CLIENT_ID`/
`DATABRICKS_CLIENT_SECRET`). Every command that reaches the workspace prints one stderr line naming
the profile, where it came from, and its host, so the target workspace is never implicit.

The CLI sends best effort command usage telemetry after leaf commands. Its log records the static
command path, package version, operating system, duration, and process outcome (exit code and, for
nonzero exits, a bounded error category). Parameter entries are limited to options explicitly
supplied on the command line. Free-form options contribute only their static name. Declared Click
choices contribute an `AGENTBRICKS_CLI_`-prefixed value. Click boolean and numeric options
contribute their typed values; non-finite floats and integers outside the log schema's 64-bit
range remain presence-only. Hidden and deprecated options follow the same policy, and repeated
options contribute one parameter entry per supplied value, up to 64 entries. Positional arguments,
raw command arguments, paths, IDs, secrets, arbitrary free-form option values, and exception
messages are omitted. Set
`AGENTBRICKS_DISABLE_TELEMETRY=1` to disable these uploads.

Telemetry runs only when a leaf command is invoked, whether its callback succeeds or fails. Help
and errors rejected during Click parsing, before a leaf starts, do not produce an event. Uploads
reuse the command's workspace client or let the Databricks SDK resolve the selected or default
profile. Authenticate first with `agentbricks auth login <profile>` or configure SDK auth;
an SDK `external-browser` profile may prompt during telemetry even for local commands such as
`init`. The CLI waits up to two seconds for client setup, authentication, and upload on a daemon
thread, returning sooner if the worker finishes. This is not a delivery guarantee: unfinished
uploads can be lost when the process exits after the allowance expires. Missing credentials or a
failed upload never change the command result.

## Quickstart

The shortest path from a blank directory to a running and deployed agent:

```sh
agentbricks auth login <profile>
agentbricks init my-agent --profile <profile>  # LangGraph by default; add --framework=openai for OpenAI Agents SDK
cd my-agent
agentbricks dev                 # optional: develop and test locally
agentbricks deploy my-agent     # deploy to Databricks
```

To move an existing project to a different workspace later, run `agentbricks auth switch <other>` in
its directory. Rerunning `agentbricks init` on a project it scaffolded changes nothing. (`--existing`
is different: it prepares a migration for a project Agent Bricks didn't generate.)

`dev` and `deploy` validate the resolved profile's authentication up front and fail fast with the
`agentbricks auth login` command to run — an unauthenticated external-browser profile would
otherwise hang on a browser login that never completes.

`agentbricks dev` runs the agent locally on `http://localhost:8000`, wrapping the Databricks Apps
local runtime so local behavior matches a deployment.

`agentbricks deploy my-agent` deploys a Databricks App named `agent-bricks-my-agent`, provisions the
stores declared in `agent.toml`, and grants the app's service principal access to the stores and
direct App-auth tool resources declared there. Use `agentbricks deployments list` to find deployed
apps, and `agentbricks deployments get agent-bricks-my-agent` to print an app's URL and status.

`agentbricks init` declares default memory and session stores in `agent.toml`, so the deployed agent has
long-term memory and durable conversation history. It creates `<name>-<6-letter-token>-memory` and
`<name>-<6-letter-token>-sessions`, and records both names in `agent.toml`.
`agentbricks deploy` creates them if they don't exist yet. Point the agent at stores you already have with
`agentbricks memory bind <name>` / `agentbricks sessions bind <name>`, or scaffold without stores using
`agentbricks init --server custom` (see [Initialize the chat app demo](#initialize-the-chat-app-demo)).

To exercise the agent (locally under `agentbricks dev` or once deployed), `agentbricks endpoint invoke` sends
it an HTTP request. MLflow tracing is on by default (`agentbricks init` binds a default
`/Shared/agentbricks_traces/<project>` experiment): `agentbricks dev` traces to a local MLflow server under
`.agentbricks/` and `agentbricks deploy` to the bound workspace experiment; `agentbricks tracing list` shows the
available traces.

## Coding-agent workflow skills

The CLI bundles an `agent-bricks-workflow` skill for creating, developing, deploying, validating,
and troubleshooting agents. It is separate from the project-specific `agent-bricks-migrate` skill
generated by `init --existing`. Bundled guidance matches the installed CLI and requires no
download or separate setup command.

New `agentbricks init` projects already include the workflow skill. One shared bundle lives at
`.agents/skills/agent-bricks-workflow/`; `.claude/skills/agent-bricks-workflow/` and
`.agent/skills/agent-bricks-workflow/` contain pointers to it, not duplicate references.
Migration discovery uses these same three roots. Installation is project-local, preserves unrelated
configuration, leaves identical skills unchanged, and refuses to overwrite a different skill.

Current coding agents discover these locations:

| Coding agent | Project skill discovery |
| --- | --- |
| [Claude Code](https://code.claude.com/docs/en/skills) | `.claude/skills/` |
| [Codex](https://developers.openai.com/codex/skills/) | `.agents/skills/` |
| [Cursor](https://cursor.com/docs/skills) | `.agents/skills/` compatibility discovery |
| [GitHub Copilot](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills) | `.agents/skills/` |
| [Gemini CLI](https://geminicli.com/docs/cli/skills/) | `.agents/skills/` compatibility discovery |
| [OpenCode](https://opencode.ai/docs/skills/) | `.agents/skills/` compatibility discovery |
| Other agents | Read the manifest directly |

The `.agent/skills/` pointer retains the existing migration discovery location for agents using that
convention. Support depends on the coding-agent version and its skill settings; restart the session
if discovery is cached. Generated `AGENTS.md` also links to the workflow skill. The skill is
advisory and does not authorize deployment or live calls.

When the CLI detects a coding-agent environment, it emits a single-line pointer to the bundled
skill on stderr, including for `--help`. Ordinary human invocations do not emit this hint.
Coding agents in existing projects can read that bundled file directly without installing
project-local skills or changing project configuration. Detection is best-effort and does not
gate skill installation during `init` or native discovery. Set `AGENTBRICKS_DISABLE_AGENT_HINT=1`
to suppress automatic hints. Hints never alter JSON stdout and are not emitted during shell
completion, SDK imports, or deployed agent runtime startup.

For a deployment-only task, local `dev` is optional. The workflow skill describes focused
verification of the deployed agent, rather than treating local execution as proof of managed
memory or session persistence.

## Inspect and clean up a project

Read `agent.toml` for the recorded agent name, declared memory/session stores, tracing, and tool
bindings. A declaration does not confirm that the resource exists or that the deployed app can use it.

| Inspect | Command | Details |
| --- | --- | --- |
| Local onboarding checks | `agentbricks doctor .` | Offline; does not verify bindings or deployed resources. |
| Agent Bricks apps in a workspace | `agentbricks -p <profile> deployments list` | Lists apps with the `agent-bricks-` prefix in the selected workspace, including other projects. |
| App status and URL | `agentbricks -p <profile> deployments get agent-bricks-<name>` | Queries the selected workspace for the named app. |
| Memory store IDs and details | `agentbricks -p <profile> memory stores list`, then `agentbricks -p <profile> memory stores get <resource-name>` | Workspace-wide; can include other projects' stores. Use the ID or resource name, not the display name. |
| Session store names and details | `agentbricks -p <profile> sessions stores list`, then `agentbricks -p <profile> sessions stores get <name>` | Workspace-wide; can include other projects' stores. Use the store name. |

Review each confirmation. Delete a store only after confirming that you own it and no other agent
needs it, using the same profile and identifier used to inspect it.

| Clean up | Command | Details |
| --- | --- | --- |
| Deployment | `agentbricks -p <profile> deployments delete agent-bricks-<name>` | Managed Runtime Store cleanup checks the app identity and runs before app deletion; a cleanup failure retains the app for retry. Memory/session stores, tracing experiments, tools, and source files are retained. |
| [Memory store](cli.md#agentbricks-memory-stores-delete) | `agentbricks -p <profile> memory stores delete <resource-name>` | Soft-deletes the store after confirmation. |
| [Session store](cli.md#agentbricks-sessions-stores-delete) | `agentbricks -p <profile> sessions stores delete <name>` | Deletes the store after confirmation. |

## Public names

Use `agentbricks` for the CLI and `AgentKitClient` from `databricks_agentkit` for the Python SDK. New
projects store local state under `.agentbricks/` and `~/.agentbricks/`, use
`server = "agentbricks"` in `agent.toml`, and deploy apps with the `agent-bricks-` prefix.

## AgentKit SDK

`AgentKitClient` adds a small resource-oriented layer over AgentKit APIs. Pass it an
authenticated Databricks `WorkspaceClient`, or omit the argument to use the
Databricks SDK's default authentication resolution:

```python
from databricks.sdk import WorkspaceClient
from databricks_agentkit import AgentKitClient

agentkit = AgentKitClient(WorkspaceClient(profile="my-workspace"))

session_store = agentkit.session_stores.create("support-agent-sessions")
session = session_store.add(actor_id="customer-123", session_id="case-456")
session.append_items(
    [
        {"type": "message", "role": "user", "content": "I need help with my cluster."},
        {"type": "message", "role": "assistant", "content": "Let's take a look."},
    ]
)

memory_store = agentkit.memory_stores.create("coding-agent-memory")
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

The root collections manage stores: `agentkit.memory_stores.create/get/list` and
`agentkit.session_stores.create/get/list`. A returned store owns operations on its
contents, such as `memory_store.add()`, `memory_store.put()`,
`memory_store.get("memory-id")`, `memory_store.list()`, and `memory_store.search()`, or
`session_store.add()`, `session_store.get("session-id")`, and `session_store.list()`.
Returned memories, sessions, and stores own their `update()` and `delete()` operations.

`memory_store.add()` fails with `ALREADY_EXISTS` when a memory already exists at the
same `actor_id`, `session_id`, and `path`. `memory_store.put()` creates the memory or
updates the existing one: it replaces the content, or with `append=True` adds `content`
as new lines, skipping lines already present. `put()` is not atomic, so concurrent
writers to one path can overwrite each other. `add()`'s `write_mode` argument is
deprecated; use `put()` instead. `memory_store.list(actor_id=..., path=...)` returns the
memories at one exact path.

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

`DurableAgentServer` runs your agent through one HTTP API for synchronous, streaming, and background
invocations. Register an `@app.invoke` handler, publish progress with `await context.emit(event)`,
and return a JSON result. You can also add your own FastAPI endpoints.

Start from a template, edit the agent code in `agent/`, and run it locally before deploying:

### CLI telemetry

```sh
agentbricks init my-agent --framework langgraph --profile <profile>
cd my-agent
agentbricks auth login <profile>
agentbricks dev
```

`init` copies a project with a chat UI and records the profile in its local `.env`. `login`
authenticates that profile for the CLI. Open the URL printed by `dev` (usually
`http://localhost:8000`) and send a message. Stop `dev` with Ctrl-C, then deploy:

```sh
agentbricks deploy my-agent --source .
agentbricks deployments get agent-bricks-my-agent
```

`deploy` creates or updates the Databricks App and provisions the stores declared in
`agent.toml`. It attempts to grant the App access to them; inspect its output for access or
tracing warnings. `deployments get` prints the App URL and status. Open the deployed chat UI
and send a message to verify it.

### Invoke from the command line

The chat UI is one client of the generated agent. To call its API directly, send a new UUID as
`id` for each turn. Both generated frameworks require a nonempty top-level `session_id` on every
request; reuse it for turns in the same conversation:

```sh
SESSION_ID=$(python3 -c 'import uuid; print(uuid.uuid4())')
INVOCATION_ID=$(python3 -c 'import uuid; print(uuid.uuid4())')
agentbricks endpoint invoke agent-bricks-my-agent \
  --path /api/invocations \
  --json "{\"id\":\"$INVOCATION_ID\",\"session_id\":\"$SESSION_ID\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}}"
```

Use `agentbricks endpoint invoke --url http://localhost:8000` instead of the App name to
call `dev` while it is running. Invoking a deployed App requires an OAuth-authenticated
profile; PAT profiles cannot access App routes. See the
[runtime guide](src/databricks_agentkit/runtime/README.md) for streaming, background requests,
and recovery.

## First customization: add a custom tool

Add `agent/tools/count_words.py` to the generated project. Use the version for your chosen
framework:

LangGraph:

```python
from langchain_core.tools import tool


@tool
def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())
```

OpenAI Agents SDK:

```python
from agents import function_tool


@function_tool
def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())
```

Both templates discover decorated tools in `agent/tools/` automatically. Run `agentbricks dev`
again and ask the local chat UI: “Use count_words to count the words in: the quick brown fox.”
The tool returns `4`. Edit `agent/agent.py` to change the model, instructions, or agent logic.
For Databricks-managed tools, see the [Agent tools guide](docs/agent-tools.md).

## Authentication

Agent Bricks CLI uses [Databricks authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
`agentbricks auth login <profile>` validates existing credentials and, if needed in an
interactive terminal, runs `databricks auth login`. It remembers the selected profile in
`~/.agentbricks/config.json`; `agentbricks logout` forgets that selection without revoking its
credentials. For non-interactive use, authenticate the profile first. You can skip `login` when
Databricks SDK default authentication is already configured, or pass `--profile/-p` before an
individual command. Use `--output json` for scripting.

### CLI telemetry

The CLI sends best effort command usage telemetry after leaf commands. Its log records the static
command path, package version, operating system, duration, and process outcome (exit code and, for
nonzero exits, a bounded error category). Parameter entries are limited to options explicitly
supplied on the command line. Free-form options contribute only their static name. Declared Click
choices contribute an `AGENTBRICKS_CLI_`-prefixed value. Click boolean and numeric options
contribute their typed values; non-finite floats and integers outside the log schema's 64-bit
range remain presence-only. Hidden and deprecated options follow the same policy, and repeated
options contribute one parameter entry per supplied value, up to 64 entries. Positional arguments,
raw command arguments, paths, IDs, secrets, arbitrary free-form option values, and exception
messages are omitted. Set
`AGENTBRICKS_DISABLE_TELEMETRY=1` to disable these uploads.

Telemetry runs only when a leaf command is invoked, whether its callback succeeds or fails. Help
and errors rejected during Click parsing, before a leaf starts, do not produce an event. Uploads
reuse the command's workspace client or let the Databricks SDK resolve the selected or default
profile. Authenticate first with `agentbricks auth login <profile>` or configure SDK auth;
an SDK `external-browser` profile may prompt during telemetry even for local commands such as
`init`. The CLI waits up to two seconds for client setup, authentication, and upload on a daemon
thread, returning sooner if the worker finishes. This is not a delivery guarantee: unfinished
uploads can be lost when the process exits after the allowance expires. Missing credentials or a
failed upload never change the command result.

## How it works

`agentbricks init` copies framework and runtime files into your project; you own and can edit
those files. The `databricks-agentbricks` dependency supplies AgentKit and the runtime library.
`agentbricks deploy` uses your source and `agent.toml` declarations to run the agent as a
Databricks App.

![Deployment: from a local project to a Databricks App](docs/deployment.svg)

## Continue with a specific task

| Task | Guide |
| --- | --- |
| Use a coding agent with the bundled workflow skill | [Coding-agent skills](docs/coding-agent-skills.md) |
| Configure memory, sessions, or the AgentKit SDK | [Memory and sessions](docs/memory-and-sessions.md) |
| Add managed MCP, sandbox, UC function, or Genie tools | [Agent tools](docs/agent-tools.md) |
| Control deployment dependencies and resource lifecycle | [Deployment and lifecycle](docs/deploy-and-maintain.md) |
| Upgrade a customized generated project | [Upgrade a generated project](docs/upgrading-generated-projects.md) |
| Use streaming, background requests, or recovery | [Runtime guide](src/databricks_agentkit/runtime/README.md) |
| Bring an existing agent | [Migration guide](docs/migrating-existing-agents.md) |
| Inspect commands and options | [CLI reference](cli.md) |
| Understand the generated chat UI | [LangGraph](src/databricks_agentbricks/templates/ui/agent-langgraph/CHAT_APP.md) or [OpenAI Agents SDK](src/databricks_agentbricks/templates/ui/agent-openai/CHAT_APP.md) |

## Commands

See [the CLI command reference](cli.md) for all commands, arguments, and options. Built-in
examples are available at each level, such as `agentbricks deploy --help`.

For zsh completion, add this to `~/.zshrc`:

```sh
eval "$(_AGENTBRICKS_COMPLETE=zsh_source agentbricks)"
```

## Contributing

To change the CLI, AgentKit, runtime, or templates in this repository, see
[CONTRIBUTING.md](CONTRIBUTING.md) for local setup, testing, and releases.
