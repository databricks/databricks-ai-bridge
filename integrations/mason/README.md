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
  memory
    stores     create | list | get | update | delete
    entries    create | get | list | search | update | delete
  sessions     create | list | get | update | delete | fork
    stores     create | list | get | update | delete
    items      list | append | pop | clear
  tracing
    setup      --catalog C --schema S [--experiment E]
    list | get | instrument
  add-sandbox  --scope SCOPE [--scope SCOPE ...]
               [--permission read_only|read_write] [--source PATH]
  deploy       <name> --source PATH [--with-memory-store N]
               [--with-session-store N] [--with-traces C.S] [--create-stores]
  deployments  list | get | logs | start | stop | delete
```

## Add a downscoped sandbox

From a Mason agent project, add the `system.ai.sandbox` MCP server and fix its
allowed resources at configuration time:

```sh
mason add-sandbox --scope catalog.schema.volume
```

The command updates `agent/mcps.py`. Every sandbox call carries the selected
downscope in MCP `_meta`, outside the tool arguments controlled by the model.
Scopes default to read-only access. Repeat `--scope` to allow more than one
resource, use `/Workspace/...` for a workspace path, or prefix a table with
`table:`:

```sh
mason add-sandbox \
  --scope catalog.schema.volume \
  --scope /Workspace/Users/alice@example.com \
  --scope table:catalog.schema.table
```

Pass `--permission read_write` to grant write access to every supplied scope,
or `--source /path/to/project` when running outside the project root. Re-running
the command with the same policy leaves the existing configuration unchanged;
a different policy fails without modifying the file so it cannot silently report
stale access. Edit or remove the generated block before intentionally changing
the sandbox policy.
