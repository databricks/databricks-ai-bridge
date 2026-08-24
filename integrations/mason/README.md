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

Configure credentials using
[Databricks CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication).
Run `mason login` to validate and remember an existing CLI profile, or pass `--profile/-p`
to select a profile for an individual command:

```sh
mason login --profile <profile>
mason sessions stores list
```

The default is stored in `~/.mason/config.json`; `mason logout` removes it.
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
  deploy       <name> --source PATH [--with-memory-store N]
               [--with-session-store N] [--with-traces C.S] [--create-stores]
  deployments  list | get | logs | start | stop | delete
```
