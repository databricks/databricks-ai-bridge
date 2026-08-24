# `databricks-mason`

Mason is an experimental CLI for the Databricks `agents/v1` preview APIs and
integrated agent deployment flows. It manages memory, sessions, tracing, and
deployments from one authenticated command.

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

For live tracing commands, install the optional MLflow dependency:

```sh
pip install 'databricks-mason[tracing]'
```

## Authentication

Authentication uses a `.databrickscfg` profile. Pass `--profile/-p`, or save a
default profile once:

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
