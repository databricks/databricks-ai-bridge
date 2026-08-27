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

## Agent workflow

```sh
mason init --framework langgraph --profile <profile> my-agent
cd my-agent
mason dev
uv run pytest
mason deploy
```

`mason init` seeds the project `.env` with the explicitly selected profile. `mason dev` runs the
project's `start-server` script. From the project directory, `mason deploy` infers the source and
deployment name and reuses that seeded profile.

Before a template merges into the default template repository, pass its repository and ref:

```sh
mason init --framework langgraph --profile <profile> \
  --repo https://github.com/<owner>/databricks-ai-bridge.git \
  --ref <branch> \
  my-agent
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
  init         [DIRECTORY] --framework openai|langgraph [--profile P]
  dev          [--source PATH] [--port N]
  memory
    stores     create | list | get | update | delete
    entries    create | get | list | search | update | delete
  sessions     create | list | get | update | delete | fork
    stores     create | list | get | update | delete
    items      list | append | pop | clear
  tracing
    setup      --catalog C --schema S [--experiment E]
    list | get | instrument
  deploy       [name] [--source PATH] [--with-memory-store N]
               [--with-session-store N] [--with-traces C.S] [--create-stores]
  deployments  list | get | logs | start | stop | delete
```
