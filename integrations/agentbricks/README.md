# Agent Bricks CLI (`agentbricks`)

Agent Bricks CLI is an experimental command-line interface for building and deploying custom
agents on Databricks. It manages memory, sessions, tracing, and deployments through one CLI.

> The underlying APIs are in preview and may need workspace enablement.

## Prerequisites

- **Python ≥3.10** to run the CLI; generated agent projects require Python 3.11+.
- **[`uv`](https://docs.astral.sh/uv/)** to scaffold, run, and deploy an agent. Resource
  commands can run without it; reading local traces requires it.
- **[Databricks CLI](https://docs.databricks.com/dev-tools/cli/)** for browser-based
  `agentbricks login`. An already-authenticated profile does not require it.

## Installation

The `databricks-agentbricks` Python distribution installs the `agentbricks` command and AgentKit.
Install it from PyPI:

```sh
pip install databricks-agentbricks
```

From source:

```sh
pip install 'git+https://github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/agentbricks'
```

The base package includes the CLI, store SDK, and `DurableAgentServer` HTTP runtime. Generated projects
declare their framework dependencies automatically.

## Quickstart

Create a new agent project with LangGraph (the default framework). Use `--framework openai`
instead for an **OpenAI Agents SDK** project. This chooses the agent framework, not the model
provider; both templates call a Databricks AI Gateway model.

```sh
agentbricks init my-agent --framework langgraph --profile <profile>
cd my-agent
agentbricks login --profile <profile>
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
`agentbricks login --profile <profile>` validates existing credentials and, if needed in an
interactive terminal, runs `databricks auth login`. It remembers the selected profile in
`~/.agentbricks/config.json`; `agentbricks logout` forgets that selection without revoking its
credentials. For non-interactive use, authenticate the profile first. You can skip `login` when
Databricks SDK default authentication is already configured, or pass `--profile/-p` before an
individual command. Use `--output json` for scripting.

## How it works

`agentbricks init` copies framework and runtime files into your project; you own and can edit
those files. The `databricks-agentbricks` dependency supplies AgentKit and the runtime library.
`agentbricks deploy` uses your source and `agent.toml` declarations to run the agent as a
Databricks App.

![Deployment: from a local project to a Databricks App](docs/deployment.svg)

## Continue with a specific task

| Task | Guide |
| --- | --- |
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
