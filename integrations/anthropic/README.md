#  Databricks Anthropic Integration

The `databricks-anthropic` package runs [Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes) self-hosted sessions in [Databricks Sandbox](https://docs.databricks.com/aws/en/compute/serverless/sandbox). Claude runs the agent loop at Anthropic. Each session's tools (bash, files, code and Databricks Genie One) run in its own sandbox in your workspace, as the identity that runs the dispatcher.

## Installation

### From PyPI
```sh
pip install databricks-anthropic
pip install "databricks-anthropic[mcp]"   # to serve Genie One / Databricks MCP tools
```

### From Source
```sh
pip install git+https://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/anthropic
```

## Key Features

- **Dispatcher:** `DatabricksSandboxDispatcher` long-polls a self-hosted environment's work queue (outbound HTTPS only, no webhook) and hands each claimed session to the launcher.
- **One sandbox per session:** `DatabricksSandboxLauncher` creates or restarts the session's sandbox, builds the worker's virtualenv once, and starts Anthropic's `EnvironmentWorker` inside it.
- **Genie One from inside the sandbox:** Databricks managed MCP servers are served as custom tools with the sandbox's own credential, so no Databricks token goes to Anthropic. `genie_ask_and_wait` polls Genie inside the sandbox, so the model makes one tool call per question.
- **Lifecycle:** `ephemeral` (default) deletes the sandbox when the session goes idle; `resumable` stops it and keeps its files for the next message. Optional export of an output folder to a Unity Catalog Volume, and a cleanup sweep for stopped or orphaned sandboxes.
- **Memory stores:** the per-session secret is passed to the worker, so attached memory stores mount at `/mnt/memory/<store>/`.

## Getting Started

You need:
- a workspace with the Databricks Sandboxes preview enabled;
- a Databricks login for the dispatcher: a CLI profile, SDK browser login, or an OAuth M2M service principal (a profile, or `DATABRICKS_HOST` / `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`);
- a Claude Console **self-hosted** environment and its environment key.

### Run the dispatcher
```sh
export ANTHROPIC_ENVIRONMENT_ID=env_...
export ANTHROPIC_ENVIRONMENT_KEY=...
databricks-anthropic doctor --profile my-workspace --mcp genie-one     # checks every prerequisite
databricks-anthropic sandbox-dispatcher --profile my-workspace --mcp genie-one
```
Sessions created on that environment, from the Console or the API, are now served in Databricks Sandbox.

### Use it from Python
```python
import asyncio

from databricks_anthropic import DatabricksSandboxDispatcher, DatabricksSandboxLauncher, auth
from databricks_anthropic.sandbox.mcp import alist_mcp_custom_tools, genie_one_server

w = auth.login(profile="my-workspace")                    # sandboxes act as this identity
server = genie_one_server(w)                              # system.ai.genie_one_mcp (needs EXECUTE)
tools = asyncio.run(alist_mcp_custom_tools(w, server))    # declare these on the agent

launcher = DatabricksSandboxLauncher(
    workspace_client=w,
    environment_id="env_...",
    environment_key="...",
    mcp_servers=[server],
)
asyncio.run(DatabricksSandboxDispatcher(launcher).run())
```

Runnable examples are in [`examples/`](examples): a quickstart (bash and files in the session's sandbox) and a Genie One data analyst. Run them from `integrations/anthropic` after copying `.env.anthropic.template` to `.env.anthropic` (git-ignored) and filling it in, for example `uv run examples/quickstart/main.py --profile my-workspace`.

### Credentials
- The Anthropic API key is only needed where you create agents and sessions; it never enters a sandbox.
- The environment key and the per-session work secret reach the sandbox in a `0600` file that the worker reads and deletes before any tool runs, not through its process environment. The agent runs as the same Linux user as the worker, so treat a sandbox as able to reach the environment key: use one environment per trust boundary and rotate the key if a session may be compromised.
- Every Databricks call the connector makes sends the User-Agent product `ClaudeManagedAgents-DatabricksSandbox/0.1.0`. Databricks CLI processes and the agent's own SDK/CLI calls in the sandbox are tagged through `DATABRICKS_SDK_UPSTREAM` / `DATABRICKS_CLI_UPSTREAM`.

### Tests
```sh
uv run --exact --group tests pytest tests/unit_tests
# Live tests (create sandboxes; RUN_SANDBOX_E2E also runs real Claude sessions on Haiku, capped at $1 each):
RUN_SANDBOX_INTEGRATION_TESTS=1 DATABRICKS_SANDBOX_PROFILE=my-profile \
  ANTHROPIC_ENV_FILE=/path/to/.env.anthropic \
  uv run --exact --group tests --extra mcp pytest tests/integration_tests
```

---

## Contribution Guide
We welcome contributions! Please see our [contribution guidelines](https://github.com/databricks/databricks-ai-bridge/tree/main/integrations/langchain) for details.

## License
This project is licensed under the [MIT License](LICENSE).

Thank you for using Databricks Anthropic!
