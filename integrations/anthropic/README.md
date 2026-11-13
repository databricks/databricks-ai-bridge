# databricks-anthropic

Run [Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) sessions in
[Databricks Sandbox](https://docs.databricks.com/aws/en/compute/serverless/sandbox) (Beta), served by a
**Databricks App**.

Anthropic runs the agent loop. Every tool call (bash, file edits, and optional Databricks MCP tools such as Genie
One) runs in an isolated Databricks Sandbox, one per session, in your workspace. A Claude Managed Agents
[self-hosted environment](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes) only queues the
work; the dispatcher App in [`app/`](app/README.md) claims each session and starts its worker in a sandbox. Inside the
sandbox the tools act with the sandbox's built-in Databricks credential, which is the App's service principal, so
Unity Catalog permissions apply to everything the agent does.

## How it works

```
Claude Console / your app          Anthropic                       Databricks workspace
  create session, send message ->  agent loop + work queue  <----  Dispatcher App (service principal)
                                   (self-hosted environment)        polls the queue (outbound HTTPS only)
                                                                    -> per-session Databricks Sandbox
                                                                       worker.py: EnvironmentWorker
                                   tool calls / results  <------->     bash, files, Genie One (MCP)
```

- **Outbound only.** The App long-polls Anthropic; nothing calls into the workspace.
- **One sandbox per session.** It is created on the session's first message, bootstrapped (`uv` venv with exactly
  pinned packages), and gets the worker, whose file hash is checked when it starts.
- **Narrow credentials.** The environment key, which can claim any session, stays in the App. A sandbox only gets
  its own work item's session token, in a `0600` file the worker reads and deletes before any tool runs. That token
  cannot read the work queue, other sessions, or memory stores that aren't attached to the session. The App never
  needs an Anthropic API key.
- **No login shells.** Commands in the sandbox run with `bash --noprofile --norc`, and the worker starts with an
  allow-listed environment, so files the agent plants in its home can't affect the next launch.

## Deploy the dispatcher App

See [`app/README.md`](app/README.md). In short, from this folder:

```bash
ANTHROPIC_ENVIRONMENT_KEY=... app/scripts/deploy.sh dev <profile> <environment id> --store-secrets
```

This builds the package wheel, deploys the bundle (the App, its secret scope and secret resources), stores the
environment key and a sandbox-name HMAC key in the scope, and starts the App. Then grant the App's service principal
least-privilege access to the data its agents may use (`app/README.md` lists the grants).

## Prerequisites

- A Databricks workspace with **Databricks Sandbox** enabled, and permission to create Apps and secret scopes.
- A Claude Developer Platform account with:
  - a **self-hosted** environment for the App, and its **environment key** (generated in the Console);
  - credits.
- For Genie One: Genie One enabled, and `EXECUTE` on the `system.ai.genie_one_mcp` MCP service for the App's
  service principal.

## Identity

Every session on the App's environment runs as the App's service principal. Whoever can create sessions on that
environment (members of the Anthropic workspace) gets that principal's Databricks access, so use a dedicated
environment, ideally in a dedicated Anthropic workspace, and grant the principal only data every such user may see.

## Genie One tools

The worker serves Databricks managed MCP servers as Managed Agents custom tools. It connects to each server from
inside the sandbox with the sandbox credential, so no Databricks token is given to Anthropic. The App enables Genie
One by default (`mcp` bundle variable). Declare the tools on your agent:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks_anthropic.sandbox.mcp import alist_mcp_custom_tools, genie_one_server

w = WorkspaceClient(config=Config(profile="my-workspace"))
tools = await alist_mcp_custom_tools(w, genie_one_server(w))   # genie_ask_and_wait, genie_get_query_result
agent = client.beta.agents.create(name=..., model=..., tools=[{"type": "agent_toolset_20260401"}, *tools])
```

`genie_ask_and_wait` calls Genie's `genie_ask` and polls `genie_poll_response` inside the sandbox, so Claude makes one
tool call per question instead of one per poll. Measured with Claude Haiku 5.5 on "get me total revenue from my
bakehouse":

| Mode | Genie tool calls by Claude | Cached input tokens read | Output tokens |
|------|----------------------------|--------------------------|---------------|
| raw tools (`genie_one_server(w, wait_tool=False)`) | 17 (1 ask + 16 polls) | ~190K | ~2.2K |
| `genie_ask_and_wait` (default) | 1 (7 polls inside the sandbox) | ~6–12K | ~110 |

## Behavior worth knowing

| Topic | Behavior |
|-------|----------|
| Lifecycle | `ephemeral` (the App's default): when the session goes idle, the worker optionally exports `outputs/` to a UC Volume, then **deletes** its sandbox; the next message gets a fresh one. `resumable` (library option): the worker **stops** the sandbox (files kept), and cleanup deletes it after `retention_days` or when the session is deleted |
| Auto-stop | Sandbox inactivity is measured by exec calls, so the worker sends an exec heartbeat while it serves a session. If it dies, `inactivity_timeout_s` (default 600) stops the sandbox |
| Runtime cap | A sandbox runs at most 24 hours per start |
| Sandbox names | Prefix + 24 hex characters of an HMAC of the session id (the App's `sandbox-id-hmac-key`), so names can't be predicted; an existing sandbox is used only if its tags match the session and environment |
| Duplicate delivery | A re-delivered work item whose worker is still running is left alone (`LaunchResult.reused`) |
| Memory stores | Attached memory stores mount at `/mnt/memory/<store>/` inside the sandbox |
| Cleanup | The App sweeps its own sandboxes (prefix **and** tags for its environment); it never touches others |
| Visibility | `list_sandboxes` returns only the caller's sandboxes, so `/api/status` is how operators see the App's |

## Library

The App is a thin host around this package, which you can also run elsewhere:

```python
import asyncio
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks_anthropic import DatabricksSandboxDispatcher, DatabricksSandboxLauncher

launcher = DatabricksSandboxLauncher(
    workspace_client=WorkspaceClient(config=Config()),   # sandboxes act as this identity
    environment_id="env_...",
    environment_key="...",
    sandbox_id_key="<random secret>",
)
asyncio.run(DatabricksSandboxDispatcher(launcher).run())
```

```sh
# After the package is released (until then, install from this folder: pip install '.[mcp]')
pip install databricks-anthropic           # dispatcher + launcher
pip install 'databricks-anthropic[mcp]'    # also serve Databricks MCP servers (Genie One) as custom tools
```

## Development

```sh
uv sync --all-extras --group tests
uv run pytest tests/unit_tests      # offline: library and App
uv run ruff check
```

Live tests in `tests/integration_tests` run real sessions (Claude Haiku, a $1 cap per session) against your
workspace and Claude account. They are opt-in:

```sh
cp .env.anthropic.template .env.anthropic   # API key, environment id and key for the tests
RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 DATABRICKS_SANDBOX_PROFILE=<profile> \
    uv run pytest tests/integration_tests -v -s
```

They cover sessions with and without Genie One, resume, memory stores, cleanup, Volume export, sandbox isolation
(what the in-sandbox token can reach), and, with `SANDBOX_E2E_APP=<app name>`, the deployed App serving a session.
Each run uses a unique sandbox prefix and deletes only what it created.

## License
This project is licensed under the [MIT License](LICENSE).

Thank you for using Databricks Anthropic!
