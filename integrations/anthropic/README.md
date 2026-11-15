# databricks-anthropic

Run [Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) sessions in
[Databricks Sandbox](https://docs.databricks.com/aws/en/compute/serverless/sandbox) (Beta), served by a
**Databricks App**.

**Patterns.** Two are supported. **A: the Databricks App** (teams and production; see [Deploy the dispatcher
App](#deploy-the-dispatcher-app)). **B: the local dispatcher** (`databricks-anthropic run` on a laptop or VM, as your
Databricks CLI login; see [Run the dispatcher locally](#run-the-dispatcher-locally)).

Anthropic runs the agent loop. Every tool call (bash, file edits, and optional Databricks MCP tools such as Genie
One) runs in an isolated Databricks Sandbox, one per session, in your workspace, **as the Databricks user who started
the session**. A Claude Managed Agents
[self-hosted environment](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes) only queues the
work; the dispatcher App in [`app/`](app/README.md) claims each session and starts its worker in a sandbox.

- **User mode (the standard deploy).** A user starts a session in the Claude Console on the App's user environment.
  The first tool call returns a one-time sign-in link; the user signs in with Databricks, confirms, and replies
  "done". The sandbox is then created with the user's own credential, so Genie One and Unity Catalog reads use that
  user's grants, capped to the App's `user_api_scopes`.
- **Service mode.** Sessions on the App's service environment run as the App's service principal, for automation and
  shared agents.

## How it works

```
Claude Console                     Anthropic                       Databricks workspace
  create session, send message ->  agent loop + work queues  <---  Dispatcher App
                                   (self-hosted environments)       polls the queues (outbound HTTPS only)
  open the sign-in link once  ---------------------------------->   /connect: Databricks sign-in, binds the session
                                                                    -> per-session Databricks Sandbox, created as
                                                                       the signed-in user (or the App, service mode)
                                   tool calls / results  <------->     worker.py: bash, files, Genie One (MCP)
```

- **Outbound only.** The App long-polls Anthropic; nothing calls into the workspace except users opening the App.
- **One sandbox per session.** It is created on the session's first tool call (after sign-in, in user mode),
  bootstrapped (`uv` venv installed from hash-locked requirements, `--require-hashes`), and gets the worker, whose
  file hash is checked when it starts.
- **Narrow credentials.** The environment keys, which can claim any session, stay in the App. A sandbox only gets its
  own work item's session token, in a `0600` file the worker reads and deletes before any tool runs. That token cannot
  read the work queue, other sessions, or memory stores that aren't attached to the session. A user's Databricks token
  stays in App memory only, for minutes; the App needs an Anthropic API key only in user mode, to read session details
  for the sign-in page.
- **No login shells.** Commands in the sandbox run with `bash --noprofile --norc`, and the worker starts with an
  allow-listed environment, so files the agent plants in its home can't affect the next launch.

## Deploy the dispatcher App

See [`app/README.md`](app/README.md). In short, from this folder (user mode, the standard deploy):

```bash
ANTHROPIC_ENVIRONMENT_KEY=... ANTHROPIC_USER_ENVIRONMENT_KEY=... CMA_ANTHROPIC_API_KEY=... \
  app/scripts/deploy.sh dev <profile> <service environment id> --store-secrets -- \
  --var user_environment_id=<user environment id> --var user_agents=<agent id> \
  --var lakebase_branch=... --var lakebase_database=... --var lakebase_endpoint=...
```

This builds the package wheel, deploys the bundle (the App, its secret scope, secret resources and Lakebase
resource), stores the secrets in the scope, and starts the App. Then give users `CAN_USE` on the App and the grants
their agents need. For service mode only (no sign-in, Lakebase or API key), deploy target `dev-service` with just the
service environment, and grant the App's service principal least-privilege access (`app/README.md` lists the grants).

## Prerequisites

- A Databricks workspace with **Databricks Sandbox** enabled, and permission to create Apps and secret scopes.
- For user mode, a **Lakebase** (Autoscaling) branch and database the App may use.
- A Claude Developer Platform account with credits and, in one workspace:
  - a **self-hosted** environment for users and one for service mode, each with its **environment key** (generated in
    the Console);
  - for user mode, an **API key** for the App.
- For Genie One: Genie One enabled, and `EXECUTE` on the `system.ai.genie_one_mcp` MCP service for the users (user
  mode) or the App's service principal (service mode).

## Identity

- **User mode:** each session is bound to the Databricks user who signs in through its link, and its sandbox acts with
  that user's grants. Whoever opens a link supplies the identity, so the sign-in page asks users to confirm only
  sessions they started, and a session bound to another user is never taken over.
- **Service mode:** every session on the service environment runs as the App's service principal. Whoever can create
  sessions on that environment (members of the Anthropic workspace) gets that principal's Databricks access, so use a
  dedicated environment, ideally in a dedicated Anthropic workspace, and grant the principal only data every such user
  may see.

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
| Memory stores | Attached memory stores mount at `/mnt/memory/<store>/` inside the sandbox (service mode; off in user mode) |
| Cleanup | The App sweeps its own sandboxes (prefix **and** tags for its environment); it never touches others |
| Visibility | `list_sandboxes` returns only the caller's sandboxes: a user's sandboxes are visible to that user, and `/api/status` is how operators see the App's |

## Run the dispatcher locally

The `databricks-anthropic` command serves the same environment queue from your laptop or a VM, with one Databricks Sandbox
per session (the same launcher and worker as the App). Sessions from the Console, a script, or Claude Code via the SDK then run as you:

**Sign in once with the Databricks CLI** (OAuth; you don't need to configure an OAuth app):
1. Copy your workspace URL from the browser address bar while you're in the workspace. Keep the `https://` and leave
   off anything after the domain.
2. Run `databricks auth login --host https://<your-workspace-url>`.
3. Sign in through the browser window the CLI opens, and approve access. If you can act as a service principal, you're
   also asked whether to sign in as yourself or as that service principal.
4. Back in the terminal, keep the suggested profile name or type your own (for example `dev`). The CLI saves the
   workspace URL under that profile and refreshes the one-hour OAuth tokens for you. To sign in again later, or to add
   another workspace, run step 2 again with that workspace's URL.

Then:

```bash
databricks-anthropic init --profile dev --environment env_...      # prompts for the environment key (hidden)
databricks-anthropic doctor        # login, Sandbox API, environment, queue, no other dispatcher running
databricks-anthropic run           # until Ctrl-C or SIGTERM; --drain exits when the queue is empty
databricks-anthropic list | cleanup [--dry-run]
```

- **Identity.** Every session on the environment runs with the identity of `--profile`, which must be a Databricks
  CLI profile (from `databricks auth login`). The profile is always named explicitly: the CLI never picks `DEFAULT`
  or `DATABRICKS_CONFIG_PROFILE`. `run` prints whose access the sessions get.
- **Settings and keys.** `~/.config/databricks-anthropic/` (override with `DATABRICKS_ANTHROPIC_CONFIG_DIR`):
  `default.toml` holds settings, `default.secrets` (mode 0600) holds the environment key and the key that names
  sandboxes. Keys are prompted for (hidden), so they stay out of shell history and output. Re-running `init` keeps the
  sandbox-name key, so resumable sandboxes stay findable.
- **One dispatcher per environment.** A per-environment lock on this machine refuses a second `run`. `run` also refuses
  while another worker is polling the environment, unless `--force`. The lock uses `fcntl`, so it is POSIX only.
- **Extras.** `--mcp genie-one` needs `pip install 'databricks-anthropic[mcp]'`.
- **Older sandboxes.** Sandboxes from the earlier dev CLI (`cma-` names, no key) are not matched by `list` or `cleanup`.
  `databricks-anthropic cleanup --legacy-prefix cma- --dry-run` shows what it would remove; drop `--dry-run` to delete.

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
