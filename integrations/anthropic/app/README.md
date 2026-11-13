# Dispatcher App: serve Claude Managed Agents sessions from a Databricks App

This Databricks App is the dispatcher for a Claude Managed Agents **self-hosted** environment. It long-polls the
environment's work queue and starts each session's worker in its own Databricks Sandbox, **as the App's service
principal**. It runs inside your workspace and is always on, so sessions started from the Claude Console or the
Anthropic API are served without anything running on a laptop.

| What | Where |
|---|---|
| Poller, launcher, worker | the `databricks-anthropic` package (`../src`), installed from the wheel that `scripts/build_wheel.sh` builds |
| App code | `app.py` (routes), `service.py` (configuration, restart loop, status) |
| Infrastructure | `databricks.yml`: the App (one instance), its secret scope and secret resources |
| Secrets | `scripts/put_secrets.py`: `env-s-key` (the environment key) and `sandbox-id-hmac-key` (random) |
| Deploy | `scripts/deploy.sh` |

## Identity and isolation

- **Sessions run as the App's service principal.** Every tool call, Genie One question and Unity Catalog read uses
  its grants. Anyone who can create a session on the App's environment gets that access, so give the App its own
  environment (ideally its own Anthropic workspace) and grant the principal only data every such user may see.
- **The environment key stays in the App.** A sandbox only receives its own session's token, which can't read the
  work queue, other sessions or memory stores that aren't attached to the session
  (`tests/integration_tests/test_80_e2e_isolation.py`).
- **Sandbox names are unpredictable** (HMAC of the session id with `sandbox-id-hmac-key`), and an existing sandbox
  is used only if its tags show it was created for that session and environment.
- **Only the App sees its sandboxes.** `list_sandboxes` returns the caller's own sandboxes, even for a workspace
  admin, so `/api/status` reports them by state.
- **App access.** Grant `CAN_USE` to the users or groups who may view the status page. There are no write routes.

## Deploy

Prerequisites:
- the Databricks CLI, logged in to the workspace with a profile that can create Apps and secret scopes;
- Databricks Sandbox enabled in the workspace;
- in the Claude Console, a **self-hosted** environment for the App and its environment key.

From `integrations/anthropic`:

```bash
# First deploy: also stores the environment key (read from $ANTHROPIC_ENVIRONMENT_KEY, or prompted for)
ANTHROPIC_ENVIRONMENT_KEY=... app/scripts/deploy.sh dev <profile> <environment id> --store-secrets

# Later deploys (code or configuration changes)
app/scripts/deploy.sh dev <profile> <environment id>

# Optional bundle variables after --, e.g. export outputs/ to a Volume or turn Genie One off
app/scripts/deploy.sh dev <profile> <environment id> -- --var export_volume=/Volumes/main/agents/outputs --var mcp=
```

`deploy.sh` builds the connector wheel, deploys the bundle, stores the secrets (the secret scope has to exist before
the App, whose secret resources are checked when it is created), and starts the App. It passes `--profile`
explicitly, because a `DATABRICKS_CONFIG_PROFILE` in your shell would override the bundle's workspace.

Once `databricks-anthropic` is on PyPI, `requirements.txt` can pin `databricks-anthropic[mcp]==<version>` instead of
the built wheel.

### Grants for the App's service principal

Find its client id with `databricks apps get cma-dispatcher-<target>`, then grant least privilege, for example:

```bash
SP=<service_principal_client_id>
databricks grants update catalog <catalog> --json '{"changes":[{"principal":"'$SP'","add":["USE_CATALOG"]}]}'
databricks grants update schema <catalog>.<schema> --json '{"changes":[{"principal":"'$SP'","add":["USE_SCHEMA","SELECT"]}]}'
databricks grants update volume <catalog>.<schema>.<volume> --json '{"changes":[{"principal":"'$SP'","add":["READ_VOLUME","WRITE_VOLUME"]}]}'
```

Genie One also needs `EXECUTE` on the `system.ai.genie_one_mcp` MCP service (securable type `mcp_service`).

### Configuration (`databricks.yml` variables)

| Variable | Default | Meaning |
|---|---|---|
| `environment_id` | (required) | The self-hosted environment the App serves |
| `secret_scope` | `claude-managed-agents-dev` / `claude-managed-agents` | Holds `env-s-key` and `sandbox-id-hmac-key` |
| `mcp` | `genie-one` | Serve Genie One from the sandbox; empty to turn it off |
| `export_volume` | empty | UC Volume folder that receives each session's `outputs/` |

### App settings (`config.env` in `databricks.yml`)

| Variable | Set to | Meaning |
|---|---|---|
| `ANTHROPIC_ENVIRONMENT_ID` | `${var.environment_id}` | The environment the App serves |
| `ANTHROPIC_ENVIRONMENT_KEY` | secret resource `env-s-key` | Its environment key; stays in the App |
| `CMA_SANDBOX_ID_KEY` | secret resource `sandbox-id-hmac-key` | HMAC key for sandbox names |
| `CMA_SANDBOX_PREFIX` | `cmaapp-` | Sandbox id prefix; scopes the App's cleanup |
| `CMA_MCP` | `${var.mcp}` | `genie-one` or empty |
| `CMA_LIFECYCLE` | `ephemeral` | `ephemeral` or `resumable` |
| `CMA_EXPORT_VOLUME` | `${var.export_volume}` | Optional Volume folder for `outputs/` |
| `CMA_MAX_CONCURRENT_LAUNCHES` | 8 (default) | Concurrent sandbox launches |
| `CMA_MAX_IDLE` | 300 (default) | Seconds a worker waits idle before it finishes the session's sandbox |
| `ANTHROPIC_BASE_URL_OVERRIDE` | unset | Alternative Anthropic API URL; must be `https` |

## Operate

| Route | Purpose |
|---|---|
| `GET /` | Status page (counters only) |
| `GET /api/status` | The same as JSON, plus this deployment's sandboxes by state |
| `GET /api/health` | 200 while the poll loop runs; 503 after three consecutive poll-loop crashes. The Apps proxy answers `/healthz` itself, so that path never reaches the App |

- **Logs:** `databricks apps logs cma-dispatcher-<target>`. Claims and launches are logged with session and work
  ids; tokens and request headers never are.
- **Rotate the environment key:** generate a new key in the Console, run `deploy.sh … --store-secrets`. Running
  workers keep their session tokens; new launches use the new key.
- **Rotate the HMAC key** (`put_secrets.py --rotate-hmac`) only with no sessions running: existing sandbox names stop
  matching.
- **Shutdown and redeploys** wait for launches the App has already claimed (each is an acknowledged turn; cancelling it would drop that turn), so a stop can take as long as the slowest launch, bounded by the platform's termination grace period.
- **Stop serving:** `databricks apps stop cma-dispatcher-<target>`. Sessions on the environment then wait in the
  queue.

## Test

```bash
uv run --group tests pytest tests/unit_tests/test_app.py     # offline
RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 SANDBOX_E2E_APP=cma-dispatcher-dev \
    DATABRICKS_SANDBOX_PROFILE=<profile> uv run pytest tests/integration_tests/test_90_e2e_app_service_mode.py -v -s
```
