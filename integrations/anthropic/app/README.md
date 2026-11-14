# Dispatcher App: serve Claude Managed Agents sessions from a Databricks App

This Databricks App is the dispatcher for Claude Managed Agents **self-hosted** environments. It runs inside your
workspace, claims each session's work over outbound HTTPS, and starts the session's worker in its own Databricks
Sandbox. Nothing runs on a laptop.

- **User mode (the standard deploy).** Sessions on the **user environment** run **as the Databricks user who signs
  in**: the sandbox, Genie One and Unity Catalog reads use that user's own grants. A user starts a session in the
  Claude Console, signs in once through a link the agent gives them, and from then on the agent's tools run as them.
- **Service mode (always on).** Sessions on the **service environment** run as the App's service principal. Use it
  for automation and shared agents over data every such user may see.

| What | Where |
|---|---|
| Poller, launcher, worker | the `databricks-anthropic` package (`../src`), installed from the wheel that `scripts/build_wheel.sh` builds |
| App code | `app.py` (routes), `service.py` (configuration, restart loop, status) |
| User mode | `user_mode.py` (configuration, identity, the `/connect` routes), `signin.py` (the user-environment watcher and sign-in links), `user_store.py` (Lakebase: bindings, sign-in codes, audit), `static/` (the `/connect` page) |
| Infrastructure | `databricks.yml`: the App (one instance), its secret scope and secret resources; user-mode targets add the user-mode secrets and the Lakebase resource |
| Secrets | `scripts/put_secrets.py`: `env-s-key`, `sandbox-id-hmac-key`, and (user mode) `env-u-key` and `api-key` |
| Deploy | `scripts/deploy.sh` |

## How a user runs a session (user mode)

1. **Start a session** in the Claude Console with an allowed agent, on the **user environment**, and ask a question.
2. **Sign in.** The agent's first tool call returns an error whose text is a one-time link:
   *"Databricks sign-in needed … open `<app>/connect?code=…` … then reply 'done'"*. The agent passes it on. The command
   didn't run: no sandbox exists yet.
3. **Confirm.** The link opens the App. Databricks signs the user in and, the first time, asks for consent to the App's
   scopes. The page shows the session, the agent and who is signed in. **Connect this session** binds the session to
   that user.
4. **Reply "done".** The next tool call starts a sandbox **with the user's credential**. The sandbox, Genie One and UC
   reads are now the user's own.

A later tool call that needs a new sandbox (for example after the session sat idle and its sandbox finished) gets a
new link. Only the session's owner can redeem it.

### What happens in the App

- **One watcher on the user environment** (a normal dispatcher with a routing launcher, `signin.py`). For each
  session's work item:
  - if a **ticket** is waiting for that session (the signed-in user's client, left by `/connect`, in memory only) and
    the session is bound to that same user, it launches the sandbox with the user's client;
  - otherwise it answers the session's pending tool calls itself with a sign-in link and releases the item. No sandbox
    starts. The answer is posted with the work item's own session token: the only credential Anthropic accepts for
    tool results.
- **Identity.** Databricks Apps forwards the user's downscoped token in `x-forwarded-access-token`. `current_user.me()`
  with it gives the user's immutable id. Sessions are bound to that id in Lakebase, never to an email. The token is
  never logged, stored or returned (a digest is cached for 60 s), and a ticket expires before the token does (at most
  30 minutes).
- **Sign-in codes** are random, stored only as SHA-256 hashes, valid for 15 minutes and single-use (redeeming inserts
  the hash into `connect_uses`; the store never updates or deletes anything).
- **Refusals.** `/connect` never takes over a session bound to another user (404), and refuses sessions on another
  environment, agents not in `user_agents`, and sessions with memory stores or vaults.
- **Requests.** `POST /api/connect` needs `X-CMA-Request: 1` and an `Origin` matching the App host, and rejects
  unknown fields. The `/connect` page has a strict CSP and writes text only with `textContent`.

## Identity and isolation

- **User mode:** each user's sandboxes are owned by that user and act with their grants, capped to the App's
  `user_api_scopes` (below). The App's service principal can't list them.
- **Service mode:** every tool call, Genie One question and Unity Catalog read uses the App's service principal.
  Anyone who can create a session on the service environment gets that access, so give the App its own environment
  (ideally its own Anthropic workspace) and grant the principal only data every such user may see.
- **The environment keys stay in the App.** A sandbox only receives its own session's token, which can't read the
  work queue, other sessions or memory stores that aren't attached to the session
  (`tests/integration_tests/test_80_e2e_isolation.py`).
- **Sandbox names are unpredictable** (HMAC of the session id with `sandbox-id-hmac-key`), and an existing sandbox
  is used only if its tags show it was created for that session and environment.
- **App access.** Grant `CAN_USE` to the users or groups who may sign in (user mode) or view the status page.

### Scopes

User mode needs these `user_api_scopes` (found with an on-behalf-of spike, design doc §16 S3): `sandbox`, `genie`,
`ai-gateway`, `sql`, and read-only catalog, schema and table access. Users consent once. The scopes don't widen what a
user may do: their own grants still apply, and the sandbox credential is capped to these scopes (clusters and jobs are
refused).

## Deploy (user mode, the standard deploy)

Prerequisites:
- the Databricks CLI, logged in with a profile that can create Apps and secret scopes; Databricks Sandbox enabled;
- a **Lakebase** (Autoscaling) project branch and database the App may use;
- in the Claude Console, in one workspace: two **self-hosted** environments (**service** and **user**), each with its
  environment key, and an **API key** for the App (a dedicated key is best).

From `integrations/anthropic`:

```bash
# First deploy: stores all four secrets (values read from the shell, never printed)
ANTHROPIC_ENVIRONMENT_KEY=... ANTHROPIC_USER_ENVIRONMENT_KEY=... CMA_ANTHROPIC_API_KEY=... \
  app/scripts/deploy.sh dev <profile> <service environment id> --store-secrets -- \
  --var user_environment_id=<user environment id> --var user_agents=<agent id>[,<agent id>] \
  --var lakebase_branch=projects/<project>/branches/<branch> \
  --var lakebase_database=projects/<project>/branches/<branch>/databases/<database> \
  --var lakebase_endpoint=projects/<project>/branches/<branch>/endpoints/<endpoint>

# Later deploys: the same command without --store-secrets
```

- `deploy.sh` builds the connector wheel, deploys the bundle, stores the secrets (the scope has to exist before the
  App, whose secret resources are checked when it is created) and starts the App. It passes `--profile` explicitly,
  because a `DATABRICKS_CONFIG_PROFILE` in your shell would override the bundle's workspace.
- The App's service principal creates and owns the schema `cma_dispatcher` (tables `bindings`, `audit`,
  `connect_codes`, `connect_uses`) on first start. Deploy the App before anyone runs it locally, so the service
  principal owns the schema.
- The App refuses to start if user mode is missing its environment, keys or agents, or if the user environment is the
  service environment. Without Lakebase it starts, but `/connect` answers 503. Sign-in links use the App's own URL
  (override with `CMA_APP_URL`).
- Check the scope's ACL with `databricks secrets list-acls <scope>`: only the manager (MANAGE) and the App's service
  principal (READ, from its secret resources) should appear. Workspace admins can manage every secret scope; scope
  ACLs can't exclude them.
- **Give users access:** `CAN_USE` on the App, and what the agent needs: Unity Catalog `USE`/`SELECT` on their data,
  `EXECUTE` on Genie One (`system.ai.genie_one_mcp`, securable type `mcp_service`) and Sandbox access.

### Service mode only

Targets `dev-service` and `prod-service` deploy a separate App (`cma-dispatcher-<target>`, with its own secret scope)
that serves only the service environment, with no Lakebase, user environment or API key. Give it its own service
environment: two Apps must not poll the same one.

```bash
ANTHROPIC_ENVIRONMENT_KEY=... app/scripts/deploy.sh dev-service <profile> <service environment id> --store-secrets
```

### Optional settings

Add bundle variables after `--`, for example `--var export_volume=/Volumes/main/agents/outputs` to export each
session's `outputs/` to a Volume, or `--var mcp=` to turn Genie One off.

Once `databricks-anthropic` is on PyPI, `requirements.txt` can pin `databricks-anthropic[mcp]==<version>` instead of
the built wheel.

### Grants for the App's service principal

Service-mode sessions use these. Find the client id with `databricks apps get cma-dispatcher-<target>`, then grant
least privilege, for example:

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
| `environment_id` | (required) | The service environment |
| `secret_scope` | per target | Holds the App's secrets |
| `mcp` | `genie-one` | Serve Genie One from the sandbox; empty to turn it off |
| `export_volume` | empty | UC Volume folder that receives each session's `outputs/` |
| `user_environment_id` | empty | User-mode targets: the user environment; must differ from `environment_id` |
| `user_agents` | empty | User-mode targets: comma-separated agent ids users may run |
| `lakebase_branch`, `lakebase_database`, `lakebase_endpoint` | empty (dev: the dev project) | User-mode targets: the Lakebase branch, database and its read-write endpoint |
| `scope_manager` | the deployer | The only principal with MANAGE on the secret scope (the App gets READ on its keys only) |

### App settings (`config.env` in `databricks.yml`)

| Variable | Set to | Meaning |
|---|---|---|
| `ANTHROPIC_ENVIRONMENT_ID` | `${var.environment_id}` | The service environment |
| `ANTHROPIC_ENVIRONMENT_KEY` | secret resource `env-s-key` | Its environment key; stays in the App |
| `CMA_SANDBOX_ID_KEY` | secret resource `sandbox-id-hmac-key` | HMAC key for sandbox names |
| `CMA_SANDBOX_PREFIX` | `cmaapp-` | Sandbox id prefix for service-mode sandboxes (user mode uses `cmausr-`) |
| `CMA_MCP` | `${var.mcp}` | `genie-one` or empty |
| `CMA_LIFECYCLE` | `ephemeral` | `ephemeral` or `resumable` (service mode; user sandboxes are always ephemeral) |
| `CMA_EXPORT_VOLUME` | `${var.export_volume}` | Optional Volume folder for `outputs/` |
| `CMA_MAX_CONCURRENT_LAUNCHES` | 8 (default) | Concurrent sandbox launches |
| `CMA_MAX_IDLE` | 300 (default) | Seconds a worker waits idle before it finishes the session's sandbox |
| `ANTHROPIC_BASE_URL_OVERRIDE` | unset | Alternative Anthropic API URL; must be `https` |
| `CMA_USER_MODE` | `1` (user-mode targets) | Enables user mode; its routes are absent otherwise |
| `CMA_USER_ENVIRONMENT_ID` | `${var.user_environment_id}` | The user environment |
| `ANTHROPIC_USER_ENVIRONMENT_KEY` | secret resource `env-u-key` | Its environment key; stays in the App |
| `CMA_ANTHROPIC_API_KEY` | secret resource `api-key` | Reads sessions (for `/connect`). Not `ANTHROPIC_API_KEY`, which the SDK would attach to every call |
| `CMA_USER_AGENTS` | `${var.user_agents}` | Agent ids users may run |
| `CMA_MAX_SESSIONS_PER_USER` | `20` | Sessions a user may bind (lifetime) |
| `CMA_APP_URL` | unset | Base URL for sign-in links; by default the App reads its own URL |
| `PGHOST`, `PGDATABASE`, `PGUSER`, ... | injected by the `lakebase` resource | The state store's connection (no password is stored) |
| `LAKEBASE_ENDPOINT` | `${var.lakebase_endpoint}` | The endpoint the App mints its 1-hour Postgres password for (the resource doesn't inject it) |

## Limits and known gaps (this version)

- **The sign-in link costs a round trip.** A session's first tool call is answered with the link instead of running,
  and the user replies after signing in.
- **Forwarded links.** Whoever opens a link supplies the identity. The page warns users to confirm only sessions they
  started; it can't tell who started the session in the Console.
- **Off in user mode:** memory stores (refused) and the resumable lifecycle (user sandboxes are ephemeral).
- **Cleanup.** The service sweep only sees `cmaapp-` sandboxes. A user sandbox is removed by its worker when the session
  goes idle; if a worker crashes, the stopped sandbox remains, owned by the user, until they delete it in Sandbox.
- **Caps.** `CMA_MAX_SESSIONS_PER_USER` counts a user's bindings over their lifetime (bindings are never deleted in this
  version); the check and the bind are not atomic. Sessions carry whatever budget they were created with in the
  Console.
- **Token revocation.** A revoked consent is refused on the next request, but the principal cache can keep serving the
  caller for up to 60 s, and a ticket already left for a session can still launch it until it expires.
- **Restarts** drop tickets (they're in memory): the session's next tool call gets a new link.

## Operate

| Route | Purpose |
|---|---|
| `GET /` | Status page (counters only) |
| `GET /api/status` | The same as JSON, plus this deployment's service-mode sandboxes by state |
| `GET /api/health` | 200 while the poll loop runs; 503 after three consecutive poll-loop crashes. The Apps proxy answers `/healthz` itself, so that path never reaches the App |
| `GET /connect`, `GET /connect.js` | User mode: the sign-in link page (strict CSP) and its script |
| `GET /api/connect?code=` | User mode: what a link is for (consumes nothing) |
| `POST /api/connect` | User mode: redeem a link (bind the session, leave a ticket) |

- **Logs:** `databricks apps logs cma-dispatcher-<target>`. Claims and launches are logged with session and work
  ids; tokens, codes and request headers never are. In user mode, the `audit` table records `signin-link`, `connect`,
  `launch` and refusals per session and user.
- **Rotate an environment key:** generate a new key in the Console, then run `deploy.sh … --store-secrets`. Running
  workers keep their session tokens; new launches use the new key.
- **Rotate the HMAC key** (`put_secrets.py --rotate-hmac`) only with no sessions running: existing sandbox names stop
  matching.
- **Shutdown and redeploys** wait for launches the App has already claimed (each is an acknowledged turn; cancelling
  it would drop that turn), so a stop can take as long as the slowest launch, bounded by the platform's termination
  grace period.
- **Stop serving:** `databricks apps stop cma-dispatcher-<target>`. Sessions on its environments then wait in the
  queue.

## Test

```bash
uv run --group tests pytest tests/unit_tests/test_app.py tests/unit_tests/test_user_mode.py   # offline
RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 SANDBOX_E2E_APP=cma-dispatcher-dev \
    DATABRICKS_SANDBOX_PROFILE=<profile> uv run pytest tests/integration_tests/test_90_e2e_app_service_mode.py -v -s
```
