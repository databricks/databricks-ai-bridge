# Runtime Store live check

Run from `integrations/mason` in the PR593 checkout. The fixture uses the actual Mason `AgentApp`
and `LakebaseDurableRuntimeStore` with an echo callback. No model endpoint, Session Store, Memory Store,
or tracing resources are needed. No live checks are performed by the unit suite.

Requirements: a user-selected Databricks OAuth profile, Databricks Apps creation/deployment access,
Lakebase availability, and the updated Runtime Store service deployed (or LiteSwapped) in that
workspace. The profile must reach the swapped service for all three Runtime Store routes. The
caller creates no Lakebase project, SQL grants, or Apps `CAN_CONNECT_AND_CREATE` resource. The app
uses its own injected SP credentials to initialize its schema/tables and read/write state.

The managed API is internal and undocumented. `runtime_store_id` is a query parameter on POST;
the body is `{"owner":{"app":{"name":"agent-mason-...","service_principal_id":"<client UUID>"}}}`.
Get returns `owner.app` and `storage_backend.lakebase`.

## Prepare locally

```bash
uv sync --group dev --default-index https://pypi-proxy.cloud.databricks.com/simple
uv build --wheel --out-dir /tmp/mason-runtime-store-dist --default-index https://pypi-proxy.cloud.databricks.com/simple
cp -R tests/e2e/fixtures/runtime_store /tmp/mason-runtime-store-app
cp /tmp/mason-runtime-store-dist/databricks_mason-0.1.6.dev0-py3-none-any.whl /tmp/mason-runtime-store-app/
```

Use a fresh source directory for each workspace/app. The copied wheel makes the deployed app use
this checkout's code. The editable local installation above also makes `mason deploy` use the
updated client. `agent.toml` selects the Mason server and disables tracing; do not run this fixture
locally as a user, since it must create its schema as the deployed app SP.

## Deploy and verify

Set `MASON_TEST_PROFILE` explicitly to the profile selected for the workspace under test. Ensure
the internal package proxy below is reachable by the Apps build, or pass the approved index for
that build environment.

```bash
MASON_TEST_PROFILE='<selected-profile>'
.venv/bin/mason --profile "$MASON_TEST_PROFILE" deploy runtime-probe-a --source /tmp/mason-runtime-store-app --pip-index-url https://pypi-proxy.cloud.databricks.com/simple
.venv/bin/python tests/e2e/runtime_store_check.py --profile "$MASON_TEST_PROFILE" --app agent-mason-runtime-probe-a --invocation-id 4f1f0e53-71f1-4f90-8900-85fb2aa29231 --output /tmp/runtime-store-before.json
```

The checker asserts the shared project is `databricks-internal-custom-agents`, reads the generated
database name from Get, and queries Postgres through the running app to prove the app SP owns the
database, schema, and both Mason tables and has CREATE capability. It sends a real Mason invocation,
reads back the completed output and persisted event, and records metadata and assertions' evidence
without credentials. Nonzero exit means failure; inspect app logs for startup failures.

Rerun the same deploy command to exercise Create → ALREADY_EXISTS → Get. The physical database
must match the first run. Then restart the process and verify the existing invocation without
sending a new one:

```bash
.venv/bin/mason --profile "$MASON_TEST_PROFILE" deployments stop agent-mason-runtime-probe-a --yes
.venv/bin/mason --profile "$MASON_TEST_PROFILE" deployments start agent-mason-runtime-probe-a
.venv/bin/python tests/e2e/runtime_store_check.py --profile "$MASON_TEST_PROFILE" --app agent-mason-runtime-probe-a --invocation-id 4f1f0e53-71f1-4f90-8900-85fb2aa29231 --read-only --output /tmp/runtime-store-after.json
```

The checker waits up to three minutes for startup connection errors and HTTP 502/503 responses to
clear; compute can become ACTIVE before the app listens. Other errors fail immediately.
Deploy a second copy as `runtime-probe-b`; compare
the two Get results: same project, different `database_id`. Repeat using each explicitly selected
admin/non-admin/no-Lakebase-permissions profile, recording the workspace, caller role, command,
result, and error code. A workspace where Lakebase or the Runtime Store handler is unavailable is
an unsupported-environment result, not a successful permission check.

## Cleanup

```bash
.venv/bin/mason --profile "$MASON_TEST_PROFILE" deployments delete agent-mason-runtime-probe-a --yes
```

The command resolves the app SP, reads and verifies the associated Runtime Store's owner, deletes
the store/database, then deletes the app. Verify Get of the recorded store name returns NOT_FOUND
and the app is gone. A permitted project administrator can also confirm the recorded physical
database is absent. The shared project stays available for other stores. Any cleanup failure
except NOT_FOUND retains the app and reports a retry command. If the app deletion fails after the
store deletion succeeds, retry the same command; the missing store is accepted.

Delete each test deployment through Mason. Direct app deletion bypasses store cleanup; with no
Runtime Store List API yet, automatic discovery of stores left by that path is not implemented.

## Evidence status

The [September 16 run](RUNTIME_STORE_RESULTS_2026-09-16.md) passed through authenticated LiteSwap
test aliases before the rebase onto main, with canonical APIProxy routing still pending. For subsequent runs, record
the exact Universe build/LiteSwap, Mason wheel hash, profiles, commands, JSON evidence, and cleanup
results with the live run. EStore schema compatibility is implemented and tested in Universe;
Mason always consumes the same API contract and requires an exact app SP match. Legacy rows may
omit the app name; any nonempty name must match the deployment. Missing owner/SP metadata fails.
