# App-managed sandbox warm pool

This opt-in prototype moves source copy, dependency installation and handler initialization
before session creation. One unused, prepared sandbox is assigned to each new session.
The resident Python process survives assignment, so its imports and prepared handler are reused.
On assignment, one remote command validates the installed revision and unused worker identity;
the orchestrator reuses that result rather than probing and preparing the handler again.
Session creation wakes the preparation scheduler immediately. The UI receives preparation
snapshots through SSE, with status polling as a fallback, and enables Send on readiness.

## Local CLI setup

From a checkout of `codex/agentbricks-sandbox-warm-pool`:

```bash
uv venv --python 3.12 .venv-sandbox
uv pip install --python .venv-sandbox/bin/python -e 'integrations/agentbricks[sandbox]'
source .venv-sandbox/bin/activate
databricks auth login --host https://e2-dogfood.staging.cloud.databricks.com --profile DEFAULT
agentbricks -p DEFAULT deploy my-agent --sandbox --source /path/to/agent \
  --sandbox-idle-timeout 1800 --sandbox-warm-pool-size 2
```

Use an OpenAI or LangGraph project containing `pyproject.toml`, `agent.toml` and its
sandbox adapter. Deploy generates `uv.lock` if missing and preserves existing locks.
With `--git-revision`, commit the lockfile first. Configure custom package indexes in
the project so resolution and sandbox installation use the same index.
This prototype requires an editable CLI install
so deploy can build the orchestrator wheel. Source installation inside each spare uses
that project's pinned dependencies, including its SDK version; the App ships the worker
control scripts. Changing an adapter inside the installed agent requires updating the
project's dependency pin and republishing it.

Open the returned URL, choose the Sandbox tab, and wait for **2 ready**. New Chat claims
one spare and the panel identifies its pool origin. Send two messages: the second should
reuse that worker and history. Start another session to get a distinct sandbox. The pool
refills in the background. A request arriving with no ready spare prepares cold.

Opening a stopped session resumes its sandbox automatically. If the selected session
became idle while the page was open, returning to the composer resumes it. The readiness
spinner stays visible until its worker is ready. Explicit Stop stays stopped in that view;
failed preparation still requires an explicit retry. The session list shows compute state,
updates from live events and a five-second metadata refresh, and sorts by last activity.

To disable and drain unassigned spares, redeploy the same App/source with
`--sandbox-warm-pool-size 0`. Stop a session through its normal UI control. Assigned
sandboxes never return to the spare pool. Stop/resume preserves their disk, but restarts
the Python worker and pays import costs again.

## Boundaries and cost

- `--sandbox-warm-pool-size` is a deployment-wide maximum for **unassigned** sandboxes,
  not a count of pools or a limit on active sessions. Default 0; prototype range 0–10.
  Preparing, ready and retiring spares all consume capacity, across App replicas.
- Only the default source revision is pooled. Source digest, entrypoint, execution
  identity, worker protocol and idle policy participate in compatibility checks.
- One controller holds a PostgreSQL advisory lock; claims use short row-lock transactions.
  Claimed spares become session mappings atomically and retain their physical worker key.
  A newer deployment generation prevents stale replicas from claiming or replenishing.
- Initialization runs speculatively under the App identity. A custom `prepare_session`
  hook must work without a user request and may run for a sandbox that is never assigned.
  Pool filling submits no agent/model invocation. Request-dependent work remains on the
  first invocation path.
- The controller checks unused spares about every 30 seconds, expires them after one hour,
  and backs off 60 seconds after a preparation failure. Replenishment is serial and can
  be slow when installation is slow. Capacity is a bound, not a readiness guarantee.
- Idle compute is the tradeoff. Assigned sessions consume compute in addition to the pool.
  If the App disappears, Lakebox idle timeout stops its unused sandboxes. This is a
  prototype controller, not a platform-level warm-pool SLA.

## Schema

Two additive tables in the sandbox schema: `warm_pool_config` stores the rollout generation,
compatible revision and capacity/backoff policy; `warm_pool_slots` stores inventory,
readiness/expiry, worker handles and assignment. Existing `sessions.bootstrap_metrics`
JSON holds pool origin and the physical worker key. No existing Runtime Store columns,
constraints or status values change; `invocations` and `invocation_events` are unchanged.

## Verification

Real PostgreSQL tests cover simultaneous reservations/claims, generation fencing,
interrupted preparation, ambiguous create, stale cleanup racing assignment, source/identity
isolation and disabling the pool. Subprocess tests verify that probing never starts a worker
and that preparation is reused by the first turn. Chromium tests check capacity and origin
labels; the existing runtime/streaming/session suite remains applicable.

```bash
cd integrations/agentbricks
AGENTKIT_TEST_POSTGRES_URL=postgresql://USER@localhost:PORT/postgres \
  python -m pytest tests/integration_tests/sandbox_store_test.py -q
python -m pytest tests/unit_tests/sandbox_runtime tests/unit_tests/sandbox_deploy_test.py \
  tests/unit_tests/sandbox_preparation_test.py tests/unit_tests/sandbox_chat_ui_test.py -q
```

The database must be disposable; tests create and drop their own unique schemas. Browser
tests require Playwright and Chromium (`SANDBOX_UI_CHROME` can select its executable).
Live measurements and deployment identity follow.

## Live demo and findings

The [combined App](https://agent-bricks-sb-lean-five-14-6051921418418893.staging.aws.databricksapps.com)
uses two unassigned warm spares and the optimized worker from
[draft #917](https://github.com/databricks/databricks-ai-bridge/pull/917).
Open Sandbox, wait for **2 ready**, then choose New Chat. This is the previously deployed
baseline; the invocation API, reconciliation and capacity changes below have not been redeployed.

Measurements and optimization tradeoffs are maintained in the
[investigation doc](https://docs.google.com/document/d/13X3RCi46JPWUuB74MOdqCbcaJ5wuVu3fjC2e1Y_aKFw/edit?tab=t.0#bookmark=id.wgjb4zzi3iwo).

## Capacity and recovery

The visible pool size is the target for unused prepared spares. It replenishes after assignment,
subject to a deployment-wide cap of 64 running/starting sandboxes, including session compute.
The other defaults are 16 concurrent invocations, 2 installation/preparation operations, and 128
accepted unfinished invocations. Hidden deploy overrides are listed in `cli.md`. Locks coordinate
App replicas; unresolved platform creates keep their reservations. Stopped mappings do not normally
consume capacity. External platform quotas may be lower.

The service reconciler observes owned sandbox mappings every 10 seconds without starting idle
compute. After loss of an App executor, recovery can follow the same remote attempt and retrieve
its journaled result. It never replays arbitrary agent code. A lost sandbox or unconfirmed worker
outcome remains a failed invocation requiring inspection. This is narrower than MAR's complete
sandbox-owned durable queue runner. New HTTP clients use `/api/invocations` and its standard
status/SSE endpoints; the built-in workbench uses its chat projection on those same resources.

The sandbox installer currently selects Python 3.12 because that is the tested stock image.
The prior distribution supports Python >=3.10 and the templates declare >=3.11; ordinary App
startup does not universally pin 3.12. Broader sandbox support needs image/interpreter selection,
lock compatibility validation, and a tested version matrix.
