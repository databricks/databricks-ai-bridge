# App-managed sandbox warm pool

This opt-in prototype moves source copy, dependency installation and handler initialization
before session creation. One unused, prepared sandbox is assigned to each new session.
The resident Python process survives assignment, so its imports and prepared handler are reused.

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

Use the existing OpenAI or LangGraph project containing `pyproject.toml`, `uv.lock`,
`agent.toml` and its sandbox adapter. This prototype requires an editable CLI install
so deploy can build the orchestrator wheel. Source installation inside each spare uses
that project's pinned dependencies, including its SDK version; the App ships the worker
control scripts. Changing an adapter inside the installed agent requires updating the
project's dependency pin and republishing it.

Open the returned URL, choose the Sandbox tab, and wait for **2 ready**. New Chat claims
one spare and the panel identifies its pool origin. Send two messages: the second should
reuse that worker and history. Start another session to get a distinct sandbox. The pool
refills in the background. A request arriving with no ready spare prepares cold.

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

## Live result — November 13, 2026

[Warm-pool App](https://agent-bricks-sb-warm-pool-1113-6051921418418893.staging.aws.databricksapps.com)
· [Workspace source](https://e2-dogfood.staging.cloud.databricks.com/#workspace/2718939784140450)
· [Runtime Store](https://e2-dogfood.staging.cloud.databricks.com/api/2.0/agents/runtime-stores/agent-bricks-sb-warm-pool-1113)

Deployed independently of `sb-workbench-1109` on e2-dogfood. The published revision
folder is readable by all workspace users. Pool size 2; sandbox idle timeout 1800 s.

- App deployment: `01f1df9d43831cab86588aa776ea2e78`
- Orchestrator wheel: `0.10.0.dev0+sandbox.f071406d05c8ea71`
- Agent source: `a9c7bf92ca2ff52275a6f3272bf06bf87dd5d2e65411e247df3a131064d7b093`
- Same frozen LangGraph workbench agent, SQLite history and default-experiment tracing;
  model `system.ai.gpt-5-mini`. No custom image or dependency changes.

Three simultaneous session requests consumed two ready spares and exercised one cold fallback:

| Measurement | Two pool hits | One cold fallback |
| --- | --- | --- |
| Internal session preparation | 2.07 / 2.11 s | 27.84 s |
| Caller-observed preparation | 4.49 / 4.52 s | 33.29 s |
| Source copy / install | Reused | 3.77 / 7.08 s |
| Handler prepare RPC | 0.55 / 0.52 s, cached | 14.17 s |
| First text after separate preparation | 3.38 / 3.81 s | 3.76 s |
| Subsequent turn first text | 2.09 / 3.14 s | 2.27 s |

The caller-observed preparation includes HTTP, the preparation queue and one-second status
polling. The cold request also queued behind the first two preparations. These are small
functional samples, not latency percentiles or an isolated model-latency benchmark.

A separate new session invoked directly without a prepare call claimed a spare and streamed
first text in **3.87 s**, completing in 4.17 s. For an assigned session, stop took **21.80 s**;
resume took **15.48 s** internally (18.67 s observed), skipped copy/install, restarted its
worker and retained conversation history. Pooling does not speed up that resume path.

Checks passed: distinct sandboxes and histories; unchanged worker ID from preparation
through two turns; cold fallback; automatic refill; direct first invocation; stop/resume
mapping and history; live Chromium capacity/origin labels with no page errors. Test-assigned
sandboxes were stopped afterward; the demo App retains its two-spare pool for new sessions.
Multi-replica races and policy changes were tested with real PostgreSQL; live validation
used one App replica. Long-running churn, credential renewal and sustained arrival rates
still need a soak test before treating this controller as production-ready.
