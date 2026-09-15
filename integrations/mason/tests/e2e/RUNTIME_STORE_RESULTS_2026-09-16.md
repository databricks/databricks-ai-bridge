# Runtime Store E2E — September 16, 2026

This records the live run **before the rebase onto main**, using consumer commit `11fedce5c8da52aa0967a672cd8c795ee7f3935b`.
The current fixture follows main's `AgentApp(runtime_store=...)`, `@app.recover`, `agent-mason-`
deployment names, and `invocations`/`invocation_events` tables. This historical result is not a
claim that live testing was repeated after the rebase; the PR description records current validation.

**Passed:** real Mason deployment, Runtime Store Create/Get/Delete, app-SP schema/table creation,
invocation writes/readback, reuse on redeploy, persistence across compute stop/start, and deletion.
**Routing limitation:** canonical `/api/2.0/agents/runtime-stores` routes returned generic 404 in
staging. This run used temporary authenticated APIProxy aliases in the LiteSwap image, so it does
not establish that the canonical routes are deployed or that an unmodified CLI can use them there.

Backend: https://github.com/databricks-eng/universe/pull/2619538, tested commit `3235711acaaa9f`.
Consumer: https://github.com/databricks/databricks-ai-bridge/pull/593, published base
`d31f0c1c3372c8c9798cb788a3237138105cd3d0` plus this PR's contract/lifecycle updates.
The deployed `databricks_mason-0.1.5.dev0-py3-none-any.whl` SHA256 was
`51cd6eca86f93e187eeb89212548efc7f693c4f3ec73ebcc82efc60930405d4d`.
The later checker startup-wait change affects only the external test, not that wheel.

## Environment and scope

- Explicit profile: `eng-ml-inference`; caller was a workspace admin.
- Service: `conversation-store`; context: `staging-aws-us-east-1`.
- LiteSwap: `testenv://liteswap/sm-runtime-0916`.
- Shared project: `databricks-internal-custom-agents`, UID
  `a5c36f44-f69e-4cc3-b978-fd4e5ae36206`, already present before the run.
- Swap startup confirmed live EStore schema `11_0`. Actual v10/v11 EStore compatibility is tested
  locally in Universe; this live run used v11 only.
- SQL ran inside deployed apps using their injected app-SP credentials. There were no manual
  SQL grants or durability `CAN_CONNECT_AND_CREATE` Apps resource attachments.
- This run does not establish non-admin human provisioning, behavior in other workspaces,
  first-time shared-project creation, or negative cross-store authorization guarantees.

## Results

| Check | Result |
| --- | --- |
| App A and C provisioning | Same shared project; different dedicated database IDs |
| SQL identity | `current_user` matched each app SP and the database owner |
| DDL | Database CREATE privilege true; app SP owned Mason schema and both tables |
| Runtime persistence | `executions` and `execution_events` contained completed invocation/output and three events |
| A redeploy | Create returned ALREADY_EXISTS; Get recovered identical backend and owner |
| A restart | Read-only check recovered the original invocation and events after compute stop/start |
| Delete running A/C | Mason deleted store/database before deleting app; both commands succeeded |
| Final cleanup | A/B/C apps, physical databases, and Runtime Store metadata all returned NOT_FOUND |
| Shared infrastructure | Project UID unchanged after deletion; LiteSwap torn down |

Successful apps were `mason-rtstore-0916a` and `mason-rtstore-0916c`. Their databases were:

```text
runtime-mason-rtstore-0916-e677bd66-6c23-43d5-9660-03c53758d478
runtime-mason-rtstore-0916-269a6805-93d0-4067-85b6-b3eb4951d672
```

A's persisted invocation was `a3b9cb11-9457-41d7-a323-50d13d6c0112`; C's was
`705b6056-405d-4e4b-89ae-98bf433897ed`. Both completed with the expected echo output and
`run.started`, the custom marker event, and `run.completed` persisted.

## Commands and transport

The temporary runner at `/tmp/liteswap-runtime-0916-njbEIx/mason_liteswap.py` used the real Mason CLI,
checker, SDK authentication, and retry handling. For Runtime Store requests only, it added
`x-databricks-traffic-id: testenv://liteswap/sm-runtime-0916` and translated the URL prefix from
`/api/2.0/agents/` to `/api/2.0/agent-conversation/`. Temporary Create/Get/Delete aliases in the image
called the same Runtime Store handlers through APIProxy authentication. Neither transport override
nor server alias is part of the production changes.

From `integrations/mason`, the deployment and final verification commands were:

```bash
RUNTIME_TEST_ALIAS=1 .venv/bin/python /tmp/liteswap-runtime-0916-njbEIx/mason_liteswap.py mason --profile eng-ml-inference deploy rtstore-0916a --source /tmp/mason-runtime-store-app-0916 --pip-index-url https://pypi-proxy.cloud.databricks.com/simple
RUNTIME_TEST_ALIAS=1 .venv/bin/python /tmp/liteswap-runtime-0916-njbEIx/mason_liteswap.py check --profile eng-ml-inference --app mason-rtstore-0916a --invocation-id a3b9cb11-9457-41d7-a323-50d13d6c0112 --output /tmp/liteswap-runtime-0916-njbEIx/app-a-proof-before.json
# Repeat the same deploy command, then run the checker with --read-only.
.venv/bin/mason --profile eng-ml-inference deployments stop mason-rtstore-0916a --yes
.venv/bin/mason --profile eng-ml-inference deployments start mason-rtstore-0916a
RUNTIME_TEST_ALIAS=1 .venv/bin/python /tmp/liteswap-runtime-0916-njbEIx/mason_liteswap.py check --profile eng-ml-inference --app mason-rtstore-0916a --invocation-id a3b9cb11-9457-41d7-a323-50d13d6c0112 --read-only --output /tmp/liteswap-runtime-0916-njbEIx/app-a-proof-after-restart.json
RUNTIME_TEST_ALIAS=1 .venv/bin/python /tmp/liteswap-runtime-0916-njbEIx/mason_liteswap.py mason --profile eng-ml-inference deploy rtstore-0916c --source /tmp/mason-runtime-store-app-0916c --pip-index-url https://pypi-proxy.cloud.databricks.com/simple
RUNTIME_TEST_ALIAS=1 .venv/bin/python /tmp/liteswap-runtime-0916-njbEIx/mason_liteswap.py check --profile eng-ml-inference --app mason-rtstore-0916c --invocation-id 705b6056-405d-4e4b-89ae-98bf433897ed --output /tmp/liteswap-runtime-0916-njbEIx/app-c-proof.json
RUNTIME_TEST_ALIAS=1 .venv/bin/python /tmp/liteswap-runtime-0916-njbEIx/mason_liteswap.py mason --profile eng-ml-inference deployments delete mason-rtstore-0916a --yes
RUNTIME_TEST_ALIAS=1 .venv/bin/python /tmp/liteswap-runtime-0916-njbEIx/mason_liteswap.py mason --profile eng-ml-inference deployments delete mason-rtstore-0916c --yes
RUNTIME_TEST_ALIAS=1 .venv/bin/python /tmp/liteswap-runtime-0916-njbEIx/mason_liteswap.py cleanup
```

All final commands succeeded. `cleanup` used authenticated Runtime Get, Apps Get, and Postgres
GetDatabase calls for each recorded resource, then GetProject to check the original project UID.
The normal reproducible procedure, once canonical routes are available, is [RUNTIME_STORE.md](RUNTIME_STORE.md).

## Environment issues encountered

- Apps builds timed out reaching the internal PyPI proxy. The devbox downloaded 67 wheels from
  that same proxy. A test-source-only bootstrap installed them offline with SHA256 checks.
  Wheels above the Apps source-export 10 MiB limit were chunked and reassembled before install.
  This was packaging setup; Mason runtime and its Lakebase credential flow were unchanged.
- App B provisioned a store but stalled building dependencies. Mason cleanup succeeded. Immediate
  same-name Apps recreation then failed before any Runtime Store call; that failed app was also
  cleaned up. A fresh app C completed the second-app check.
- After stop/start, Apps reported ACTIVE before the HTTP listener was ready. The checker now waits
  for startup connection failures and HTTP 502/503 for up to three minutes; other failures propagate.

Local validation: 547/547 Mason unit tests passed; changed Python files passed Ruff check/format and
focused ty checks. Full-package ty was limited by uninstalled optional LangGraph/OpenAI dependencies.
Universe passed 139 cases across eight selected targets plus API/proto/OpenAPI builds, then 136
cases across five targets after syncing merged EStore changes; the LiteSwap image built successfully.

Raw local evidence: `/tmp/liteswap-runtime-0916-njbEIx/`, especially `final-live-proof.json`,
`cleanup-proof.json`, `runtime-api-requests.jsonl`, `swap-final.log`, and `lite-down.log`.
No credentials are included in this document or the proof JSON. Uploaded test source directories
are retained as reproducibility artifacts; no test app/database or swap remains active.
