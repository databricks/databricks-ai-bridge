# Lean sandbox worker prototype

This experimental branch tests a smaller **worker dependency graph**, stacked on
[#915](https://github.com/databricks/databricks-ai-bridge/pull/915) and the lazy
imports from [#910](https://github.com/databricks/databricks-ai-bridge/pull/910).
The deployed App orchestrator continues to use #915. Source copy, locked install,
project build, persistent worker, session isolation, and direct SSE are unchanged.

## Live result (November 14)

[Lean chat App](https://agent-bricks-sb-lean-1114-6051921418418893.staging.aws.databricksapps.com) · [Published source](https://e2-dogfood.staging.cloud.databricks.com/#workspace/Users/james.wu@databricks.com/agentbricks_deployments/agent-bricks-sb-lean-1114/revisions/fa6cbaa68f55c20596d1cd5218f7806e1de1a6e3c88e16282678b084400e2fca/source) ·
[Traces](https://e2-dogfood.staging.cloud.databricks.com/#mlflow/experiments/1476495253353374/traces) ·
[Draft #917](https://github.com/databricks/databricks-ai-bridge/pull/917)

Choose **New Chat**, then use `system.ai.gpt-5-mini` for comparison. The App has
**zero warm spares** and a 600-second sandbox idle timeout, so preparation exposes
the cold-worker cost. Source files are readable by all workspace users. The
user's existing demo App/project are unchanged. The temporary control App and
all benchmark/build sandboxes were stopped; their source and run records remain.

Medians from three fresh sessions per variant on e2-dogfood, ordered
full/lean/lean/full/full/lean. Both variants use identical agent/worker code and
shared dependency versions; this full control is not the exact older demo wheel
set. Provider VM/cache warmth is uncontrolled. These are small-sample observations,
not production latency percentiles.

| Measurement | Full dependencies | Lean worker |
| --- | ---: | ---: |
| Installed distributions | 180 | 101 |
| Dependency installation | 5.10 s | 1.70 s |
| Handler preparation | 14.88 s | **10.38 s (30% lower)** |
| Total session preparation | 25.41 s | **17.36 s (32% lower)** |
| Warm follow-up: client time to first text | 2.58 s | 2.54 s |
| Stop: observed range | 22.6–78.0 s | 21.5–22.6 s |

Handler ranges were 14.75–15.18 s and 10.09–10.49 s. The largest measured change
is tracing import: **4.63 → 0.90 s**. The lean worker still spends ~4.23 s importing
the agent/project framework stack and ~2.80 s in AgentKit imports, including the
Databricks SDK. Those remain targets for a narrower import graph; this change
does not eliminate cold Python initialization. Persistent workers avoid paying
those imports again on follow-up turns.

All **12 real invocations** streamed successfully. Each of six sessions called
`get_current_time`, recalled a synthetic codeword on the second turn using the
managed Session Store, and reused its worker. All 12 exported traces were fetched
and inspected, with root/model spans and tool spans where expected. Both clean
installations passed `uv pip check`; all five unpublished wheel versions were
verified. Lean contains no full MLflow, Skinny, Databricks Agents, Connect, Arrow,
NumPy, or pandas. First-party OpenAI/client imports also passed; live OpenAI
invocation and optional integration compatibility were not tested.

The first control tool turn took 31.7 s to first text; its model spans totaled
~3.1 s, so most delay is outside those spans. Subsequent first-tool turns were
5.1–5.7 s. We do not attribute that outlier to this optimization. Stop times are
also too variable to establish a reliable improvement: nearly all observed stop
time is waiting for the provider to report `stopped`.

[Evidence](evidence/lean-worker-1114.json) records the tested worker commit
`0b9ffc41`, orchestrator commit `307200f2`, wheel/lock/source hashes, deployed
revisions, per-session timings, invocation/trace IDs, inspected spans, package
inventories, and cleanup. No UI source changed in this PR; verification used the
live API and SSE rather than a new browser recording.

## Package boundary

The agent worker uses `mlflow-tracing` for framework autologging and exported
traces. Full MLflow, Databricks Agents (the model-serving package), Spark Connect,
and Arrow are unnecessary for this template's chat, Python tools, and managed
session history. Numerical clients and model-logging dependencies become explicit
extras on the integration packages: `genie`, `vector-search`, `uc`, and
`model-logging`. AgentBricks developer/provisioning commands use the `cli` extra
with MLflow Skinny; framework extras no longer require Databricks Agents.

This is a proposed dependency boundary, **not a release-ready change to existing
package defaults**. It changes installation requirements for existing consumers.
Before release, choose either separate worker distributions or a compatibility
migration for extras; validate all optional integrations and minimum versions.
OpenAI imports are checked, but the live benchmark uses LangGraph. No claim is
made about cold startup for arbitrary custom servers or large user dependencies.

## Reproduce

Use two environments: install the normal #915 CLI separately, and vendor the lean
worker wheels only into the agent project. Deploying with the lean checkout's CLI
would also change the orchestrator and invalidate this comparison.

```sh
# From this experimental checkout, prepare a separate normal CLI.
git fetch origin
git worktree add ../sandbox-orchestrator origin/codex/agentbricks-sandbox-warm-pool
uv venv /tmp/lean-cli --python 3.12
uv pip install --python /tmp/lean-cli/bin/python \
  -e '../sandbox-orchestrator/integrations/agentbricks[sandbox]'
source /tmp/lean-cli/bin/activate
agentbricks -p DEFAULT init /tmp/lean-source --framework langgraph

# Build the unpublished worker wheels from this experimental checkout.
python integrations/agentbricks/scripts/build_lean_worker_demo.py \
  --source /tmp/lean-source --output /tmp/lean-demo --suffix mytest

# Run these where public PyPI is reachable. Do not publish a lock targeting an
# internal package mirror that the sandbox cannot access.
(cd /tmp/lean-demo/full && uv lock --python 3.12)
(cd /tmp/lean-demo/lean && uv lock --python 3.12)

# Use the CLI installed from codex/agentbricks-sandbox-warm-pool (#915).
agentbricks -p DEFAULT deploy sb-full-mytest --sandbox \
  --source /tmp/lean-demo/full --sandbox-warm-pool-size 0 --sandbox-idle-timeout 600
agentbricks -p DEFAULT deploy sb-lean-mytest --sandbox \
  --source /tmp/lean-demo/lean --sandbox-warm-pool-size 0 --sandbox-idle-timeout 600
```

The builder produces ordinary, content-versioned wheels of all five distributions
and declares them as direct project dependencies so `uv.sources` resolves every
unpublished wheel. Both variants use the same wheel files and agent code. The
full control requests the optional heavy packages; the lean variant does not.
Installation uses ordinary `uv sync --locked` and `uv pip check`. There is no
package exclusion or site-packages patch. The normal bootstrap installs the
project wheel after syncing its resolved dependencies.

No CLI options, runtime-store schema, or warm-pool behavior change here. Existing
CLI workflow guidance remains applicable; these unpublished wheel instructions
are specific to the experiment.
