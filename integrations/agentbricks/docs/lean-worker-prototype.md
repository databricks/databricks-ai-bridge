# Lean sandbox worker prototype

This experimental branch tests a smaller **worker dependency graph**, stacked on
[#915](https://github.com/databricks/databricks-ai-bridge/pull/915) and the lazy
imports from [#910](https://github.com/databricks/databricks-ai-bridge/pull/910).
The deployed App orchestrator continues to use #915. Source copy, locked install,
project build, persistent worker, session isolation, and direct SSE are unchanged.

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
# From this branch; SOURCE is an existing scaffolded LangGraph agent.
python integrations/agentbricks/scripts/build_lean_worker_demo.py \
  --source /path/to/agent --output /tmp/lean-demo --suffix mytest

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
