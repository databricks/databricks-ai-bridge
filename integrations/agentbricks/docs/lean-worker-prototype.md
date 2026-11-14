# Lean worker and managed warm-pool demo

This prototype combines a smaller worker dependency graph, install-time Python
bytecode compilation, and the App-managed warm pool from
[#915](https://github.com/databricks/databricks-ai-bridge/pull/915). It includes real
managed MCP web search, Python tools, managed session history, tracing, and direct
SSE streaming. The App orchestrator uses the normal #915 dependencies; only agent
workers use the experimental wheels from this branch.

[Live App](https://agent-bricks-sb-lean-five-14-6051921418418893.staging.aws.databricksapps.com) ·
[Published source](https://e2-dogfood.staging.cloud.databricks.com/#workspace/Users/james.wu@databricks.com/agentbricks_deployments/agent-bricks-sb-lean-five-14/revisions/09cdecbbc8fc88306acb3f0f66e63341f9653dbcb417573434632222e851c2d8/source) ·
[Traces](https://e2-dogfood.staging.cloud.databricks.com/#mlflow/experiments/1476495253550735/traces) ·
[Optimization findings](https://docs.google.com/document/d/13X3RCi46JPWUuB74MOdqCbcaJ5wuVu3fjC2e1Y_aKFw/edit?tab=t.0#bookmark=id.wgjb4zzi3iwo)

## Try the combined App

Open the **Sandbox** tab and wait for **2 ready** warm spares. Choose **New Chat**;
the panel shows the pool allocation, preparation breakdown, and startup duration.
Select `system.ai.gpt-5-mini` and ask: "Use web search to find the official
Databricks documentation for Model Context Protocol. Return a title and URL."
Follow up in the same chat to reuse its resident worker. **Events** and trace
links show tool calls. **Stop session** ends that session's sandbox.

Pool size is the target number of **unassigned** prepared sandboxes, not a limit
on active sessions. Each session gets its own sandbox; a claimed worker survives
the assignment, and the controller replenishes the pool. Assigned sandboxes never
return to the pool. A miss still pays cold creation, installation, and imports.
The deployed sandbox idle timeout is 600 seconds.

The pool pays dependency installation and handler preparation before assignment.
First graph construction still discovers configured MCP tools on invocation;
model/search latency remains on the request path. Compilation adds installation
work. Measurements and tradeoffs are maintained in the linked Google Doc.

## Build a local CLI and worker source

Use Python 3.12, `uv`, and an authenticated Databricks CLI profile. From a checkout
of `codex/agentbricks-lean-worker-prototype`:

```sh
# Keep the normal orchestrator CLI separate from the experimental worker wheels.
git fetch origin
git worktree add ../sandbox-orchestrator origin/codex/agentbricks-sandbox-warm-pool
uv venv /tmp/lean-cli --python 3.12
uv pip install --python /tmp/lean-cli/bin/python \
  -e '../sandbox-orchestrator/integrations/agentbricks[sandbox]'
source /tmp/lean-cli/bin/activate

agentbricks -p DEFAULT init /tmp/lean-source --framework langgraph
cp integrations/agentbricks/src/databricks_agentbricks/templates/agent-langgraph/agent/mcps.py \
  /tmp/lean-source/agent/mcps.py
python integrations/agentbricks/scripts/build_lean_worker_demo.py \
  --source /tmp/lean-source --output /tmp/lean-demo --suffix mydemo \
  --compile-bytecode

agentbricks -p DEFAULT tools add mcp system.ai.web_search --auth app \
  --source /tmp/lean-demo/lean
(cd /tmp/lean-demo/lean && uv lock --python 3.12)

agentbricks -p DEFAULT deploy sb-lean-mydemo --sandbox \
  --source /tmp/lean-demo/lean \
  --sandbox-warm-pool-size 2 --sandbox-idle-timeout 600
```

Run lock resolution where public PyPI is reachable; the sandbox must be able to
reach the index recorded in `uv.lock`. The builder creates content-versioned
wheels for all five distributions, vendors them into the project, and declares
normal `uv.sources` requirements. The sandbox performs `uv sync --locked` and
builds/installs the project under its own Python interpreter. There is no package
exclusion or site-packages patch. The builder also produces a full-dependency
variant for comparisons.

To measure cold preparation separately, deploy a different App with
`--sandbox-warm-pool-size 0`, then use
`scripts/benchmark_sandbox_websearch.py --help`. That benchmark intentionally
rejects warm-pool hits and stops its test sessions. Keep generated output outside
the repository.

## Scope

The worker uses `mlflow-tracing`; provisioning commands retain MLflow Skinny.
Numerical, UC, and model-logging dependencies become optional integration extras.
These dependency-default changes need a compatibility plan before release.
Live validation covers LangGraph; OpenAI has import checks, and arbitrary custom
servers and the optional integration/minimum-version matrix remain unverified.
No new CLI surface or runtime-store schema is introduced by this branch.
