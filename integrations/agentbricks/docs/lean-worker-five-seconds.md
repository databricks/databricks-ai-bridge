# Five-second handler preparation

Follow-up to the [lean dependency prototype](lean-worker-prototype.md), November 14,
2026. The target here is **handler preparation**, after sandbox creation, source
copy, and installation. It is not five seconds from New Chat to a ready session.

[Latest chat App](https://agent-bricks-sb-lean-five-14-6051921418418893.staging.aws.databricksapps.com) ·
[Published source](https://e2-dogfood.staging.cloud.databricks.com/#workspace/Users/james.wu@databricks.com/agentbricks_deployments/agent-bricks-sb-lean-five-14/revisions/458f3484aba214f34e9e787b1b62802750c58f05b684febe825630366617acbd/source) ·
[Traces](https://e2-dogfood.staging.cloud.databricks.com/#mlflow/experiments/1476495253550735/traces) ·
[Draft #917](https://github.com/databricks/databricks-ai-bridge/pull/917)

Choose **New Chat** and `system.ai.gpt-5-mini`. The App has **zero warm spares** and
a 600-second idle timeout. Source is readable by all workspace users. The prior
lean App remains available for comparison, and the user's original demo is unchanged.

## What changed

1. **Compile Python during installation.** Both first-party templates set
   `[tool.uv] compile-bytecode = true`; the existing sandbox `uv sync --locked`
   honors this setting. Compilation runs under the actual sandbox interpreter,
   Python 3.12.3. It adds work to installation and avoids compiling imported
   dependencies on the handler's critical path. No custom image is required.
2. **Load MCP only when configured.** The LangGraph template and AgentKit helper
   defer MCP imports until a server is used. An empty manifest still gets
   validated. Configured MCP tools and their authentication behavior remain supported.
3. **Reduce import-cache bookkeeping.** Fresh installs skip the distribution-wide
   RECORD scan. Recording the loaded-module list avoids thousands of `Path`
   allocations. Replacement workers can still use the learned read-ahead manifest;
   read-side path validation and resource limits are unchanged.

The installed set stays at **101 distributions**. This follow-up removes work
from startup without disabling tracing, tools, streaming, or managed history.

## Measurements

Medians from three fresh sessions per variant on e2-dogfood:

| Measurement | Prior lean worker | New candidate |
| --- | ---: | ---: |
| Dependency installation (includes compilation) | 1.67 s | 5.89 s |
| Handler preparation | 10.19 s | 4.57 s |
| Total session preparation (server) | 17.11 s | 15.59 s |
| Session preparation observed by client | 18.93 s | 17.52 s |
| Warm follow-up: client time to first text | 2.42 s | 2.34 s |
| Stop: client wall time | 22.65 s | 22.67 s |

Candidate handler range: **4.57–4.61 s**, versus 9.97–10.31 s baseline.
Candidate stops ranged from 21.45 to 75.86 s, largely waiting for the provider to
report stopped; no shutdown improvement is established.

The first candidate tool invocation took 29.72 s to first text and 41.27 s to
completion after preparation. Its worker preparation was 70 ms and model spans
totaled 2.58 s; most delay was inside graph execution outside the model spans.
The cause is unresolved. This outlier is retained in the evidence; faster handler
startup does not guarantee faster first responses.

The earlier full-versus-lean results (14.88 → 10.38 s handler) are a separate
experiment. This comparison starts from the already-lean worker. Both Apps use
the normal #915 orchestrator, the same model and dependency versions, and fresh
agent environments. Provider VM/cache state is uncontrolled. Three observations
per variant do not establish a latency percentile or a five-second guarantee.

Most of the handler gain is **work moved into installation**. Total readiness
improves much less. Prepared warm spares can pay both costs before a user arrives;
without a ready spare, New Chat still waits for installation and initialization.

Direct-hook probes helped attribute the result: compile-only took 4.72 s, and the
combined candidate took 3.98 s. These each have one sample and exclude App/process
transport overhead; they must not be compared directly to the App handler metric.
Import-cache bookkeeping fell from about 460 ms to 8 ms. Skipping unconfigured MCP
saved about 0.3 s with compiled dependencies. Compile-only and combined probes
are exploratory; the final App comparison pins every shared dependency version.

## Remaining work

The compiled worker still spends about **1.3 s in AgentKit/Databricks SDK imports**
and **1.7 s importing the project/chat framework**. The SDK eagerly imports and
constructs generated service clients; narrowing our import path still executes
its package initializer. OpenAI's resource package similarly loads unrelated
resources. Supported lazy initialization in those upstream packages is the next
structural opportunity. These costs are not all removable, and their savings
have not been measured.

Splitting out more installed-but-unused CLI/server packages may simplify the
worker distribution, but will not save import time unless they are on the loaded
graph. Arbitrary custom agents still pay for their own initialization. The
existing prepared-worker pool addresses that user-visible wait; bytecode alone
does not preserve initialized classes, framework state, or clients.

## Validation and identity

The tested worker implementation is `2ded21b4a9c358fc8112a57bd9001e907e25a240`;
the unchanged orchestrator is `307200f2e55819d51c23fa6138ae90886dacce67`, version
`0.10.0.dev0+sandbox.a1e2b00761e032bb`. The source bundle is
`86604f7e3bf3cf7c3c12c36744a7a32b2209aaa100ecb69ce35262e1a5fcbde4`.

Focused MCP/import-cache/adapter tests: **76 passed, 15 skipped**. A clean sandbox
install passed `uv pip check` and a real loopback HTTP MCP discovery/tool call
(`19 + 23 = 42`), including a check that unconfigured MCP was not imported.

All **12 final-comparison invocations** streamed successfully: six sessions
called a real Python tool, recalled prior-turn history, and reused the same worker.
All 12 exported traces were fetched and inspected, including successful root,
model, and tool spans. Every measured sandbox and all direct probe/build sandboxes
were confirmed stopped.

The [evidence record](evidence/lean-worker-five-seconds-1114.json) includes exact
wheels, lock hashes, per-session timings, inspected trace spans, and cleanup.
Initial candidate measurements used two newer LangGraph patch versions; those
are recorded separately as exploratory results, excluded from the final comparison.
The dependency-contract migration and optional integration/minimum-version matrix
remain unvalidated. Live invocation uses LangGraph; OpenAI has import checks only.
No UI source changed, so verification uses the live API and SSE, not a new browser
recording. No runtime-store schema or CLI interface changed.

## Reproduce

Use the [normal #915 CLI setup](lean-worker-prototype.md#reproduce) separately from
the experimental worker checkout. From the latter:

```sh
agentbricks -p DEFAULT init /tmp/five-source --framework langgraph
# The normal CLI scaffolds its own template; copy the updated lazy MCP helper.
cp integrations/agentbricks/src/databricks_agentbricks/templates/agent-langgraph/agent/mcps.py \
  /tmp/five-source/agent/mcps.py
python integrations/agentbricks/scripts/build_lean_worker_demo.py \
  --source /tmp/five-source --output /tmp/five-demo --suffix five-demo \
  --compile-bytecode

# Run where public PyPI is reachable, including a temporary build sandbox.
(cd /tmp/five-demo/lean && uv lock --python 3.12)
agentbricks -p DEFAULT deploy sb-lean-five-demo --sandbox \
  --source /tmp/five-demo/lean --sandbox-warm-pool-size 0 --sandbox-idle-timeout 600
```

For a matched benchmark, seed the candidate's `uv.lock` with the baseline lock
**before** resolving the changed wheel requirements; verify the installed package
maps afterward. Otherwise new transitive releases can silently change the comparison.
Here every installed version matched except the content-versioned AgentBricks wheel.
Reusing the published source and its lock reproduces the final dependency set.
