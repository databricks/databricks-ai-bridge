# Lean worker with managed web search

The [live prototype](https://agent-bricks-sb-lean-five-14-6051921418418893.staging.aws.databricksapps.com)
includes **`system.ai.web_search`**, configured through the normal CLI with
`--auth app`. It uses the sandbox's App workspace identity and the managed MCP
service. Choose **New Chat**, select `system.ai.gpt-5-mini`, and ask it to search
official Databricks documentation and cite a URL.

[Published source](https://e2-dogfood.staging.cloud.databricks.com/#workspace/Users/james.wu@databricks.com/agentbricks_deployments/agent-bricks-sb-lean-five-14/revisions/09cdecbbc8fc88306acb3f0f66e63341f9653dbcb417573434632222e851c2d8/source) ·
[Traces](https://e2-dogfood.staging.cloud.databricks.com/#mlflow/experiments/1476495253550735/traces) ·
[Draft #917](https://github.com/databricks/databricks-ai-bridge/pull/917)

The [prior lean App](https://agent-bricks-sb-lean-1114-6051921418418893.staging.aws.databricksapps.com)
also has web search enabled for the comparison.

The published project differs from the previous compiled lean prototype only in
`agent.toml`, which now declares web search. The lockfile, project code, and all
five worker wheels are identical. Source is readable by all workspace users.

## Benchmark

Three fresh sessions per build, zero warm spares, 600-second idle timeout,
LangGraph, e2-dogfood, November 14, 2026. Each session performs four streamed turns:
first web search, another web search, a Python tool call, and managed-history
recall. Search prompts require official documentation and a returned HTTPS URL.
The benchmark checks successful tool events. Verification additionally fetches
exported traces and inspects the tools' returned URLs.
Both builds use the same prompts, model, and shared dependency versions. The
candidate sessions ran first, followed by the prior lean build with search enabled.
Invocation timers start after explicit session preparation completes.

Medians from three sessions per build, both with web search configured:

| Measurement | Prior lean + search | Compiled lean + search |
| --- | ---: | ---: |
| Dependency installation (includes compilation) | 1.61 s | 5.94 s |
| Handler preparation | 10.18 s | 4.52 s |
| Total session preparation (server) | 17.13 s | 15.42 s |
| First search: graph construction | 0.64 s | 0.98 s |
| Warm search: graph construction | 0.26 s | 0.28 s |
| First search: client time to first text | 12.44 s | 10.92 s |
| Warm search: client time to first text | 10.68 s | 11.20 s |
| First search: search-tool time per turn | 7.21 s | 5.33 s |
| Warm search: search-tool time per turn | 6.58 s | 6.52 s |
| History recall: client time to first text | 2.68 s | 2.99 s |
| Stop: client wall time | 21.53 s | 22.55 s |

Candidate handler range: **4.50–4.53 s**. Handler preparation plus first graph
construction totals a median **5.51 s** (prior lean: 10.82 s). The five-second
handler metric leaves additional first-use setup on the invocation path.

Candidate first searches ranged from 10.69 to 18.00 s to first text; warm searches
ranged from 8.93 to 11.51 s. The model can issue multiple searches in a turn.
Search-tool time sums tool spans within each turn; all calls and ranges are
preserved in the evidence.

Handler preparation imports/configures the project. **In the compiled candidate,
MCP imports and tool discovery run during graph construction on invocation.** Graph construction
includes that work plus the graph itself; its timing does not isolate discovery.
The template currently discovers tools on every invocation, including turns that
do not ultimately call search. Search response time also includes model calls,
search-service latency, and session/checkpoint/streaming overhead.

The earlier [five-second measurements](lean-worker-five-seconds.md) used no
configured MCP server. This comparison enables real search on both builds;
changing remote-service latency and model-selected tool calls prevent attributing
response-time differences solely to startup changes. Three sessions per build do
not establish latency percentiles.
Compilation is still paid during installation; total cold readiness remains
longer than the handler step.

## Reproduce

Start with the [compiled lean source and normal CLI setup](lean-worker-five-seconds.md#reproduce).
Before deploying that project:

```sh
agentbricks -p DEFAULT tools add mcp system.ai.web_search --auth app \
  --source /tmp/five-demo/lean
agentbricks -p DEFAULT deploy sb-lean-five-demo --sandbox \
  --source /tmp/five-demo/lean --sandbox-warm-pool-size 0 --sandbox-idle-timeout 600

# Run from this experimental checkout with the normal CLI environment.
python integrations/agentbricks/scripts/benchmark_sandbox_websearch.py \
  --app-url 'https://<your-deployed-app>' --source /tmp/five-demo/lean \
  --output /tmp/websearch-benchmark --sessions 3
```

The output directory must be new. `benchmark.json` records source/lock/wheel
identity, preparation phases, first text/completion times, graph-build times,
tool events, invocation/trace IDs, and stop results. The script rejects a mismatched
published source, missing search configuration, warm-pool hits, omitted/failed
tool calls, and lost managed history. It stops every session in `finally`.

All **24 invocations** succeeded and streamed text. All **12 search turns**
returned URLs (13 actual searches), with successful MCP tool events and
fetched/inspected MLflow tool spans. Python tools, managed history, and worker
reuse passed in all six sessions. All six sandboxes were stopped.

[Evidence](evidence/lean-worker-websearch-1114.json) records exact source/wheel/lock
identity, prompts, per-turn timings, search-result URLs, inspected spans, and cleanup.
Candidate worker: `2ded21b4`; baseline worker: `0b9ffc41`; benchmark: `156bdecd`;
orchestrator: `307200f2`. Candidate source SHA-256:
`8d20e7615b26faf8e293fe780c217bdfe6994a32fc57b6382e647f992edc2529`.

No dependency or SDK/orchestrator behavior changes were needed to enable search.
The dependency-contract compatibility work described in the earlier report
remains open. Live invocation covers LangGraph; live OpenAI web search and custom
servers are outside this measurement.
