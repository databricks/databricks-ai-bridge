# Sandbox prototype: compatibility and review guide

This draft adds source publication to `agentbricks deploy --sandbox` and places
sandbox orchestration, the App HTTP/SSE server, chat UI, installation, resident
worker, and framework adapters in `databricks_agentkit.sandbox`. The App starts
with `python -m databricks_agentkit.sandbox`; no separate orchestrator repository
or custom image is required. The workspace must enable the sandbox preview and
managed Runtime Store APIs. This is an experimental path, not a released SDK feature.

Start with the [local UV demo instructions](sandbox-demo.md).

## Template compatibility

| Project | Requirements |
| --- | --- |
| OpenAI template from this draft | Use this draft's SDK in the agent's locked dependencies; the generated `runtime/sandbox_entry.py` re-exports its OpenAI adapter. |
| LangGraph template from this draft | Same SDK requirement; the generated graph factory accepts `checkpointer_override` for managed or sandbox-local checkpoints. |
| Previously generated OpenAI project | Add the entrypoint shim and update the SDK lock. Its `agent.agent` and `agent.mcps` functions must match the current template interfaces. |
| Previously generated LangGraph project | Add the shim, update the SDK lock, and add the graph factory's `checkpointer_override` argument and use it when constructing the graph. |
| Custom Python handler | Supply `--entrypoint module:function`; implement the callable contract in the runtime notes. Framework-specific streaming, history and tracing require an adapter. |
| Arbitrary HTTP server or other framework template | Not automatically supported. General server launch, readiness and invocation adapters remain future work. |

An editable CLI does not update the SDK installed inside a source bundle. Pin the
agent's `databricks-agentbricks` dependency to the published prototype commit
(using the `integrations/agentbricks` subdirectory) or a wheel built from it, then
regenerate `uv.lock`. The sandbox executes `uv sync --locked` and requires a
Python 3.12-compatible, buildable project with accessible dependencies. Local
absolute-path dependencies must be replaced with deployable references.

For an OpenAI shim, use:

```python
from databricks_agentkit.sandbox.adapters.openai import invoke  # noqa: F401
```

For LangGraph, import from `databricks_agentkit.sandbox.adapters.langgraph`.
Compare a migrated project with the current template before deployment. This
prototype does not rewrite existing agent source or silently upgrade its lock.

## User journey

1. Initialize an OpenAI or LangGraph project and pin the prototype SDK as above.
2. Configure tools with App authentication and bind the required resources.
3. Run `agentbricks deploy demo --sandbox --source PROJECT --sandbox-idle-timeout 1800`.
   This uploads immutable source/revision metadata, provisions the App-owned Runtime
   Store, grants the App source access and deploys the generic orchestrator.
4. Open the returned chat URL. New Chat allocates one sandbox for the session,
   installs the source, and initializes the persistent worker before enabling Send.
   Direct invocation can perform the same setup lazily.
5. Subsequent turns reuse the worker. Stop terminates that session's sandbox;
   resume reuses retained files but initializes a fresh process.

Tools configured with request-user authentication are rejected by sandbox deploy;
credentials are currently App-owned. The sandbox UI includes readiness phases,
source links, model selection and stop/resume. Lakebase holds session mappings and
shared durable invocation state/events; the source catalog stays in Workspace.

## Review boundaries and remaining gaps

- CLI publication/deployment: `services/sandbox_deploy.py`, `cli/prepared_deploy.py`
  and the environment inspection/invocation/debug commands.
- App APIs, UI and orchestration: `databricks_agentkit/sandbox/app.py`,
  `orchestrator.py`, `session_client.py`, `preparation.py` and `store.py`.
- Sandbox execution: `bootstrap.py`, `persistent.py`, `runner.py`, `adapters/`;
  file streaming uses ExecuteCommand and direct App-to-client SSE. Runtime events store
  one attempt-file descriptor plus lifecycle updates; tokens bypass SQL.
- The separate import/MLflow dependency experiment is not required by this runtime.
  Earlier standalone examples and internal measurements are archived separately.

The draft retains compatibility paths for earlier manifests and builder artifacts;
new deployments use per-session source installation. Remaining work includes generic
custom-server support, durable recovery after sandbox-disk loss, approval/resume,
request-user authorization, multi-tenant isolation review, connection/write-volume
load tests, and release dependency planning. Legacy builder paths are experimental;
this draft does not claim improved cold-start latency or production readiness.

See [runtime notes](../src/databricks_agentkit/sandbox/README.md) for the detailed
worker, streaming, readiness and debugging contracts.

## Runtime Store deduplication

Sandbox invocations now use the existing `Runtime`, durable executor and
`LakebaseDurableRuntimeStore`. Sandbox dispatch is an execution callback;
`SessionSandboxClient` still owns compute. Chat status/history are read projections.

There are **no changes to existing Runtime Store table columns, indexes, constraints,
or status values**. Sandbox request metadata uses `request` JSON; progress, timing and
attempt-file descriptors use `invocation_events.event` JSON. Additive code APIs provide history listing and
allow the Apps connection factory to receive the existing recovery affinity option.

The old deployment-schema `invocations` table remains untouched and unused. Its
historical runs are not migrated; `sessions` mappings and preparation state are retained.
Deploy initializes the standard runtime schema when absent. Failed runs stay terminal;
explicit retry creates a new invocation, while reconnect preserves identity. Recovery
claims stale work but does not automatically replay sandbox code or side effects.

The SSE adapter relays attempt-file text directly to clients. It checks durable state
and lifecycle events once per second; text delivery does not wait for those checks.
This needs load testing and is not a claim of improved latency.

## Validation for this draft

The initial draft passed 621 focused checks and a wheel build. The deduplication
follow-up is checked with runtime, sandbox, browser and real local PostgreSQL tests;
see the PR for the executed results. These are local checks, without a fresh live
Databricks deployment or a minimum-version dependency matrix.

## Direct streaming validation (Nov 13)

The workbench uses `sandbox file → ExecuteCommand(tail -F) → App → SSE` for
live text. Invocation claims, lifecycle events and final results stay in the
shared Runtime Store. One attempt-file descriptor replaces per-token events;
existing table schemas are unchanged. The [diagram](diagrams/sandbox-direct-streaming.png)
has [editable SVG source](diagrams/sandbox-direct-streaming.svg). The
[source overview](diagrams/sandbox-source-overview.png) also reflects lifecycle/final
results in Lakebase, with [editable SVG](diagrams/sandbox-source-overview.svg).

Focused unit and offline Chromium checks cover replay, subscriber cancellation,
attempt changes, bounded preview buffering, SQL-free live delivery, CLI SSE and
browser reconnect/fallback. Existing sandbox/adapter checks also passed.

Live validation on `agent-bricks-sb-workbench-1109`, deployment
`01f1df28e9221a4ab309692960f38bad`, runtime wheel
`0.10.0.dev0+sandbox.1e00de4664dd487e`, with pinned source revision
`95ebfb0fdf8723d9` and `system.ai.gpt-5-mini`:

- Prepared a new dedicated sandbox; disconnected during a response and rebuilt
  the preview through a second SSE subscription to the same run.
- Observed live text while the durable status endpoint contained no preview;
  completed result matched the SSE final result. Unit checks verify no token
  events are written. Repeating the submission returned the original run.
- CLI `--stream` delivered the next response on the same resident worker, with
  conversation memory retained. Stop was confirmed; completed SSE replay still
  returned the answer with the sandbox stopped.

One response delivered 20 snapshots after reconnect, with 0.38 s between the
last changed preview and final status. This is a functional spot check, not a
controlled latency comparison or a browser rendering benchmark. Full MAR
StreamItems/Session Store commit-discard, subscriber fan-out and file retention
remain outside this prototype.

Current rollout: `01f1df29996b1c868c474280ddfb5d63`, runtime wheel
`0.10.0.dev0+sandbox.b1e30159b428a98c`, includes the final runtime documentation
correction. Executable Python/JavaScript is unchanged from the live checks above.

The subsequent [matched browser latency investigation](sandbox-streaming-latency.md)
measured App gRPC-to-SSE delivery at 68 → 13 ms and approximate runner-to-DOM
delivery at 166 → 91 ms. Warm first text remained about 2.6 s in five prompt pairs.
Model output already had roughly half-second gaps before runner writes; stream
discovery and final-status polling also remain. The report includes the small-sample
limits, fixed-rate controls, completion outlier and next steps. Its diagnostic App
and test sandboxes were stopped; the main workbench was unchanged.
