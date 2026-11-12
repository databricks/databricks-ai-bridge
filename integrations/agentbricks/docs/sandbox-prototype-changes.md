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
invocation snapshots; the source catalog stays in Workspace.

## Review boundaries and remaining gaps

- CLI publication/deployment: `services/sandbox_deploy.py`, `cli/prepared_deploy.py`
  and the environment inspection/invocation/debug commands.
- App APIs, UI and orchestration: `databricks_agentkit/sandbox/app.py`,
  `orchestrator.py`, `session_client.py`, `preparation.py` and `store.py`.
- Sandbox execution: `bootstrap.py`, `persistent.py`, `runner.py`, `adapters/`;
  file streaming uses ExecuteCommand and timed snapshot publication.
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

## Validation for this draft

- 621 focused tests pass: sandbox APIs, source publication, allocation/reuse,
  resident worker processes, preparation/stop, streaming, offline browser UI,
  CLI output/help, framework scaffolding and workflow guidance.
- Ruff checks/format pass. A wheel was built and checked for the App, static UI,
  command protocol and both framework entrypoints.
- The broader unit suite was interrupted after 3,012 passes. It reported three
  model-upgrade failures with missing optional `gepa`/`litellm` dependencies and
  one terminal-table wrapping failure; all four reproduce without the sandbox changes.
  The sandbox App fixture's duplicate
  monkeypatch teardown error was fixed and its tests passed in the focused rerun.
- These runs use the existing local test environment, not a fresh minimum-version
  dependency matrix. Live Databricks integration and scale tests were not rerun
  for the review branch. The running App was not redeployed by PR publication.
