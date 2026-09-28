# Agent Bricks CLI agent-tool matrix

## MCP registration validation

For a focused check of `agentbricks tools add mcp`, install the current `databricks-agentbricks` wheel and pytest
into a virtual environment, then run:

```bash
RUN_AGENTBRICKS_MCP_E2E=1 AGENTBRICKS_E2E_PROFILE=<profile> \
  python -m pytest tests/e2e/mcp_validation_test.py -v -s
```

This runs the installed CLI against a real workspace for both LangGraph and OpenAI projects.
It verifies that missing services fail without changing project files, valid services are added,
duplicate adds are idempotent, and listing/removing bindings still works. The default valid service
is `system.ai.web_search`; override it with `AGENTBRICKS_E2E_MCP_SERVICE`. Only service metadata is read
remotely; no tools are invoked and no workspace resources are created.

## Full runtime matrix

This suite proves that CLI edits and direct `agent.toml` edits reach the same runtime code.
It creates two LangGraph projects (CLI/direct), runs each with `agentbricks dev`, deploys each to
Databricks Apps, and semantically exercises sandbox, `system.ai.web_search`, a local Python tool,
and a temporary Unity Catalog function. The result is 16 evidence rows.

## Run

```bash
cd integrations/agentbricks
uv build --wheel --out-dir /tmp/agentbricks-tooling-dist
uv run python tests/e2e/tool_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --wheel /tmp/agentbricks-tooling-dist/databricks_agentbricks-0.3.0-py3-none-any.whl \
  --output /tmp/agentbricks-tool-matrix-df1 \
  --uc-schema aifx_benchmarks.agentbricks_agent_tools_e2e \
  --template-repo /absolute/path/to/databricks-ai-bridge \
  --template-ref your-feature-branch
```

The profile must identify a workspace with Databricks Apps, `system.ai.sandbox`,
`system.ai.web_search`, and permission to create a schema/function. The suite discovers and starts
a SQL warehouse. Override its defaults with `--warehouse-id` or `--uc-schema catalog.schema`.
Deployed Databricks Apps accept programmatic calls under `/api/*` with OAuth Bearer tokens. If the
workspace profile uses a PAT, pass an OAuth profile for the same workspace with
`--app-auth-profile`.
The template repo/ref flags make `agentbricks init` read the exact checkout under test and avoid remote
clone throttling; provide both or omit both to test the default upstream template.

Direct authoring does not call `agentbricks tools add`: it replaces `agent.toml` with
`fixtures/direct_agent.toml`. CLI authoring invokes the three managed `agentbricks tools add ...`
commands. Both paths then create the same user-owned, framework-native Python tool file with no
Python entry in `agent.toml`. Every exact command and code-authoring step is captured in
`commands.log`.

The CLI path first verifies that an unavailable MCP service is rejected without changing
`agent.toml`, then checks that removing the absent binding is harmless. The subsequent dev and
deployed tool matrix exercises valid managed tools.

## Verify existing evidence

```bash
uv run python tests/e2e/tool_matrix.py \
  --verify-evidence /tmp/agentbricks-tool-matrix-df1/evidence.json
```

Success is exactly `16 passed, 0 failed, 0 skipped`. Temporary Apps and the UC function are deleted
after a successful run. Pass `--keep-resources` while debugging.

## Durable session header checks

`runtime_session_headers.py` tests an already deployed, isolated template against its real model
and Lakebase Runtime Store. It does not provision or delete resources. Run it separately for
`lg-api`, `lg-ui`, `oa-api`, and `oa-ui`. Custom servers do not instantiate the durable Runtime
and are outside this queueing suite.

In the isolated deployment copy, install the SDK wheel under test, copy
`runtime_session_instrumentation.py` beside the app entrypoint, and call
`install(app, build={...})` after registering the real invocation and recovery hooks. Build metadata
records the variant, SDK commit, wheel hash, app name, and Lakebase resource coordinates. The build
probe also computes the installed runtime source hashes; no environment variables or credentials
are returned. Do not add this instrumentation to the shipped templates or a shared application.

```bash
uv run python tests/e2e/runtime_session_headers.py \
  --profile YOUR-PROFILE \
  --app-url https://YOUR-ISOLATED-APP.databricksapps.com \
  --variant lg-api \
  --expected-wheel /path/to/exact-deployed-sdk.whl \
  --output /tmp/runtime-session-evidence
```

Every run creates a new directory with credential-redacted HTTP request/response transcripts,
invocation and session IDs, per-case results, and artifact hashes. Requests use the selected
profile's OAuth credentials in memory. `--cases concurrent-fifo,event-replay` runs a bounded subset.
The build check requires all five installed runtime source hashes to match the supplied wheel,
and verifies its declared deployment archive hash. A nonempty provenance field alone is not a pass.

The suite covers four foreground/background and JSON/SSE transport combinations, six concurrent
turns in one session, cross-session overlap, idempotent retries and conflicts, header validation,
and cursor replay. Each request carries `X-Databricks-Session-Id`; cookie/body session identity
is not accepted as a substitute. Persisted start/end events establish FIFO order and nonoverlap.
The probes wrap, rather than replace, real model handlers. Delays are bounded to 45 seconds.

Heartbeat, stale, and scan periods are shortened to 1, 8, and 1 seconds in the copied test app.
Read-only probe routes expose actual `Runtime.get_invocation()`, `get_events()`, and `wait()` calls.
After independently deploying multiple workers against the same Runtime Store, pass
`--require-multi-worker` to require that the FIFO turns themselves execute on at least two worker
boot IDs. Without that flag, a passing run does not establish multi-worker contention. The suite
does not claim browser interaction, crash recovery, or stale-worker write fencing coverage.

Retain the app, Lakebase resources, wheel, and deployment source until PR review and merge, so
reviewers can repeat live runs and query the recorded invocation IDs directly. The runner never
provisions, restarts, scales, or deletes resources.
