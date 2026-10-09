# Agent Bricks CLI agent-tool matrix

## Sandbox model roundtrip

`sandbox_roundtrip_test.py` verifies that successful sandbox output and a controlled Python
execution error both reach the next model turn, over JSON and SSE. Use a LangGraph agent with
`system.ai.sandbox` configured and `databricks-langchain==0.21.0` installed:

```bash
RUN_AGENTBRICKS_SANDBOX_E2E=1 AGENTBRICKS_E2E_AGENT_URL=<agent-url> \
  AGENTBRICKS_E2E_OUTPUT=/tmp/sandbox-roundtrip \
  python -m pytest tests/e2e/sandbox_roundtrip_test.py -v -s
```

For a deployed App, supply an OAuth bearer token through `AGENTBRICKS_E2E_APP_TOKEN`; never
record it in test artifacts. Set `AGENTBRICKS_E2E_MODEL` to use a different entitled model.
The test requires one sandbox call, a matching tool-call ID, the expected structured execution
outcome, and a final assistant response containing the unique marker. The original tool message
may still contain LangChain metadata: the provider strips it from the outbound model request,
not from the stored transcript.

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
Databricks Apps, and semantically exercises Sandbox volume reads,
`system.ai.web_search`, a local Python tool, a temporary Unity Catalog function, and a configured
Genie Agent space. It also verifies automatic Apps resources for the temporary Sandbox volume
scope. The result is 20 evidence rows plus deploy-time grant snapshots. Sandbox table scopes are
excluded until Databricks Connect supports table downscoping.

## Run

```bash
cd integrations/agentbricks
uv build --wheel --out-dir /tmp/agentbricks-tooling-dist
uv run python tests/e2e/tool_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --wheel /tmp/agentbricks-tooling-dist/databricks_agentbricks-0.5.0.dev0-py3-none-any.whl \
  --output /tmp/agentbricks-tool-matrix-df1 \
  --uc-schema supervisor_agent.mason_agent_tools_e2e \
  --genie-space-id "$AGENTBRICKS_E2E_GENIE_SPACE_ID" \
  --commit-sha <pushed-40-character-sha> \
  --bridge-sha <pushed-40-character-sha> \
  --source-root /absolute/path/to/databricks-ai-bridge \
  --template-repo /absolute/path/to/databricks-ai-bridge \
  --template-ref your-feature-branch
```

The profile must identify a workspace with Databricks Apps, `system.ai.sandbox`,
`system.ai.web_search`, a 32-character Genie Space ID, and permission to create a schema, functions,
and a volume.
Pass the space with `--genie-space-id` or `AGENTBRICKS_E2E_GENIE_SPACE_ID`. The suite discovers and starts
a SQL warehouse. Override its defaults with `--warehouse-id` or `--uc-schema catalog.schema`.
Deployed Databricks Apps accept programmatic calls under `/api/*` with OAuth Bearer tokens. If the
workspace profile uses a PAT, pass an OAuth profile for the same workspace with
`--app-auth-profile`.
`--source-root` ties the claimed commit to the checkout's HEAD and byte-compares the changed Agent
Bricks modules in the wheel against that checkout. The template repo/ref flags make `agentbricks init`
read the exact checkout under test and avoid remote clone throttling; provide both or omit both to use
the verified installed wheel template.
When `--bridge-sha` is supplied, the generated App pins Agent Bricks and LangChain to that immutable
bridge commit; otherwise it uses the wheel built for this run. Pass `--preprovisioned-app-catalog-access`
when Apps already have catalog access and the runner identity cannot grant `USE CATALOG` itself.
Omit `--profile` and `--app-auth-profile` to use ambient OAuth environment credentials, as the gated
nightly integration test does.

Direct authoring does not call `agentbricks tools add`: it replaces `agent.toml` with
`fixtures/direct_agent.toml`. CLI authoring invokes four managed `agentbricks tools add ...` commands.
Both paths then create the same user-owned, framework-native Python tool file with no Python entry
in `agent.toml`. Every exact command and code-authoring step is captured in `commands.log`.

The CLI path first verifies that an unavailable MCP service is rejected without changing
`agent.toml`, then checks that removing the absent binding is harmless. The subsequent dev and
deployed tool matrix exercises valid managed tools.

The deployed cases do not pre-grant the temporary UC function. They require `agentbricks deploy` to
create the function/volume/Genie Apps resources, then inspect those permissions before
invoking the App. The runner attaches a separate user-managed `sql_warehouse` App resource with
`CAN_USE` for the Genie Space's backing warehouse before invoking Genie, and verifies the resource
survives the CLI-authored repeat deploy; Agent Bricks does not grant this transitive dependency.
Built-in `system.ai` MCP services use platform-managed access defaults and are
validated through live Sandbox and web-search calls rather than direct grant inspection. External
MCP services still receive direct service/catalog/schema grants. The temporary declared function
calls a second, undeclared function: the harness proves Agent Bricks did not
grant that transitive function, applies and verifies its required direct manual grant, and only
then invokes the declared function. The CLI-authored deployment is repeated to prove grant
idempotency and preservation of unrelated App resources.

## Verify existing evidence

```bash
uv run python tests/e2e/tool_matrix.py \
  --verify-evidence /tmp/agentbricks-tool-matrix-df1/evidence.json
```

Success is exactly `20 passed, 0 failed, 0 skipped`, two deploy grant snapshots, and one idempotent
repeat deploy. Temporary Apps, their synced `agentbricks_deployments/<app>` source folders, stores,
Lakebase roles, and UC resources are deleted after every run, passing or failing, with cleanup results
saved in `evidence.json`; App deletion is not considered complete until a follow-up read confirms
absence. Pass `--keep-resources` to retain resources while debugging. The gated nightly test reports bounded dev/deploy log tails and
captures App runtime logs for failed deployed cases.

## Declarative user-auth scope matrix

`auth_scope_matrix.py` deploys four projects covering LangGraph and OpenAI Agents SDK harnesses,
each with either an explicit-only `sql` scope or the union of explicit `sql` plus managed web-search
`ai-gateway` inference. Every project contains a request-bound code-first SQL tool that proves the
invoking user can read a temporary marker while the App principal is denied. Combined cases also
invoke managed web search. The runner verifies configured and effective App scopes, a pushed source
SHA freshness marker in App logs, OAuth invocation status, and resource cleanup.

Build and push the source commit before running because deployed projects pin their runtime to that
exact remote SHA:

```bash
cd integrations/agentbricks
uv build --wheel --out-dir /tmp/agentbricks-auth-scope-dist
uv run python tests/e2e/auth_scope_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --wheel /tmp/agentbricks-auth-scope-dist/databricks_agentbricks-0.5.0.dev0-py3-none-any.whl \
  --output /tmp/agentbricks-auth-scope-matrix \
  --uc-schema aifx_benchmarks.agentbricks_auth_scope_e2e \
  --source-repo https://github.com/databricks/databricks-ai-bridge.git \
  --source-ref <full-pushed-commit-sha>
```

Verify saved evidence without workspace access:

```bash
uv run python tests/e2e/auth_scope_matrix.py \
  --verify-evidence /tmp/agentbricks-auth-scope-matrix/evidence.json
```

Success is exactly `4 passed, 0 failed, 0 skipped` with both cleanup checks true. Credentials and
workspace identifiers are not written to `evidence.json`; detailed local logs remain under the
output directory for diagnosis and report generation.
