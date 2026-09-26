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
  --wheel /tmp/agentbricks-auth-scope-dist/databricks_agentbricks-0.3.0-py3-none-any.whl \
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
