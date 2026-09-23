# Mason agent-tool matrix

## MCP registration validation

For a focused check of `mason tools add mcp`, install the Mason wheel and pytest into a virtual
environment, then run:

```bash
RUN_MASON_MCP_E2E=1 MASON_E2E_PROFILE=<profile> \
  python -m pytest tests/e2e/mcp_validation_test.py -v -s
```

This runs the installed CLI against a real workspace for both LangGraph and OpenAI projects.
It verifies that missing services fail without changing project files, valid services are added,
duplicate adds are idempotent, and listing/removing bindings still works. The default valid service
is `system.ai.web_search`; override it with `MASON_E2E_MCP_SERVICE`. Only service metadata is read
remotely; no tools are invoked and no workspace resources are created.

## Full runtime matrix

This suite proves that CLI edits and direct `agent.toml` edits reach the same runtime code.
It creates two LangGraph projects (CLI/direct), runs each with `mason dev`, deploys each to
Databricks Apps, and semantically exercises sandbox, `system.ai.web_search`, a local Python tool,
and a temporary Unity Catalog function. The result is 16 evidence rows.

## Run

```bash
cd integrations/mason
uv build --wheel --out-dir /tmp/mason-tooling-dist
uv run python tests/e2e/tool_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --wheel /tmp/mason-tooling-dist/databricks_mason-0.1.0.dev0-py3-none-any.whl \
  --output /tmp/mason-tool-matrix-df1 \
  --uc-schema aifx_benchmarks.mason_agent_tools_e2e \
  --template-repo /absolute/path/to/databricks-ai-bridge \
  --template-ref your-feature-branch
```

The profile must identify a workspace with Databricks Apps, `system.ai.sandbox`,
`system.ai.web_search`, and permission to create a schema/function. The suite discovers and starts
a SQL warehouse. Override its defaults with `--warehouse-id` or `--uc-schema catalog.schema`.
By default, the runner grants each created App service principal `USE CATALOG`, `USE SCHEMA`, and
function `EXECUTE`; the runner identity therefore needs permission to make those grants. Pass
`--preprovisioned-app-catalog-access` only when App identities already receive `USE CATALOG` from
workspace provisioning; schema and function grants remain explicit.
Deployed Databricks Apps accept programmatic calls under `/api/*` with OAuth Bearer tokens. If the
workspace profile uses a PAT, pass an OAuth profile for the same workspace with
`--app-auth-profile`.
The template repo/ref flags make `mason init` read the exact checkout under test and avoid remote
clone throttling; provide both or omit both to test the default upstream template.

Omit `--profile` (and `--app-auth-profile`) to authenticate from ambient Databricks environment
credentials instead of a CLI profile — e.g. a service principal via `DATABRICKS_HOST` /
`DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`. Because those are OAuth, one identity covers
both deploys and the deployed App's `/api/*` calls. This is how the gated nightly integration test
(`tests/integration_tests/test_tool_matrix.py`, enabled by `RUN_MASON_INTEGRATION_TESTS=1`) drives
this suite.

Direct authoring does not call `mason tools add`: it replaces `agent.toml` with
`fixtures/direct_agent.toml`. CLI authoring invokes the three managed `mason tools add ...`
commands. Both paths then create the same user-owned, framework-native Python tool file with no
Python entry in `agent.toml`. Every exact command and code-authoring step is captured in
`commands.log`.

The CLI path first verifies that an unavailable MCP service is rejected without changing
`agent.toml`, then checks that removing the absent binding is harmless. The subsequent dev and
deployed tool matrix exercises valid managed tools.

## Verify existing evidence

```bash
uv run python tests/e2e/tool_matrix.py \
  --verify-evidence /tmp/mason-tool-matrix-df1/evidence.json
```

Success is exactly `16 passed, 0 failed, 0 skipped`. Temporary Apps and the UC function are deleted
after every run, including failures. Pass `--keep-resources` to retain them while debugging.
