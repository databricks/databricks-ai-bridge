# Mason agent-tool matrix

This suite proves that CLI-generated and directly authored Python integration registries reach the
same runtime code. It creates two LangGraph projects (CLI/direct), runs each with `mason dev`, deploys each to
Databricks Apps, and semantically exercises sandbox, `system.ai.web_search`, a local Python tool,
and a temporary Unity Catalog function. The result is 16 evidence rows.

## Run

```bash
cd integrations/mason
uv build --wheel --out-dir /tmp/mason-tooling-dist
uv run python tests/e2e/tool_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --wheel /tmp/mason-tooling-dist/databricks_mason-0.1.1.dev0-py3-none-any.whl \
  --output /tmp/mason-tool-matrix-df1 \
  --uc-schema aifx_benchmarks.mason_agent_tools_e2e \
  --template-repo /absolute/path/to/databricks-ai-bridge \
  --template-ref your-feature-branch
```

The profile must identify a workspace with Databricks Apps, `system.ai.sandbox`,
`system.ai.web_search`, and permission to create a schema/function. The suite discovers and starts
a SQL warehouse. Override its defaults with `--warehouse-id` or `--uc-schema catalog.schema`.
Deployed Databricks Apps accept programmatic calls under `/api/*` with OAuth Bearer tokens. If the
workspace profile uses a PAT, pass an OAuth profile for the same workspace with
`--app-auth-profile`.
The template repo/ref flags make `mason init` read the exact checkout under test and avoid remote
clone throttling; provide both or omit both to test the default upstream template.

Direct authoring does not call `mason tools add`: it replaces `agent/databricks_tools.py` with
`fixtures/direct_databricks_tools.py` and creates the user-owned Python tool file. CLI authoring
invokes all four `mason tools add ...` commands, then implements the generated Python stub. Every
exact command and generated-file step is captured in `commands.log`.

## Verify existing evidence

```bash
uv run python tests/e2e/tool_matrix.py \
  --verify-evidence /tmp/mason-tool-matrix-df1/evidence.json
```

Success is exactly `16 passed, 0 failed, 0 skipped`. Temporary Apps and the UC function are deleted
after a successful run. Pass `--keep-resources` while debugging.
