# Mason agent-tool matrix

This suite proves that CLI edits and direct `agent.toml` edits reach the same runtime code.
It creates two LangGraph projects (CLI/direct), runs each with `mason dev`, deploys each to
Databricks Apps, and semantically exercises sandbox, `system.ai.web_search`, a local Python tool,
and a temporary Unity Catalog function. The result is 16 semantic rows plus four pre-runtime
validation checks.

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
Deployed Databricks Apps accept programmatic calls under `/api/*` with OAuth Bearer tokens. If the
workspace profile uses a PAT, pass an OAuth profile for the same workspace with
`--app-auth-profile`.
The template repo/ref flags make `mason init` read the exact checkout under test and avoid remote
clone throttling; provide both or omit both to test the default upstream template.

Both lanes write the user-owned `agent/tools/matrix_marker.py` directly. Before `mason dev` or
deploy, the CLI-authoring lane proves that undeclared code produces the non-blocking `MASON001`
warning, a temporarily declared missing entry point is a hard failure, the restored valid
declaration passes `mason tools check`, and `mason tools run` returns the deterministic marker. This
lane uses `mason tools add` only for sandbox, MCP, and Unity Catalog function attachments, then
appends the literal Python activation record. Direct authoring replaces the whole manifest with
`fixtures/direct_agent.toml`, including that same record. Both lanes then exercise the custom tool
through dev and deploy.

Every command, stdout/stderr stream, return code, and status for those probes is stored under
`validation_checks` in `evidence.json`. Semantic dev/deploy calls remain under `rows`; successful App
deletes and the UC-function drop are persisted under `cleanup` before the driver can exit zero. All
file and command steps are also captured in `commands.log`.

If a semantic or validation check fails, the driver intentionally retains the Apps and UC function
for debugging and exits nonzero. `--keep-resources` also retains them explicitly; because that omits
required cleanup proof, it is a diagnostic run and does not produce successful evidence.

## Verify existing evidence

```bash
uv run python tests/e2e/tool_matrix.py \
  --verify-evidence /tmp/mason-tool-matrix-df1/evidence.json
```

Success reports `16 passed, 0 failed, 0 skipped` semantic rows,
`validation: 4 passed, 0 failed, 0 skipped`, and successful cleanup for both temporary Apps plus the
UC function. Pass `--keep-resources` while debugging; that intentional retention exits nonzero.
