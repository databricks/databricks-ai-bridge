# Mason agent-tool matrix

This suite proves that CLI edits and direct `agent.toml` edits reach the same runtime code.
It creates four projects (OpenAI/LangGraph × CLI/direct), runs each with `mason dev`, deploys
each to Databricks Apps, and semantically exercises sandbox, `system.ai.web_search`, a local
Python tool, and a temporary Unity Catalog function. The result is 32 evidence rows.

## Run

```bash
cd integrations/mason
uv build --wheel --out-dir /tmp/mason-tooling-dist
uv run python tests/e2e/tool_matrix.py \
  --profile df1 \
  --wheel /tmp/mason-tooling-dist/databricks_mason-0.1.0.dev0-py3-none-any.whl \
  --output /tmp/mason-tool-matrix-df1
```

The profile must identify a workspace with Databricks Apps, `system.ai.sandbox`,
`system.ai.web_search`, and permission to create a schema/function. The suite discovers and starts
a SQL warehouse. Override its defaults with `--warehouse-id` or `--uc-schema catalog.schema`.

Direct authoring does not call `mason tools add`: it replaces `agent.toml` with
`fixtures/direct_agent.toml` and creates the user-owned Python tool file. CLI authoring invokes all
four `mason tools add ...` commands, then implements the generated Python stub. Every exact command
and generated-file step is captured in `commands.log`.

## Verify existing evidence

```bash
uv run python tests/e2e/tool_matrix.py \
  --verify-evidence /tmp/mason-tool-matrix-df1/evidence.json
```

Success is exactly `32 passed, 0 failed, 0 skipped`. Temporary Apps and the UC function are deleted
after a successful run. Pass `--keep-resources` while debugging.

