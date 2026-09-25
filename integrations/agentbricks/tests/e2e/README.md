# Agent Bricks CLI agent-tool matrix

## MCP registration validation

For a focused check of `ab tools add mcp`, install the current `databricks-agentbricks` wheel and pytest
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
It creates two LangGraph projects (CLI/direct), runs each with `ab dev`, deploys each to
Databricks Apps, and semantically exercises separate Sandbox table and volume reads,
`system.ai.web_search`, a local Python tool, a temporary Unity Catalog function, and a configured
Genie Agent space. It also verifies automatic Apps resources for the temporary Sandbox table and
volume scopes. The result is 24 evidence rows plus deploy-time grant snapshots.

## Run

```bash
cd integrations/agentbricks
uv build --wheel --out-dir /tmp/agentbricks-tooling-dist
uv run python tests/e2e/tool_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --wheel /tmp/agentbricks-tooling-dist/databricks_agentbricks-0.2.0-py3-none-any.whl \
  --output /tmp/agentbricks-tool-matrix-df1 \
  --uc-schema supervisor_agent.mason_agent_tools_e2e \
  --genie-space-id "$AGENTBRICKS_E2E_GENIE_SPACE_ID" \
  --commit-sha <pushed-40-character-sha> \
  --source-root /absolute/path/to/databricks-ai-bridge \
  --template-repo /absolute/path/to/databricks-ai-bridge \
  --template-ref your-feature-branch
```

The profile must identify a workspace with Databricks Apps, `system.ai.sandbox`,
`system.ai.web_search`, a 32-character Genie Space ID, and permission to create a schema, functions,
a table, and a volume.
Pass the space with `--genie-space-id` or `AGENTBRICKS_E2E_GENIE_SPACE_ID`. The suite discovers and starts
a SQL warehouse. Override its defaults with `--warehouse-id` or `--uc-schema catalog.schema`.
Deployed Databricks Apps accept programmatic calls under `/api/*` with OAuth Bearer tokens. If the
workspace profile uses a PAT, pass an OAuth profile for the same workspace with
`--app-auth-profile`.
`--source-root` ties the claimed commit to the checkout's HEAD and byte-compares the changed Agent
Bricks modules in the wheel against that checkout. The template repo/ref flags make `ab init` read
an explicit template source; provide both or omit both to use the verified installed wheel template.

Direct authoring does not call `ab tools add`: it replaces `agent.toml` with
`fixtures/direct_agent.toml`. CLI authoring invokes four managed `ab tools add ...` commands.
Both paths then create the same user-owned, framework-native Python tool file with no Python entry
in `agent.toml`. Every exact command and code-authoring step is captured in `commands.log`.

The CLI path first verifies that an unavailable MCP service is rejected without changing
`agent.toml`, then checks that removing the absent binding is harmless. The subsequent dev and
deployed tool matrix exercises valid managed tools.

The deployed cases do not pre-grant the temporary UC function. They require `ab deploy` to create
create the function/table/volume/Genie Apps resources, then inspect those permissions before
invoking the App. Built-in `system.ai` MCP services use platform-managed access defaults and are
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

Success is exactly `24 passed, 0 failed, 0 skipped`, two deploy grant snapshots, and one idempotent
repeat deploy. Temporary Apps and both UC functions are deleted after a successful run, with cleanup
results saved in `evidence.json`; App deletion is not considered complete until a follow-up read
confirms absence. Pass `--keep-resources` while debugging.
