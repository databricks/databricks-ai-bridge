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
Databricks Apps, and semantically exercises sandbox, `system.ai.web_search`, a local Python tool,
and a temporary Unity Catalog function. The result is 16 evidence rows.

## Run

```bash
cd integrations/agentbricks
uv build --wheel --out-dir /tmp/agentbricks-tooling-dist
uv run python tests/e2e/tool_matrix.py \
  --profile df1 \
  --app-auth-profile df1-oauth-mcp \
  --wheel /tmp/agentbricks-tooling-dist/databricks_agentbricks-0.2.0-py3-none-any.whl \
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
The template repo/ref flags make `ab init` read the exact checkout under test and avoid remote
clone throttling; provide both or omit both to test the default upstream template.

Direct authoring does not call `ab tools add`: it replaces `agent.toml` with
`fixtures/direct_agent.toml`. CLI authoring invokes the three managed `ab tools add ...`
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

## Governed external connection matrix

`connection_matrix.py` is a separate exact eight-cell suite for existing bearer-token UC
Connections:

| Axis | Values |
| --- | --- |
| Setup | existing through `ab auth connections bind` |
| Transport | MCP, HTTP |
| Framework | LangGraph, OpenAI Agents |
| Execution | foreground, background |

The primary matrix holds `principal = "app"` constant. Two Apps group the framework axis; each App
declares both transports and runs foreground and background invocations. Generated native framework
tools call only `from databricks_agentkit.auth import context`. Focused controls additionally exercise
unknown aliases, forbidden authorization headers, and missing request identity.

Both existing UC HTTP Connections must use the `BEARER_TOKEN` credential type. The HTTP provider
receives `GET <http-path>` with `request_id` and `marker` query parameters. The MCP provider receives
a JSON-RPC `tools/call` for `<mcp-tool>` with the same arguments. Each response must contain its
configured provider marker and the expected identity marker.

```bash
cd integrations/agentbricks
uv build --wheel --out-dir /tmp/agentbricks-connection-dist
uv run python tests/e2e/connection_matrix.py \
  --profile <workspace-profile> \
  --app-auth-profile <oauth-profile-on-the-same-workspace> \
  --wheel /tmp/agentbricks-connection-dist/databricks_agentbricks-*.whl \
  --output /tmp/agentbricks-connection-e2e \
  --existing-mcp-connection main.agentbricks_connection_e2e.existing_mcp \
  --existing-http-connection main.agentbricks_connection_e2e.existing_http \
  --mcp-marker AGENTBRICKS_MCP_OK \
  --http-marker AGENTBRICKS_HTTP_OK \
  --user-marker <expected-user-identity>
```

Optional fixture-shape flags are `--mcp-tool` (default `agentbricks_connection_probe`) and
`--http-path` (default `/agentbricks-e2e`). For a provider that exposes both transports through MCP
JSON-RPC, pass `--probe-mode mcp-jsonrpc`, `--mcp-tool <tool>`, and optional
`--mcp-arguments '<json-object>'`. Alias flags customize the local binding names.

The harness copies the wheel into each deployment source so Apps build the exact artifact under
test. It emits one-minute deploy/status ticks, scans subprocess output before persisting it, never
records authorization or provider credentials, atomically writes `evidence.json`, and attempts
cleanup for every candidate App. Existing UC Connections are never mutated or deleted.
`--keep-resources` intentionally makes final verification fail and is only for diagnosis.

Verify already-written evidence without workspace inputs:

```bash
uv run python tests/e2e/connection_matrix.py \
  --verify-evidence /tmp/agentbricks-connection-e2e/evidence.json
```

Success requires eight unique passing rows, the exact four controls (matrix cardinality, unknown
alias, forbidden header, and missing identity), a clean sensitive-data scan, and
every cleanup entry marked `deleted` or `not_found`.
