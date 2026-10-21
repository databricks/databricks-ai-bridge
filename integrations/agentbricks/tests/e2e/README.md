# Agent Bricks CLI live checks

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
The test requires one sandbox call, a matching tool-call ID, a metadata-rich text block with a
generated `id`, the expected structured execution outcome, and a final assistant response
containing the unique marker. The original tool message retains the LangChain metadata: the
provider strips it from the outbound model request, not from the stored transcript.

## OBO Chat History Regression

`chat_history.py` tests a deployed LangGraph or OpenAI Agents app with UI enabled and a
user-authenticated Genie/MCP binding. It submits a prompt in the browser, requires a tool call and
the expected answer, then reloads the page without another invocation. Fixed apps must return
the persisted prompt and response messages and display the saved answer. LangGraph history is
compared with invocation output; OpenAI native session items are compared before and after reload.

```bash
python tests/e2e/chat_history.py \
  --url https://<app>.aws.databricksapps.com \
  --app-auth-profile <workspace-oauth-profile> \
  --framework langgraph \
  --model <gateway-model> --tool genie_get_query_result \
  --prompt '<read-only query against your synthetic fixture>' \
  --answer-marker '<expected fixture value>' \
  --expect restored --output /tmp/history-fixed
```

Deploy the same configuration from the baseline and PR revisions, using immutable SDK pins
as described in [CONTRIBUTING.md](../../CONTRIBUTING.md). For the negative control, run the
identical command against the baseline URL with `--expect auth-error` for LangGraph: the
invocation must still succeed, but reopening must return exactly HTTP 401
`MCP_USER_AUTHORIZATION_MISSING`. For OpenAI Agents, use `--framework openai` and
`--expect empty-history`: the invocation must succeed, but reopening returns HTTP 200 with an
empty transcript because baseline history reads the public rather than user-scoped session ID.
The runner never supplies forwarded-user headers or uses a PAT fallback. Install Playwright
and Chrome in the test environment; screenshots and credential-free JSON evidence are saved
under `--output`.

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
