# Integration Tests

## Motivation

`databricks-ai-bridge` is a bridge library -- it connects upstream SDKs (LangChain, OpenAI, etc.) to Databricks platform services (Vector Search, Genie, MCP, Lakebase, FMAPI, Model Serving). The bridge layer handles auth forwarding, parameter translation, response transformation, and API compatibility. Unit tests with mocks verify logic in isolation, but they cannot catch the regressions that matter most: a real API response changing shape, auth credentials failing to propagate, or a new model silently breaking tool calling.

These integration tests exist to catch exactly those failures. Every test hits live Databricks APIs. There are no mocks. Tests are designed to replicate real user patterns from the [app-templates](https://github.com/databricks/app-templates) repo -- the same agent code that customers deploy.

> **See also:** [`.claude/skills/integration-tests.md`](../.claude/skills/integration-tests.md) -- a Claude Code skill that codifies the principles and patterns for *writing* integration tests in this repo. This document covers what tests exist and why; the skill covers how to add new ones.

## Architecture

### Two-Repo Model

Tests live in this public repo (`databricks/databricks-ai-bridge`). CI orchestration and secrets live in a separate private repo (`databricks-eng/ai-oss-integration-tests-runner`). The runner repo contains no test code -- it checks out this repo, injects secrets as environment variables, and runs pytest.

This separation keeps workspace credentials out of the public repo while letting anyone read and contribute to the tests.

### Skip Gates

Every test suite is gated by an environment variable. When the variable is unset, the entire suite is skipped via a module-level `pytestmark`:

```python
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_VS_INTEGRATION_TESTS") != "1",
    reason="RUN_VS_INTEGRATION_TESTS is not set",
)
```

This lets integration test files coexist in the public repo without running during normal `pytest` invocations or PR CI.

### CI Jobs

The runner executes 7 parallel jobs nightly (and on-demand via `workflow_dispatch`), plus a weekly maintenance job:

| Job | Timeout | Gate Variable |
|-----|---------|---------------|
| `langchain-tests` | 20 min | *(always runs)* |
| `vector-search-tests` | 20 min | `RUN_VS_INTEGRATION_TESTS` |
| `genie-tests` | 20 min | `RUN_GENIE_INTEGRATION_TESTS` |
| `mcp-tests` | 20 min | `RUN_MCP_INTEGRATION_TESTS` |
| `fmapi-tool-calling-tests` | 60 min | `RUN_FMAPI_TOOL_CALLING_TESTS` |
| `lakebase-tests` | 20 min | `LAKEBASE_INTEGRATION_TESTS` |
| `obo-credential-tests` | 30 min | `RUN_OBO_INTEGRATION_TESTS` |
| `obo-redeploy-serving` *(weekly)* | -- | *(separate workflow)* |

If any job fails, an alert fires identifying which jobs broke and linking to the run.

### Triggering CI from a PR Branch

The runner checks out `databricks-ai-bridge` at `ref: main` by default. To test a PR branch before merging:

1. Create a branch in the runner repo
2. Change `ref: main` to `ref: <your-bridge-feature-branch>` in `.github/workflows/integration-tests.yml`
3. Push and trigger the workflow (via `gh workflow run` or the Actions tab UI, selecting your runner branch)
4. Delete the temporary runner branch after tests pass

See the [integration-tests skill](../.claude/skills/integration-tests.md) for detailed CLI commands.

---

## Coverage by Area

### 1. Vector Search

**What the bridge provides:** A `VectorSearchRetrieverToolMixin` (shared base), `IndexDetails` and `parse_vector_search_response()` utilities, plus wrapper classes for LangChain (`VectorSearchRetrieverTool`, `DatabricksVectorSearch`) and OpenAI (`VectorSearchRetrieverTool`). The bridge handles auth conversion (CLI profile to PAT, since `VectorSearchClient` doesn't support CLI profile auth), column validation against Unity Catalog schemas, response parsing into framework-native types, and dynamic tool schema generation for LLM function calling.

**Test files:**

| Layer | File | What it tests |
|-------|------|---------------|
| Core | `tests/integration_tests/vector_search/test_bridge_utilities.py` | `IndexDetails` property extraction for both delta-sync and direct-access indexes; `parse_vector_search_response()` with doc_uri remapping, column filtering, score inclusion; `validate_and_get_text_column()` and `validate_and_get_return_columns()` |
| Core | `tests/integration_tests/vector_search/test_vectorsearch_behavior.py` | Smoke tests against the raw VS SDK -- text search on managed embeddings, vector search on direct-access, filter pass-through, column selection. Establishes that the upstream service returns expected response shapes. |
| LangChain | `integrations/langchain/tests/integration_tests/test_langchain_vectorsearch.py` | Tool init (name/description/resources), tool invocation, `DatabricksVectorSearch` similarity search with scores/filters/columns, kwargs pass-through (`score_threshold`, `query_type=HYBRID`), auth paths (PAT, OAuth M2M) |
| OpenAI | `integrations/openai/tests/integration_tests/test_openai_vectorsearch.py` | Tool init (OpenAI `ChatCompletionToolParam` schema generation, `strict` field removal), tool execution, static + dynamic filters, kwargs pass-through, `num_results` override at execute time, auth paths |

**Key regressions these tests guard against:**
- Score values returned as empty strings instead of floats (a real regression that was caught)
- `VectorSearchClient` auth not propagating through the wrapper's credential forwarding logic
- `IndexDetails` misclassifying index types, causing the wrong query path (text vs. vector) to be selected
- OpenAI tool schema containing `strict`/`additionalProperties` fields that break non-GPT models

**Infrastructure:** Two pre-provisioned indexes in `integration_testing.databricks_ai_bridge_vs_test`: `delta_sync_managed` (managed embeddings via `databricks-bge-large-en`) and `direct_access_test` (1024-dim self-managed vectors). Session-scoped fixtures cache index details and search results to minimize API calls.

### 2. Genie

**What the bridge provides:** A core `Genie` class that wraps the Genie REST API (`/api/2.0/genie/spaces/...`) with conversation management, polling orchestration, result parsing (markdown tables or pandas DataFrames), type coercion, token-budget truncation, and MLflow tracing. The LangChain `GenieAgent` factory wraps this as a `RunnableLambda` with `include_context` and `message_processor` options. There is no OpenAI Genie wrapper.

**Test files:**

| Layer | File | What it tests |
|-------|------|---------------|
| Core | `tests/integration_tests/genie/test_genie_behavior.py` | `ask_question()` end-to-end: non-empty result, populated conversation_id, query/description/suggested_questions fields; conversation continuity via follow-up questions; pandas mode (DataFrame output with typed columns) |
| Core | `tests/integration_tests/genie/test_genie_bridge.py` | `_parse_query_result()` output format: markdown table structure (header/separator/data rows), SQL query field population; text-only responses ("Hello" produces text, no SQL); pandas column naming and type conversion; values not all NaN |
| LangChain | `integrations/langchain/tests/integration_tests/test_langchain_genie.py` | `GenieAgent` init (default/custom name, empty space_id error, invalid space_id error); execution (response structure, AIMessage types); `include_context=True` (returns 3 named messages: query_reasoning, query_sql, query_result); pandas mode (dataframe key in output); conversation continuity; auth paths (PAT, OAuth M2M) |

**Key regressions these tests guard against:**
- Type coercion bugs in `_parse_query_result()` (FLOAT/DOUBLE/DECIMAL rendered in scientific notation, TIMESTAMP format parsing, NaN values from bad parsing)
- Conversation routing: passing a `conversation_id` must route to `create_message`, not `start_conversation`
- Pandas mode returning all-object dtypes (meaning type conversion silently failed)
- The `include_context` flag not propagating through the `RunnableLambda` wrapping

**Infrastructure:** Requires a pre-configured Genie Space (ID provided via `GENIE_SPACE_ID`). Session-scoped fixtures make 3 Genie API calls (data question, follow-up, "Hello") for core tests and 4 for LangChain tests, caching all responses.

### 3. MCP

**What the bridge provides:** A core `DatabricksMCPClient` that connects to Databricks-hosted MCP servers (UC Functions, Vector Search, Genie, DBSQL, external) using OAuth via `DatabricksOAuthClientProvider`. The client supports `list_tools()`/`call_tool()` (sync and async), MLflow resource extraction (`get_databricks_resources()`), and enhanced error diagnostics. LangChain wraps this as `DatabricksMCPServer`/`DatabricksMultiServerMCPClient` (with `from_uc_function()` and `from_vector_search()` convenience constructors). OpenAI provides two wrappers: `McpServerToolkit` (for chat completions function calling) and `McpServer` (for the Agents SDK, with MLflow tracing on `call_tool()`).

Unlike Vector Search, MCP uses plain `WorkspaceClient()` (SDK-native auth) -- no PAT conversion needed because auth is delegated to `DatabricksOAuthClientProvider`.

**Test files:**

| Layer | File | What it tests |
|-------|------|---------------|
| Core | `databricks_mcp/tests/integration_tests/test_mcp_core.py` | UC Functions: list_tools, call_tool (echo), schema-level listing; Vector Search: list/call tools; DBSQL: expected tool names (`execute_sql`, `execute_sql_read_only`, `poll_sql_result`), read-only execution; Genie: list/call tools; Raw `streamablehttp_client` session (bypassing `DatabricksMCPClient`); Error paths: nonexistent function, nonexistent tool name, wrong arguments; Auth: OAuth M2M, PAT |
| Core | `databricks_mcp/tests/integration_tests/test_mcp_resources.py` | MLflow resource extraction: UC functions produce `DatabricksFunction`, VS produces `DatabricksVectorSearchIndex`, Genie produces `DatabricksGenieSpace`; resource counts match tool counts; resource names use dot notation (not `__`) |
| LangChain | `integrations/langchain/tests/integration_tests/test_langchain_mcp.py` | `DatabricksMCPServer`: URL construction, connection dict structure, `get_tools()` returns `BaseTool`, tool invocation, `handle_tool_error` propagation, timeout conversion; `from_vector_search()`: URL pattern, tool invocation, schema-level listing; `DatabricksMultiServerMCPClient`: multi-server tool loading + invocation; Error paths; `DatabricksMcpHttpClientFactory`: creates fresh auth provider per request (token freshness) |
| OpenAI | `integrations/openai/tests/integration_tests/test_openai_mcp.py` | `McpServerToolkit`: `get_tools()` returns `ToolInfo` with OpenAI function-calling schema, name prefixing, tool execution, convenience constructors, auth paths; `McpServer` (Agents SDK): async context manager, list/call tools, MLflow trace creation, timeout defaults/override; Error paths for both wrappers |

**Key regressions these tests guard against:**
- `DatabricksOAuthClientProvider` token expiry causing stale tokens (the `DatabricksMcpHttpClientFactory` test verifies fresh providers per request)
- Tool name normalization (`__` to `.`) breaking resource declarations
- URL construction from catalog/schema/name components producing malformed endpoints
- `McpServerToolkit` name prefixing colliding across multiple servers
- MLflow tracing not attaching to `McpServer.call_tool()` (verified via `mlflow.get_last_active_trace()`)

**Infrastructure:** UC function `integration_testing.databricks_ai_bridge_mcp_test.echo_message`, the same VS index used by VS tests, a Genie Space (same `GENIE_SPACE_ID`), and the built-in DBSQL MCP endpoint (`/api/2.0/mcp/sql`). Session-scoped fixtures cache tool listings and call results.

### 4. FMAPI Tool Calling

**What the bridge provides:** `DatabricksOpenAI`/`AsyncDatabricksOpenAI` clients that wrap the standard OpenAI SDK with Databricks-specific middleware: `strict` field stripping for non-GPT models, empty assistant content fix for Claude models, and response/input ID truncation to 64 characters (FMAPI generates longer IDs but rejects them on subsequent turns). `ChatDatabricks` (LangChain) delegates to these clients, supporting both Chat Completions and Responses API paths with `bind_tools()`.

**Motivation -- early detection of broken models:** These tests were created after two customer-reported regressions: the `strict: True` field injected by OpenAI Agents SDK v0.6.4+ broke non-GPT models ([PR #269](https://github.com/databricks/databricks-ai-bridge/pull/269)), and empty assistant content with `tool_calls` broke Claude models ([PR #333](https://github.com/databricks/databricks-ai-bridge/pull/333)). Both were caught by customers, not CI. Beyond these specific fixes, Databricks FMAPI continuously onboards new models, each with subtly different tool calling behavior. The FMAPI tests exist to catch these failures *the moment a new model appears*, before customers hit them.

**How dynamic model discovery works:** The `discover_chat_models()` utility (in `src/databricks_ai_bridge/test_utils/fmapi.py`) runs at pytest collection time:
1. Lists all `databricks-*` serving endpoints with `llm/v1/chat` task
2. Queries each endpoint's capabilities API for `function_calling: true`
3. Excludes models in the skip list
4. Returns the remaining models as `pytest.mark.parametrize` values

When a new model is added to FMAPI, it **automatically appears in the test matrix without any code change**. If it breaks tool calling in any way, the nightly CI catches it immediately.

**The skip list** (`src/databricks_ai_bridge/test_utils/fmapi.py`) serves as a documented registry of known model incompatibilities with explanations (e.g., "hallucinates tool names in agent loop", "requires thought_signature on function calls", "Responses API only"). When a model is fixed upstream, its skip entry is removed and tests automatically resume covering it.

**Test files:**

| Layer | File | What it tests |
|-------|------|---------------|
| OpenAI | `integrations/openai/tests/integration_tests/test_fmapi_tool_calling.py` | **Agents SDK (async):** single-turn tool call round-trip, multi-turn with conversation history (exercises ID truncation), streaming events; **Sync client:** same 3 variants via raw `chat.completions.create()`; **Responses API:** same 3 variants. All parametrized across discovered models. Uses a real UC function (`echo_message`) via `McpServer` for async tests. |
| LangChain | `integrations/langchain/tests/integration_tests/test_fmapi_tool_calling.py` | **Sync LangGraph:** single-turn (math problem with `add`/`multiply` tools, expects "45"), multi-turn with `MemorySaver` checkpointer, streaming (`stream_mode="updates"`); **Async LangGraph:** same 3 variants; **Responses API:** same 3 variants. All parametrized across discovered models. |

**What each test variant catches:**
- **Single-turn:** Model not recognizing tool definitions, malformed `function.arguments` JSON, missing `tool_call.id`, wrong `tool_call.type`
- **Multi-turn:** ID handling bugs (the 64-char truncation workaround), inability to process `tool` role messages in history, failure to call tools on follow-up turns. Without multi-turn tests, a model could pass single-turn but fail in any real agent scenario.
- **Streaming:** Malformed streaming deltas where function name/arguments are split across SSE events. A model can return correct non-streaming responses but produce broken streaming -- this happens in practice.

Every test body is wrapped in a retry (3 attempts) to absorb transient FMAPI errors and model non-determinism. The 60-minute CI timeout reflects the large test matrix (10-20+ models x 3 variants x 2-3 API paths).

### 5. Lakebase

**What the bridge provides:** A core layer with `LakebasePool`/`AsyncLakebasePool` (psycopg connection pools with OAuth token rotation), `LakebaseClient` (SQL execution + PostgreSQL role/permission management via the `databricks_auth` extension), and `AsyncLakebaseSQLAlchemy` (SQLAlchemy async engine with token injection). LangChain wraps these as `DatabricksStore`/`AsyncDatabricksStore` (LangGraph `BaseStore` backed by `PostgresStore`) and `CheckpointSaver`/`AsyncCheckpointSaver` (LangGraph `PostgresSaver`). OpenAI wraps as `AsyncDatabricksSession` (OpenAI Agents SDK `SQLAlchemySession` with engine caching).

**Two deployment modes:** Provisioned (dedicated instance via `instance_name`) and autoscaling (serverless via `project`+`branch` or `autoscaling_endpoint`). Host resolution and token minting use different SDK APIs for each mode. The tests cover both.

**Test files:**

| Layer | File | What it tests |
|-------|------|---------------|
| Core | `tests/integration_tests/lakebase/test_lakebase_integration.py` | **Validation errors** (10 tests): missing params, conflicting params, invalid identity types, empty inputs; **Permission matrix**: schema privileges (USAGE, CREATE, ALL), table privileges (SELECT, INSERT, ALL), sequence privileges, grants on specific tables, multiple schemas; **Role management**: create_role idempotency, create_role for service principals; **End-to-end**: full permission setup flow; **Error scenarios with secondary SPs**: no-role user can't connect (`PoolTimeout`), no-role user can't create_role, limited-permission user can't grant, limited-permission user can't create_role (missing CAN MANAGE); **Object-not-found**: nonexistent schema/table/role produce `ValueError`; **Autoscaling** (15 tests): project+branch, endpoint, and branch-as-resource-path modes each test connect+execute, create_role, and grants |
| LangChain | `integrations/langchain/tests/integration_tests/test_langchain_lakebase.py` | `DatabricksStore`: put/get round-trip, search, list namespaces; `AsyncDatabricksStore`: same 4 operations; `CheckpointSaver`: write + get tuple round-trip; `AsyncCheckpointSaver`: same; Autoscaling variants (project+branch, endpoint) mirror all provisioned tests |
| OpenAI | `integrations/openai/tests/integration_tests/test_memory_session.py` | `AsyncDatabricksSession`: CRUD lifecycle, session isolation, pop-empty-returns-none, add-empty-noop, complex nested message data, chronological ordering; Autoscaling: project+branch and endpoint CRUD |

**Key regressions these tests guard against:**
- OAuth token cache expiry causing connection failures (tokens expire every 15 minutes; pools must rotate them transparently)
- Permission grant SQL injection (all grants use `psycopg.sql` composition, but wrong composition could produce invalid SQL)
- `create_role()` not being idempotent (must handle `DuplicateObject` gracefully)
- Autoscaling host resolution returning the wrong endpoint type (must find READ_WRITE, not READ_ONLY)
- LangGraph `PostgresStore` migration tracker skipping table creation on re-runs (cleanup fixtures drop tables before AND after tests)

**Permission testing uses secondary service principals:** The CI injects credentials for two additional SPs beyond the main CI SP:
- A "no-role" SP that has never had `create_role()` called -- verifies that unauthorized access fails with meaningful errors, not silent data leaks
- A "limited" SP with a PostgreSQL role but without CAN MANAGE or GRANT OPTION -- verifies that privilege escalation fails

**Cleanup patterns:** Core tests use inline `CREATE TABLE`/`DROP TABLE` in fixtures. LangChain tests drop well-known LangGraph table names before and after the module. OpenAI tests generate unique table names per test (UUID suffix) and drop in order respecting foreign keys.

### 6. OBO (On-Behalf-Of) Credentials

**What the bridge provides:** `ModelServingUserCredentials` -- a `CredentialsStrategy` that retrieves the invoking user's downscoped OBO token inside Model Serving (via `mlflowserving.scoring_server.agent_utils.fetch_obo_token()`), falling back to `DefaultCredentials` outside Model Serving. This enables deployed agents to act as the user who invoked them, not the service principal that owns the deployment -- critical for data governance.

**Why this needs integration tests:** OBO involves multiple moving parts across two deployment surfaces (Model Serving and Databricks Apps), each with a different mechanism for obtaining user identity. The only way to verify end-to-end identity propagation is to deploy real agents and call them as different users.

**The two-SP model:** Tests use two service principals:
- **SP-A** (the CI SP): deploys and owns the agents, also acts as the first test caller
- **SP-B** (a separate "end user" SP): calls the same agents as a different identity

The fundamental assertion: when SP-A calls the agent, it sees SP-A's identity; when SP-B calls, it sees SP-B's. If both see the same identity, OBO is broken.

**Two deployment fixtures, two OBO mechanisms:**

| | Model Serving Fixture | App Fixture |
|---|---|---|
| **Location** | `tests/integration_tests/obo/model_serving_fixture/` | `tests/integration_tests/obo/app_fixture/` |
| **Agent type** | MLflow `ResponsesAgent` (pyfunc) | OpenAI Agents SDK agent |
| **OBO mechanism** | `ModelServingUserCredentials` reads the OBO token from the scoring server runtime | `x-forwarded-access-token` HTTP header injected by the Apps proxy |
| **Identity check** | `SELECT integration_testing.databricks_ai_bridge_mcp_test.whoami()` via SQL Statement Execution API | `WorkspaceClient.current_user.me()` SDK call |
| **Deployment** | `mlflow.pyfunc.log_model()` + `agents.deploy()` | Databricks App (source code deployed) |
| **Needs periodic redeploy** | Yes (pip deps frozen at model log time) | No (deps in `pyproject.toml`, resolved at deploy) |

**Test file:**

| File | What it tests |
|------|---------------|
| `tests/integration_tests/obo/test_obo_credential_flow.py` | `TestModelServingOBO` (3 tests): SP-A and SP-B get different responses; SP-A's response contains SP-A's client ID; SP-B's response contains SP-B's client ID. `TestAppsOBO` (3 identical tests targeting `apps/{app_name}`). Both use `DatabricksOpenAI` Responses API as the client. |

**Warmup:** The Model Serving endpoint uses `scale_to_zero=True`. The `serving_endpoint_ready` fixture polls endpoint state (up to 20 attempts at 30s intervals = 10 minutes max) before tests run. App lifecycle (deploy before tests, stop after) is managed by the CI runner workflow, not the test code.

**Weekly redeploy (`deploy_serving_agent.py`):** The Model Serving endpoint must run on the latest SDK versions (`databricks-openai`, `databricks-ai-bridge`, `databricks-sdk`, `mlflow`) because pip requirements are frozen at model log time. A separate weekly CI workflow re-logs the model with current package versions and redeploys. The App fixture does not need this because its dependencies resolve from `pyproject.toml` at deploy time.

---

## Running Tests Locally

### Auth Setup

Integration tests authenticate via environment variables or CLI profile. The typical local setup:

```bash
# Option 1: CLI profile (converted to PAT internally by test fixtures)
export DATABRICKS_CONFIG_PROFILE=your-profile

# Option 2: Direct env vars (same as CI)
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
export DATABRICKS_CLIENT_ID=your-sp-client-id
export DATABRICKS_CLIENT_SECRET=your-sp-secret
```

**Important:** If using a CLI profile, unset `DATABRICKS_HOST`/`DATABRICKS_CLIENT_ID`/`DATABRICKS_CLIENT_SECRET` to avoid conflicts. The VS and Genie test fixtures convert CLI profile auth to PAT automatically (required because `VectorSearchClient` doesn't support CLI profile auth).

### Running a Specific Suite

```bash
# Vector Search
cd /path/to/databricks-ai-bridge
RUN_VS_INTEGRATION_TESTS=1 python -m pytest tests/integration_tests/vector_search/ -v

# Genie (needs GENIE_SPACE_ID)
RUN_GENIE_INTEGRATION_TESTS=1 GENIE_SPACE_ID=your-space-id \
  python -m pytest tests/integration_tests/genie/ -v

# MCP
RUN_MCP_INTEGRATION_TESTS=1 python -m pytest databricks_mcp/tests/integration_tests/ -v

# Lakebase (needs LAKEBASE_INSTANCE_NAME or autoscaling vars)
LAKEBASE_INTEGRATION_TESTS=1 LAKEBASE_INSTANCE_NAME=your-instance \
  python -m pytest tests/integration_tests/lakebase/ -v

# FMAPI tool calling
RUN_FMAPI_TOOL_CALLING_TESTS=1 \
  python -m pytest integrations/openai/tests/integration_tests/test_fmapi_tool_calling.py -v

# OBO (needs all OBO env vars + pre-deployed agents)
RUN_OBO_INTEGRATION_TESTS=1 OBO_TEST_SERVING_ENDPOINT=your-endpoint \
  OBO_TEST_APP_NAME=your-app OBO_TEST_CLIENT_ID=sp-b-id OBO_TEST_CLIENT_SECRET=sp-b-secret \
  python -m pytest tests/integration_tests/obo/ -v
```

### Running LangChain/OpenAI Wrapper Tests

These live in their respective integration directories and use the same gate variables:

```bash
# LangChain VS tests
cd integrations/langchain
RUN_VS_INTEGRATION_TESTS=1 python -m pytest tests/integration_tests/test_langchain_vectorsearch.py -v

# OpenAI MCP tests
cd integrations/openai
RUN_MCP_INTEGRATION_TESTS=1 python -m pytest tests/integration_tests/test_openai_mcp.py -v
```

### Environment Variables Reference

| Variable | Required By | Description |
|----------|-------------|-------------|
| `DATABRICKS_HOST` | All | Workspace URL |
| `DATABRICKS_CLIENT_ID` | CI | Service principal client ID |
| `DATABRICKS_CLIENT_SECRET` | CI | Service principal client secret |
| `DATABRICKS_CONFIG_PROFILE` | Local | CLI profile name (alternative to above) |
| `RUN_VS_INTEGRATION_TESTS` | VS | Set to `1` to enable |
| `RUN_GENIE_INTEGRATION_TESTS` | Genie | Set to `1` to enable |
| `GENIE_SPACE_ID` | Genie, MCP (Genie tools) | Genie space identifier |
| `RUN_MCP_INTEGRATION_TESTS` | MCP | Set to `1` to enable |
| `RUN_FMAPI_TOOL_CALLING_TESTS` | FMAPI | Set to `1` to enable |
| `LAKEBASE_INTEGRATION_TESTS` | Lakebase | Set to `1` to enable |
| `LAKEBASE_INSTANCE_NAME` | Lakebase (provisioned) | Provisioned instance name |
| `LAKEBASE_PROJECT` | Lakebase (autoscaling) | Autoscaling project name |
| `LAKEBASE_BRANCH` | Lakebase (autoscaling) | Autoscaling branch name |
| `LAKEBASE_AUTOSCALING_ENDPOINT` | Lakebase (autoscaling) | Autoscaling endpoint name |
| `RUN_OBO_INTEGRATION_TESTS` | OBO | Set to `1` to enable |
| `OBO_TEST_CLIENT_ID` | OBO | Second SP (end-user) client ID |
| `OBO_TEST_CLIENT_SECRET` | OBO | Second SP (end-user) client secret |
| `OBO_TEST_SERVING_ENDPOINT` | OBO | Pre-deployed Model Serving endpoint |
| `OBO_TEST_APP_NAME` | OBO | Pre-deployed Databricks App name |

---

## Directory Structure

```
tests/integration_tests/
  conftest.py                            # Shared: workspace_client (with CLI->PAT conversion), markers
  vector_search/
    conftest.py                          # VS fixtures: indexes, VectorSearchClient, skip gate
    test_bridge_utilities.py             # IndexDetails, parse_vector_search_response, validators
    test_vectorsearch_behavior.py        # Raw VS SDK smoke tests (response shape, scores, filters)
  genie/
    conftest.py                          # Genie fixtures: cached responses, skip gate
    test_genie_behavior.py               # ask_question, conversation continuity, pandas mode
    test_genie_bridge.py                 # _parse_query_result: markdown tables, type conversion
  lakebase/
    test_lakebase_integration.py         # LakebaseClient: permissions, roles, validation, autoscaling
  obo/
    test_obo_credential_flow.py          # Identity isolation: Model Serving + Apps, two SPs
    deploy_serving_agent.py              # Weekly redeploy script for Model Serving endpoint
    model_serving_fixture/
      whoami_serving_agent.py            # MLflow ResponsesAgent using ModelServingUserCredentials
    app_fixture/                         # Databricks App using OpenAI Agents SDK + x-forwarded-access-token

integrations/langchain/tests/integration_tests/
  test_langchain_vectorsearch.py         # VectorSearchRetrieverTool + DatabricksVectorSearch
  test_langchain_genie.py               # GenieAgent wrapper
  test_langchain_mcp.py                 # DatabricksMCPServer + DatabricksMultiServerMCPClient
  test_fmapi_tool_calling.py            # ChatDatabricks tool calling via LangGraph
  test_langchain_lakebase.py            # DatabricksStore + CheckpointSaver

integrations/openai/tests/integration_tests/
  test_openai_vectorsearch.py           # VectorSearchRetrieverTool (OpenAI schema)
  test_openai_mcp.py                    # McpServerToolkit + McpServer (Agents SDK)
  test_fmapi_tool_calling.py            # DatabricksOpenAI tool calling via Agents SDK
  test_memory_session.py                # AsyncDatabricksSession (memory storage)

databricks_mcp/tests/integration_tests/
  conftest.py                            # MCP fixtures: UC functions, VS, DBSQL, Genie clients
  test_mcp_core.py                       # DatabricksMCPClient: list/call across 4 resource types
  test_mcp_resources.py                  # MLflow resource extraction and naming
```
