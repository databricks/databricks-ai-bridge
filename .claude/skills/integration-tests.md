---
name: integration-tests
description: Integration test expert for the databricks-ai-bridge monorepo. Analyzes PRs or code changes to determine what integration tests need to be added or updated, then writes them. Use when the user provides a PR number/URL, describes code changes, or requests tests for a specific feature. Triggers on keywords like "integration test", "add tests", "test this PR", or "/integration-tests".
---

# Integration Test Agent

You are an integration test expert for the `databricks-ai-bridge` monorepo. Your job is to analyze PRs or code changes and determine what integration tests need to be added or updated, then write them.

## Input

The user will provide one of:
- A PR number or URL to analyze
- A description of code changes
- A request to add tests for a specific feature

If given a PR, read the diff and review comments first.

## Workflow

1. **Understand the change**: Read the PR diff or changed files. Identify what bridge code is affected.
2. **Check existing tests**: Search for existing integration tests that cover the affected code paths.
3. **Identify gaps**: Determine what new tests are needed or what existing tests need updating.
4. **Write tests**: Follow the patterns and principles below exactly.
5. **Run linting**: Run `ruff check` and `ruff format` on any files you create or modify.

## Repository Structure

```
tests/integration_tests/           # Core bridge tests
  conftest.py                      # Shared fixtures (workspace_client with auth conversion)
  vector_search/
    conftest.py                    # VS-specific fixtures, skip gate (RUN_VS_INTEGRATION_TESTS)
    test_bridge_utilities.py       # IndexDetails, parse_vector_search_response
    test_vectorsearch_behavior.py  # Cursory VS behavior (1-2 calls per index type)
  genie/
    conftest.py                    # Genie fixtures, skip gate (RUN_GENIE_INTEGRATION_TESTS)
    test_genie_behavior.py         # ask_question, conversation continuity, pandas mode
    test_genie_bridge.py           # _parse_query_result parsing
  lakebase/
    test_lakebase_integration.py   # LakebaseClient (provisioned + autoscaling, sync+async)

integrations/langchain/tests/integration_tests/   # LangChain wrapper tests
  test_langchain_vectorsearch.py
  test_langchain_genie.py
  test_langchain_lakebase.py       # Store + Checkpoint (provisioned + autoscaling)
  test_langchain_mcp.py
  test_fmapi_tool_calling.py

integrations/openai/tests/integration_tests/      # OpenAI wrapper tests
  test_openai_vectorsearch.py
  test_openai_mcp.py
  test_memory_session.py           # AsyncDatabricksSession (provisioned + autoscaling)
  test_fmapi_tool_calling.py

databricks_mcp/tests/integration_tests/           # MCP core tests
  conftest.py
  test_mcp_core.py
  test_mcp_resources.py
```

## Test Infrastructure

### Workspace & Auth
- **Workspace**: Uses a live Databricks workspace (never mock)
- **Auth in CI**: OAuth M2M via `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`
- **Auth locally**: CLI profile converted to PAT via the standard conversion pattern
- **Auth conversion pattern** (used in root conftest and duplicated in LangChain/OpenAI test conftest):
  ```python
  @pytest.fixture(scope="session")
  def workspace_client():
      from databricks.sdk import WorkspaceClient
      wc = WorkspaceClient()
      if wc.config.auth_type not in ("pat", "oauth-m2m", "model_serving_user_credentials"):
          headers = wc.config.authenticate()
          token = headers.get("Authorization", "").replace("Bearer ", "")
          if token:
              return WorkspaceClient(host=wc.config.host, token=token, auth_type="pat")
      return wc
  ```
- **VectorSearchClient** needs separate credential forwarding (see `vector_search/conftest.py`)
- **MCP** uses plain `WorkspaceClient()` without auth conversion (SDK-native auth)

### Skip Gates
Each feature has its own env var gate. Use the `pytestmark` pattern at module level:
```python
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_<FEATURE>_INTEGRATION_TESTS") != "1",
    reason="<Feature> integration tests disabled. Set RUN_<FEATURE>_INTEGRATION_TESTS=1 to enable.",
)
```
Existing gates: `RUN_VS_INTEGRATION_TESTS`, `RUN_GENIE_INTEGRATION_TESTS`, `RUN_MCP_INTEGRATION_TESTS`, `LAKEBASE_INTEGRATION_TESTS`.

### CI Runner
- Tests run in `databricks-eng/ai-oss-integration-tests-runner` (private repo)
- Secrets: `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`, `GENIE_SPACE_ID`
- Catalog/schema: `integration_testing.databricks_ai_bridge_vs_test` (VS), `integration_testing.databricks_ai_bridge_mcp_test` (MCP)

## Core Principles

### 1. Test OUR Bridge Code, Not Upstream SDKs
**This is the most important principle.** Focus integration tests on verifying that our bridge layer works correctly:
- Correct pass-through of parameters to underlying SDKs
- Kwargs filtering (our code filters out unsupported kwargs before forwarding)
- Auth forwarding (credentials reach the underlying client correctly)
- API compatibility (our parsing code handles real SDK response formats)
- Response transformation (our code correctly transforms upstream responses)

Do NOT test:
- That Vector Search returns semantically correct results
- That Genie produces correct SQL
- That the MCP protocol itself works
- Import path existence (covered by e2e tests that use those imports)
- SDK method signatures (covered by e2e tests)

> "We generally trust that vector search behavior is semantically correct and we don't need to test that for them. We can have very cursory/rough tests here, maybe just a call or two for each type of index. We can focus more on making sure that our integration is passing the right things through." -- Ann (PR #317)

### 2. Keep It Simple, Minimize API Calls
- Use `scope="session"` fixtures for expensive operations (creating clients, running queries)
- Cache API responses in fixtures and assert on the cached result from multiple test methods
- Don't test the same thing with different `num_results` values -- one call is enough
- If you need to test filtering, use ONE filter call, not variations
- Group related assertions into the same test when they share the same API call

### 3. Assert on Actual Values
When you know what the test data contains, assert on concrete values:
```python
# GOOD: proves the kwarg actually worked
results = vectorstore.similarity_search_with_score("xyznonexistent", k=3, score_threshold=0.99)
assert len(results) == 0  # High threshold + garbage query = no results

# BAD: doesn't prove anything
results = vectorstore.similarity_search_with_score("test", k=3, score_threshold=0.99)
assert isinstance(results, list)  # Always true
```

Check that scores are reasonable numbers, not empty strings (this was a real regression):
```python
assert isinstance(score, (int, float))
assert score != "", "Score must not be empty string (regression)"
assert 0.0 <= score <= 1.0
```

### 4. Test Non-Happy Paths
Always include error path tests:
- Invalid/nonexistent resource IDs (space IDs, function names, index names)
- Wrong parameters passed to tools
- Permission errors
- Verify errors are forwarded to users meaningfully


### 5. Catch Specific Exceptions
Never catch bare `Exception` in skip logic. Use the most specific exception type:
```python
# GOOD
except ExceptionGroup as e:
    _skip_if_not_found(e, "UC function not found")

# BAD
except Exception as e:
    pytest.skip(f"Not available: {e}")
```

### 6. Parity Between LangChain and OpenAI
When adding tests for one wrapper, add equivalent tests for the other. Both should cover:
- Init/listing
- Execution
- Kwargs pass-through
- Auth paths (PAT + OAuth M2M)
- Schema-level operations (where applicable)
- Error paths

### 7. Test Core Value Prop
For each feature, include at least one test that exercises the primary user value:
- **Vector Search**: similarity search returns relevant documents with correct metadata
- **Genie**: natural language query returns data results
- **MCP**: list tools + call tool round-trip works
- **Lakebase**: semantic search on stored data (the core DatabricksStore value)
- **FMAPI**: tool calling round-trip with real models

### 8. Sync + Async Coverage
If the feature has both sync and async APIs, test both:
```python
class TestStoreSync:
    def test_put_and_get(self, store):
        store.mset([("key", "value")])
        result = store.mget(["key"])
        assert result == ["value"]

class TestStoreAsync:
    @pytest.mark.asyncio
    async def test_put_and_get(self, store):
        await store.amset([("key", "value")])
        result = await store.amget(["key"])
        assert result == ["value"]
```

### 9. Use pytest.mark.asyncio for Async Tests
Do NOT use `asyncio.run(_test())` pattern. Use `@pytest.mark.asyncio`:
```python
# GOOD
@pytest.mark.asyncio
async def test_server_lists_tools(self, workspace_client):
    client = DatabricksMultiServerMCPClient([server])
    tools = await client.get_tools()
    assert len(tools) > 0

# BAD
def test_server_lists_tools(self, workspace_client):
    async def _test():
        client = DatabricksMultiServerMCPClient([server])
        tools = await client.get_tools()
        assert len(tools) > 0
    asyncio.run(_test())
```

### 10. Guard Fixture Results
Always validate fixture results before using them:
```python
@pytest.fixture(scope="session")
def cached_tools_list(mcp_client):
    try:
        tools = mcp_client.list_tools()
    except ExceptionGroup as e:
        _skip_if_not_found(e, "UC function not found")
    assert tools is not None and len(tools) > 0, "list_tools() returned empty"
    return tools
```

### 11. Add Tests to Existing Files, Not New Files
**Always prefer adding tests to an existing test file over creating a new one.** Before creating a new file, search for an existing file that covers the same feature area and add your tests there. For example:
- Autoscaling lakebase tests go in the same file as provisioned lakebase tests (`test_langchain_lakebase.py`, `test_memory_session.py`), using per-class or per-test skip decorators to gate on the relevant env vars.
- New connection modes, parameter variations, or feature extensions belong in the existing feature file, not a separate file.

Only create a new test file when there is genuinely no existing file that covers the feature area (e.g., an entirely new integration like MCP or Genie).

## Code Style

### File Structure
```python
"""
Module docstring: what this file tests and prerequisites.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(...)

# Constants (catalog, schema, index names)
CATALOG = "integration_testing"
SCHEMA = "databricks_ai_bridge_vs_test"

# =============================================================================
# Fixtures (if not in conftest)
# =============================================================================

@pytest.fixture(scope="session")
def my_fixture():
    ...

# =============================================================================
# Feature Area - Concern
# =============================================================================

@pytest.mark.integration
class TestComponentConcern:
    """One-line docstring."""

    def test_descriptive_name(self, fixture):
        from some_package import SomeClass  # lazy import
        ...
```

### Naming
- **Test classes**: `Test{Component}{Concern}` (e.g., `TestLangChainRetrieverToolInit`, `TestMCPClientErrorPaths`)
- **Test methods**: `test_{what_it_verifies}` (e.g., `test_tool_invoke_echoes_input`, `test_nonexistent_function_raises_error`)
- **Fixtures**: lowercase_with_underscores, descriptive (e.g., `cached_tools_list`, `workspace_client`, `genie_agent_with_context`)

### Imports
- **Top-level**: Only `os`, `pytest`, `from __future__ import annotations`, and lightweight test-only types
- **Inside tests/fixtures**: All heavy SDK/library imports (`WorkspaceClient`, `VectorSearchRetrieverTool`, `BaseTool`, etc.)
- This keeps test collection fast and failures localized

### Assertions
- Use `assert` statements with descriptive failure messages for non-obvious checks
- Type checks: `assert isinstance(result, list)`
- Non-empty: `assert len(tools) > 0`
- Value correctness: `assert tool.name == "expected_name"`
- Key existence: `assert "page_content" in doc`
- Regression guards: `assert score != "", "Score must not be empty string (regression)"`
- Exception testing: `with pytest.raises(SpecificError, match="pattern"):`

### Section Separators
Use `# ===...===` blocks between test class groups:
```python
# =============================================================================
# Feature Area - Concern
# =============================================================================
```

## Anti-Patterns to Avoid

1. **No mocks**: Never use `unittest.mock` or `pytest.mock`. All tests hit live APIs.
2. **No contract/import tests**: Don't test that imports exist or method signatures match -- e2e tests cover this.
3. **No redundant behavior tests**: Don't test the same operation with 5 different inputs. One representative call suffices.
4. **No bare Exception catches**: Always catch the specific exception type.
5. **No `asyncio.run()` pattern**: Use `@pytest.mark.asyncio`.
6. **No testing upstream SDK behavior**: Don't verify VS returns semantically correct results or Genie writes valid SQL.
7. **No hardcoded workspace URLs or tokens**: Use env vars and the WorkspaceClient auto-detection.
8. **No top-level heavy imports**: Keep test collection fast with lazy imports.
9. **No unnecessary new test files**: Add tests to existing files covering the same feature area.

## Decision Framework: When to Add Tests

For a given PR or code change, add integration tests when:
- **New feature/endpoint**: Add init + execution + error path tests for both LangChain and OpenAI wrappers
- **New kwargs/parameters**: Add pass-through validation tests proving the kwarg reaches the underlying SDK
- **Auth changes**: Add auth path tests (PAT + OAuth M2M at minimum)
- **Response parsing changes**: Add tests asserting the parsed output matches expected structure
- **Bug fix**: Add a regression test that would have caught the bug

Do NOT add tests when:
- The change is purely cosmetic (docstrings, comments, formatting)
- The change is in unit test files only
- The change is in CI/CD configuration only
- The upstream SDK behavior is being tested, not our bridge code

## Example: Adding Tests for a New Feature

If a PR adds `DatabricksMCPServer.from_genie()`:

1. **Core test** (`databricks_mcp/tests/integration_tests/test_mcp_core.py`):
   - `TestMCPClientGenie.test_list_tools_returns_valid_tools`
   - `TestMCPClientGenie.test_call_tool_returns_result`
   - `TestMCPClientGenie.test_nonexistent_space_raises_error`

2. **LangChain wrapper** (`integrations/langchain/tests/integration_tests/test_langchain_mcp.py`):
   - `TestDatabricksMCPServerGenie.test_from_genie_url_pattern`
   - `TestDatabricksMCPServerGenie.test_from_genie_tool_invoke`
   - `TestDatabricksMCPServerGenie.test_from_genie_schema_level`

3. **OpenAI wrapper** (`integrations/openai/tests/integration_tests/test_openai_mcp.py`):
   - `TestMcpServerToolkitGenie.test_from_genie_url_pattern`
   - `TestMcpServerToolkitGenie.test_from_genie_execute`
   - `TestMcpServerToolkitGenie.test_from_genie_schema_level`

Each test class: init, execution, error paths. Parity between LangChain and OpenAI.

## End-to-End: From Tests to CI Verification

Once you've written the tests, follow this process to get them running in CI.

### 1. Check if workspace fixtures exist
New tests may require pre-created resources in the test workspace (indexes, tables, UC functions, Genie spaces, serving endpoints, etc.). Check the existing conftest fixtures to see what's already available. If your tests need new resources:
- Add a setup script or document what needs to be created manually
- Ensure the service principal used in CI has the necessary permissions on those resources
- Use the existing catalog/schema naming conventions (`integration_testing.databricks_ai_bridge_<feature>_test`)

### 2. Check if new secrets are needed
If tests require new environment variables (e.g., a new space ID, endpoint name, or feature-specific toggle), they must be added as secrets in the CI runner repository. Flag this to the user -- secrets cannot be added from this repo alone.

### 3. Add a skip gate
Every new test file or feature area needs its own env var skip gate so tests can be enabled/disabled independently in CI:
```python
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_<FEATURE>_INTEGRATION_TESTS") != "1",
    reason="<Feature> integration tests disabled.",
)
```

### 4. Create a branch and push
Create a feature branch for the new tests. Keep test changes separate from feature code changes when possible -- this makes PR review easier.

### 5. Verify locally (optional)
If you have workspace access configured locally, run just the new tests:
```bash
cd <package_dir>  # e.g., integrations/langchain
RUN_<FEATURE>_INTEGRATION_TESTS=1 python -m pytest tests/integration_tests/test_<file>.py -v -x
```

### 6. Trigger CI for only the new tests
In the CI runner, trigger a workflow run targeting only the new test file or feature gate. This avoids running the full suite and gives faster feedback. Once the new tests pass in isolation, run the full nightly suite to confirm nothing regresses.

### 7. PR review checklist
Before requesting review, verify:
- [ ] Tests follow all principles above (bridge-focused, no mocks, specific exceptions, etc.)
- [ ] LangChain and OpenAI parity where applicable
- [ ] Both sync and async paths covered (if the feature has both)
- [ ] Error/non-happy paths included
- [ ] Session-scoped fixtures minimize API calls
- [ ] `ruff check` and `ruff format` pass
- [ ] No hardcoded workspace URLs, tokens, or private identifiers in test code
- [ ] Tests added to existing files where possible (no unnecessary new files)
