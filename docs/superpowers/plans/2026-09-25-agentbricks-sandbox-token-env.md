# Agent Bricks Sandbox Token Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make new Agent Bricks sandbox bindings expose the selected App or request-user Databricks credential to sandbox code by default, while preserving legacy manifest behavior.

**Architecture:** Extend the protected sandbox policy in `agent.toml` with a boolean `include_databricks_token_env`. Parse it in both CLI and runtime models, then have both framework adapters emit one shared protected metadata shape. Identity continues to be selected by the existing `auth` field and workspace-client resolver.

**Tech Stack:** Python 3.10+, dataclasses, Click, tomlkit/tomllib, pytest, LangGraph MCP adapter, OpenAI Agents MCP adapter, Databricks Agent Bricks CLI.

**Spec:** `docs/superpowers/specs/2026-09-25-agentbricks-sandbox-token-env.md`

## Global Constraints

- New `ab tools add sandbox` bindings default `include_databricks_token_env` to `true`.
- `--no-include-databricks-token-env` opts a new binding out.
- Legacy manifests that omit the field parse as `false`.
- Only sandbox policies accept the field, and it must be a TOML boolean.
- LangGraph and OpenAI adapters must prevent caller metadata from overriding protected policy.
- `auth = "user"` remains request-user/OBO; `auth = "app"` remains App service principal.
- Live evidence must cover App/SP and User/OBO in both `ab dev` and `ab deploy` without recording credentials.

---

### Task 1: Manifest model and CLI default

**Files:**
- Modify: `integrations/agentbricks/src/databricks_agentbricks/agent_project.py`
- Modify: `integrations/agentbricks/src/databricks_agentbricks/cli/tools.py`
- Test: `integrations/agentbricks/tests/unit_tests/agent_project_test.py`
- Test: `integrations/agentbricks/tests/unit_tests/tools_test.py`

**Interfaces:**
- Produces: `ToolPolicy.include_databricks_token_env: bool`
- Produces: `ToolSpec.sandbox(..., include_databricks_token_env: bool = True)`
- Produces: CLI flag pair `--include-databricks-token-env/--no-include-databricks-token-env`

- [x] **Step 1: Write failing manifest and CLI tests**

Add cases that independently assert: the default CLI writes `true`; the negative flag writes
`false`; a manifest with the key omitted loads as `false`; and explicit true/false round-trips.

```python
assert loaded.tools[0].policy.include_databricks_token_env is True
assert "include_databricks_token_env = true" in manifest.read_text()
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
uv run --group tests pytest tests/unit_tests/tools_test.py tests/unit_tests/agent_project_test.py -q
```

Expected: failures because `ToolPolicy` and the Click command do not expose the new field.

- [x] **Step 3: Implement minimal manifest and CLI support**

Add the boolean field with a compatibility default of `False`, validate direct manifest values,
serialize it for sandbox bindings, default `ToolSpec.sandbox` and the CLI to `True`, and thread the
Click value through `add_sandbox_to_manifest`.

- [x] **Step 4: Run tests to verify GREEN**

Run the same focused pytest command and require all tests to pass.

- [x] **Step 5: Commit**

```bash
git add integrations/agentbricks/src/databricks_agentbricks/agent_project.py integrations/agentbricks/src/databricks_agentbricks/cli/tools.py integrations/agentbricks/tests/unit_tests/agent_project_test.py integrations/agentbricks/tests/unit_tests/tools_test.py
git commit -m "feat(agentbricks): default sandbox token env on"
```

### Task 2: Runtime parser and protected metadata

**Files:**
- Modify: `integrations/agentbricks/src/databricks_agentkit/runtime/tool_manifest.py`
- Modify: `integrations/agentbricks/src/databricks_agentkit/langgraph/mcp.py`
- Modify: `integrations/agentbricks/src/databricks_agentkit/openai/mcp.py`
- Test: `integrations/agentbricks/tests/unit_tests/tool_runtime_test.py`
- Test: `integrations/agentbricks/tests/unit_tests/mcp_request_user_test.py`

**Interfaces:**
- Consumes: `policy.include_databricks_token_env`
- Produces: `ToolRecord.include_databricks_token_env: bool`
- Produces: `sandbox_meta(tool: ToolRecord) -> dict[str, Any]`

- [x] **Step 1: Write failing runtime and adapter tests**

Extend direct-manifest coverage to assert the parsed boolean and this literal MCP metadata in
LangGraph. Add a parameterized adapter test proving App and User auth both receive identical policy
metadata while retaining their independently selected workspace clients. For OpenAI, pass a
conflicting caller value and assert the protected `true` wins.

```python
assert kwargs["meta"] == {
    "downscope": {"workspace_paths": [{"path": "/Workspace/Shared", "permission": "read_only"}]},
    "include_databricks_token_env": True,
}
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
uv run --group tests pytest tests/unit_tests/tool_runtime_test.py tests/unit_tests/mcp_request_user_test.py -q
```

Expected: failures because runtime records discard the flag and adapters emit only `downscope`.

- [x] **Step 3: Implement minimal runtime support**

Parse and validate the boolean in `ToolRecord`. Build both protected fields in `sandbox_meta`, use
that function in the LangGraph interceptor, and have `_DownscopedMcpServer` overwrite caller values
with the complete protected metadata in OpenAI.

- [x] **Step 4: Run tests to verify GREEN**

Run the same focused pytest command and require all tests to pass.

- [x] **Step 5: Commit**

```bash
git add integrations/agentbricks/src/databricks_agentkit integrations/agentbricks/tests/unit_tests/tool_runtime_test.py integrations/agentbricks/tests/unit_tests/mcp_request_user_test.py
git commit -m "feat(agentkit): forward sandbox token policy"
```

### Task 3: User documentation and full local verification

**Files:**
- Modify: `integrations/agentbricks/README.md`
- Modify: `integrations/agentbricks/cli.md`

**Interfaces:**
- Documents: generated manifest field, default behavior, opt-out, App/SP vs User/OBO semantics.

- [x] **Step 1: Update user-facing docs**

Show the new CLI option and a manifest example. Explain that enabling token exposure does not choose
identity; `auth` still does that. State that omitted legacy fields remain disabled.

- [x] **Step 2: Run package verification**

```bash
uv run --group tests pytest tests/unit_tests/ -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check
```

- [ ] **Step 3: Commit**

```bash
git add integrations/agentbricks/README.md integrations/agentbricks/cli.md docs/superpowers
git commit -m "docs(agentbricks): explain sandbox token exposure"
```

### Task 4: Draft PR before live E2E

**Files:**
- Create: draft PR in `databricks/databricks-ai-bridge`

**Interfaces:**
- Produces: a reviewable draft PR with unit verification and the pending E2E matrix.

- [ ] **Step 1: Run pre-PR verification and inspect the diff**
- [ ] **Step 2: Push the branch with account-scoped OSS GitHub credentials**
- [ ] **Step 3: Create and attach a draft PR**
- [ ] **Step 4: Verify the remote PR head matches the local commit**

### Task 5: Live `ab dev` and `ab deploy` identity matrix

**Files:**
- Create: `/tmp/e2e-report-agentbricks-sandbox-token-env.md`

**Interfaces:**
- Consumes: a workspace profile, a workspace-path sandbox scope, and the draft branch wheel.
- Produces: four evidence rows for `dev/app`, `dev/user`, `deploy/app`, and `deploy/user`.

- [ ] **Step 1: Prepare an isolated E2E project from the branch**
- [ ] **Step 2: Start the monitored `ab dev` process and test App/SP**
- [ ] **Step 3: Test User mode in `ab dev` and label local-profile identity accurately**
- [ ] **Step 4: Deploy the App-auth variant and verify sandbox `current_user.me()` is the App SP**
- [ ] **Step 5: Deploy the User-auth variant and verify sandbox `current_user.me()` is the caller**
- [ ] **Step 6: Record sanitized commands, outputs, resource IDs, and cleanup status in the report**

Every live probe runs sandbox Python equivalent to:

```python
from databricks.sdk import WorkspaceClient
me = WorkspaceClient().current_user.me()
print({"userName": me.user_name, "applicationId": me.application_id})
```

### Task 6: Eval-metric HTML report and PR evidence

**Files:**
- Create: `20260925_agentbricks-sandbox-token-env-e2e.html` in the report staging directory.
- Modify: draft PR description.

**Interfaces:**
- Consumes: the sanitized E2E evidence report.
- Produces: a published metric report URL with round-trip SHA verification.

- [ ] **Step 1: Generate a clean HTML metric report**
- [ ] **Step 2: Publish to the dogfood report viewer and verify the downloaded SHA**
- [ ] **Step 3: Add the 2x2 metric table, report URL, and caveats to the PR description**
- [ ] **Step 4: Push any E2E-only fixtures or documentation updates as a new commit**
- [ ] **Step 5: Run final verification and monitor PR CI to green**
