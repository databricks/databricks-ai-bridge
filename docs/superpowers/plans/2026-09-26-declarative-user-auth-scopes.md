# Declarative User Auth Scopes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let code-first Agent Bricks tools declare request-user authentication and extra Databricks Apps OAuth scopes in `agent.toml`, while unioning those scopes with managed-tool inference.

**Architecture:** Add one immutable `UserAuthConfig` parser in the runtime manifest module and reuse it from the comment-preserving CLI project model and invocation policy. Deployment keeps the existing conservative Apps reconciliation, changing only the request-auth trigger and required-scope source. A focused deployed matrix proves LangGraph/OpenAI parity, explicit-only and explicit-plus-inferred scopes, OBO success, App-principal denial, and cleanup.

**Tech Stack:** Python 3.10+, `tomllib`/`tomli`, `tomlkit`, Click, Databricks SDK/Apps, pytest, uv, Ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-26-declarative-user-auth-scopes-design.md`

## Global Constraints

- Existing managed `auth = "user"` manifests remain valid without `[auth.user]`.
- Final required scopes are the deduplicated union of managed inference and `additional_api_scopes`; existing App scopes remain preserved.
- Custom Python tools remain code-first; do not add synthetic Python manifest bindings or source-code scope inference.
- Explicit scope names are structurally validated but not allowlisted client-side.
- Non-empty `additional_api_scopes` requires `required = true`.
- Declarative request-user auth is supported only by `[agent].server = "agentbricks"`.
- Runtime consumers continue to obtain user clients through the request-bound `workspace_client_for("user")` seam.
- Deployed e2e runtime dependencies must resolve from a pushed git SHA.
- Do not log credentials, forwarded tokens, private workspace URLs, or private asset identifiers in published evidence.

---

### Task 1: Shared manifest parser and CLI project model

**Files:**
- Modify: `integrations/agentbricks/src/databricks_agentkit/runtime/tool_manifest.py`
- Modify: `integrations/agentbricks/src/databricks_agentbricks/agent_project.py`
- Test: `integrations/agentbricks/tests/unit_tests/agent_project_test.py`

**Interfaces:**
- Produces: `UserAuthConfig(required: bool = False, additional_api_scopes: tuple[str, ...] = ())`
- Produces: `parse_user_auth(document: Mapping[str, Any]) -> UserAuthConfig`
- Produces: `load_user_auth() -> UserAuthConfig`
- Produces: `AgentProject.user_auth: UserAuthConfig`

- [ ] **Step 1: Write failing valid-parser and reader-parity tests**

Add table-driven tests that load this manifest through both `AgentProject.load()` and runtime parsing:

```toml
schema_version = 1
[agent]
framework = "langgraph"
server = "agentbricks"
[auth.user]
required = true
additional_api_scopes = ["sql", "ai-gateway", "sql"]
```

Assert `UserAuthConfig(required=True, additional_api_scopes=("sql", "ai-gateway"))`; assert the absent block returns `UserAuthConfig()`; load/write/reload and assert a `# keep auth comment` line survives.

- [ ] **Step 2: Write failing invalid-parser parity tests**

Parametrize CLI/runtime readers over: non-table `auth`, non-table `auth.user`, unknown keys, non-bool `required`, non-array scopes, non-string scope, empty scope, leading/trailing whitespace, `\n`/control characters, and non-empty scopes with `required=false`. Assert CLI raises `AgentCliError`, runtime raises `ToolManifestError`, and both messages contain the same field-specific fragment.

- [ ] **Step 3: Run parser tests and observe the expected failures**

Run:

```bash
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks pytest integrations/agentbricks/tests/unit_tests/agent_project_test.py -q
```

Expected: new tests fail because `UserAuthConfig`, `parse_user_auth`, `load_user_auth`, and `AgentProject.user_auth` do not exist.

- [ ] **Step 4: Implement the immutable shared parser**

In `tool_manifest.py`, add:

```python
@dataclass(frozen=True)
class UserAuthConfig:
    required: bool = False
    additional_api_scopes: tuple[str, ...] = ()


def parse_user_auth(document: Mapping[str, Any]) -> UserAuthConfig:
    raw_auth = document.get("auth")
    if raw_auth is None:
        return UserAuthConfig()
    if not isinstance(raw_auth, Mapping):
        raise ToolManifestError("agent.toml auth must be a table.")
    raw_user = raw_auth.get("user")
    if raw_user is None:
        return UserAuthConfig()
    if not isinstance(raw_user, Mapping):
        raise ToolManifestError("agent.toml auth.user must be a table.")
    unexpected = set(raw_user) - {"required", "additional_api_scopes"}
    if unexpected:
        fields = ", ".join(sorted(unexpected))
        raise ToolManifestError(f"agent.toml auth.user has unsupported fields: {fields}.")
    required = raw_user.get("required", False)
    if not isinstance(required, bool):
        raise ToolManifestError("agent.toml auth.user.required must be a boolean.")
    raw_scopes = raw_user.get("additional_api_scopes", [])
    if not isinstance(raw_scopes, list):
        raise ToolManifestError("agent.toml auth.user.additional_api_scopes must be an array.")
    scopes: list[str] = []
    for scope in raw_scopes:
        if not isinstance(scope, str) or not scope or scope != scope.strip():
            raise ToolManifestError("agent.toml auth.user.additional_api_scopes must contain non-empty scope names without surrounding whitespace.")
        if any(ord(char) < 32 or ord(char) == 127 for char in scope):
            raise ToolManifestError("agent.toml auth.user.additional_api_scopes cannot contain control characters.")
        scopes.append(scope)
    deduplicated = tuple(dict.fromkeys(scopes))
    if deduplicated and not required:
        raise ToolManifestError("agent.toml auth.user.additional_api_scopes requires auth.user.required = true.")
    return UserAuthConfig(required=required, additional_api_scopes=deduplicated)


def load_user_auth() -> UserAuthConfig:
    path = project_root() / "agent.toml"
    with path.open("rb") as source:
        return parse_user_auth(tomllib.load(source))
```

The parser permits only `required` and `additional_api_scopes`, rejects malformed scope strings (`not value`, `value != value.strip()`, or `any(ord(char) < 32 or ord(char) == 127 for char in value)`), and deduplicates with `tuple(dict.fromkeys(scopes))`.

- [ ] **Step 5: Thread the parsed config through `AgentProject`**

Add `user_auth: tool_manifest.UserAuthConfig = tool_manifest.UserAuthConfig()` to the constructor, set `self.user_auth`, call `tool_manifest.parse_user_auth(document)` in `load()`, translate `ToolManifestError` to `AgentCliError`, and pass the value to the constructor. `create()` continues to use the default and no write helper is added.

- [ ] **Step 6: Run the parser tests to green**

Run the command from Step 3. Expected: all `agent_project_test.py` tests pass.

- [ ] **Step 7: Commit the parser slice**

```bash
git add integrations/agentbricks/src/databricks_agentkit/runtime/tool_manifest.py integrations/agentbricks/src/databricks_agentbricks/agent_project.py integrations/agentbricks/tests/unit_tests/agent_project_test.py
git commit -m "feat: parse declarative user auth scopes"
```

### Task 2: Runtime invocation policy

**Files:**
- Modify: `integrations/agentbricks/src/databricks_agentkit/runtime/auth.py`
- Test: `integrations/agentbricks/tests/unit_tests/runtime_app_test.py`
- Test: `integrations/agentbricks/tests/unit_tests/request_bound_app_test.py`

**Interfaces:**
- Consumes: `load_user_auth() -> UserAuthConfig` and existing `load_tools()`
- Produces: `InvocationAuthPolicy(user_tools: tuple[str, ...] = (), user_required: bool = False)`

- [ ] **Step 1: Write failing policy tests**

Assert `InvocationAuthPolicy(user_required=True).requires_user` is true, the all-default policy is false, and a manifest with declarative auth plus one user-auth managed binding yields:

```python
assert app.auth_policy.user_tools == ("search",)
assert app.auth_policy.user_required is True
assert app.auth_policy.requires_user is True
```

Also add a custom-tool-only manifest test with no `[[tools]]` and assert request headers are required.

- [ ] **Step 2: Run the focused policy tests and observe failures**

```bash
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks pytest integrations/agentbricks/tests/unit_tests/runtime_app_test.py integrations/agentbricks/tests/unit_tests/request_bound_app_test.py -q
```

Expected: constructor/field assertions fail because `user_required` is not defined.

- [ ] **Step 3: Implement `user_required` and single-snapshot parsing**

Extend the dataclass with `user_required: bool = False`, change `requires_user` to `return self.user_required or bool(self.user_tools)`, and have `from_manifest()` load one TOML document, call `parse_user_auth(document)`, parse tools from that same document via a shared document-level helper, and return keyword arguments:

```python
return cls(
    user_tools=tuple(tool.id for tool in tools if tool.auth == "user"),
    user_required=user_auth.required,
)
```

Keep positional compatibility by leaving `user_tools` as the first field.

- [ ] **Step 4: Run focused policy tests to green**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Commit runtime policy**

```bash
git add integrations/agentbricks/src/databricks_agentkit/runtime/auth.py integrations/agentbricks/tests/unit_tests/runtime_app_test.py integrations/agentbricks/tests/unit_tests/request_bound_app_test.py
git commit -m "feat: require user credentials for code-first tools"
```

### Task 3: Deployment trigger, scope union, and CLI wording

**Files:**
- Modify: `integrations/agentbricks/src/databricks_agentbricks/cli/app_auth.py`
- Modify: `integrations/agentbricks/src/databricks_agentbricks/cli/deploy.py`
- Modify: `integrations/agentbricks/cli.md`
- Test: `integrations/agentbricks/tests/unit_tests/deploy_auth_test.py`

**Interfaces:**
- Consumes: `AgentProject.user_auth`
- Produces: unchanged `requires_user_auth(project) -> bool`
- Produces: unchanged `required_user_api_scopes(project) -> set[str]`

- [ ] **Step 1: Write failing deployment contract tests**

Add fixtures that insert `[auth.user]` directly into a project manifest and reload it. Assert:

```python
assert requires_user_auth(explicit_only) is True
assert required_user_api_scopes(explicit_only) == {"sql"}
assert required_user_api_scopes(combined) == {"sql", "ai-gateway"}
```

Cover duplicate explicit/inferred `sql`, declarative auth on a custom server (error), and a deploy invocation where `--allow-user-scope-update` is accepted without managed tools.

- [ ] **Step 2: Run deploy-auth tests and observe failures**

```bash
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks pytest integrations/agentbricks/tests/unit_tests/deploy_auth_test.py -q
```

Expected: explicit-only cases return false/empty or reject the CLI flag.

- [ ] **Step 3: Implement trigger and union**

Initialize `managed_user_auth = any(tool.auth == "user" for tool in managed)` and `user_auth = project.user_auth.required or managed_user_auth`; validate the Agent Bricks server whenever `user_auth` is true; keep the explicit managed-binding identity check when `managed_user_auth` is true. Initialize scope reconciliation from `set(project.user_auth.additional_api_scopes)` before adding managed inference.

- [ ] **Step 4: Update CLI copy and reference docs**

Use declarative wording:

```text
Allow Agent Bricks to add missing user API scopes to an existing App for declarative user auth. Once added, later deploys do not need this flag.
```

Change the no-auth error to require request-user auth in `agent.toml`, not specifically a managed tool. Mirror the exact option help in `cli.md`.

- [ ] **Step 5: Run deploy-auth and CLI help tests to green**

```bash
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks pytest integrations/agentbricks/tests/unit_tests/deploy_auth_test.py integrations/agentbricks/tests/unit_tests/cli_ergonomics_test.py -q
```

- [ ] **Step 6: Commit deployment behavior**

```bash
git add integrations/agentbricks/src/databricks_agentbricks/cli/app_auth.py integrations/agentbricks/src/databricks_agentbricks/cli/deploy.py integrations/agentbricks/tests/unit_tests/deploy_auth_test.py integrations/agentbricks/cli.md
git commit -m "feat: reconcile declared user API scopes"
```

### Task 4: User documentation

**Files:**
- Modify: `integrations/agentbricks/README.md`

**Interfaces:**
- Documents: `[auth.user]`, union semantics, validation boundary, custom-tool client seam, and deploy-update flag.

- [ ] **Step 1: Add a complete manifest example and behavior notes**

Document:

```toml
[auth.user]
required = true
additional_api_scopes = ["sql"]
```

Explain that managed bindings continue to infer scopes, additions are unioned, Apps validates supported scope names, custom tools use the request-bound `workspace_client_for("user")`, and existing Apps need `--allow-user-scope-update` only when scopes are missing.

- [ ] **Step 2: Run doc-facing help smoke test**

```bash
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks agentbricks deploy --help
```

Assert the printed option copy matches `cli.md`.

- [ ] **Step 3: Commit docs**

```bash
git add integrations/agentbricks/README.md
git commit -m "docs: describe declarative user auth scopes"
```

### Task 5: Focused deployed e2e matrix

**Files:**
- Create: `integrations/agentbricks/tests/e2e/auth_scope_matrix.py`
- Modify: `integrations/agentbricks/tests/e2e/README.md`

**Interfaces:**
- Inputs: workspace profile, OAuth App profile, pushed repository URL/SHA, wheel, output directory, SQL warehouse/catalog/schema
- Outputs: redacted `evidence.json`, `commands.log`, per-case deployment/invocation logs, cleanup state, exit 0 only when every matrix assertion passes

- [ ] **Step 1: Add offline matrix-contract tests inside the harness**

Implement `--verify-evidence PATH` to assert exactly four case rows (`langgraph/openai` × `explicit/combined`), configured/effective scope equality, `sql` in every case, `ai-gateway` only in combined cases, OBO marker present, App-principal marker absent, invocation completed, and cleanup verified.

- [ ] **Step 2: Author generated projects for both framework harnesses**

For each framework, initialize a project, write `[auth.user]`, and replace the sample code-first tool with a request-bound SQL Statements API tool using `workspace_client_for("user")`. Combined cases also add `system.ai.web_search` with `auth = "user"`. Pin `databricks-agentbricks` to the pushed git SHA in each project `pyproject.toml`.

- [ ] **Step 3: Provision a user-only SQL asset and negative control**

Create a unique table/view marker readable by the invoking user; revoke/avoid grants to the generated App service principals. Before invoking the agent, call the same SQL operation as each App principal and record a permission denial without serializing principal IDs or tokens.

- [ ] **Step 4: Deploy and monitor every case**

Deploy with `--allow-user-scope-update`; poll build/compute at one-minute intervals; read App metadata and assert exact configured/effective scopes; invoke the deployed API with OAuth; assert the SQL marker and, for combined cases, web-search evidence.

- [ ] **Step 5: Clean resources and verify cleanup**

Delete temporary Apps and SQL assets in `finally`, then query for absence and record booleans in evidence. Keep resources only behind an explicit debugging flag.

- [ ] **Step 6: Run Ruff on the harness and commit**

```bash
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks ruff check integrations/agentbricks/tests/e2e/auth_scope_matrix.py
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks ruff format --check integrations/agentbricks/tests/e2e/auth_scope_matrix.py
git add integrations/agentbricks/tests/e2e/auth_scope_matrix.py integrations/agentbricks/tests/e2e/README.md
git commit -m "test: add declarative auth scope e2e matrix"
```

### Task 6: Verification, live evidence, report, and PR

**Files:**
- Create outside git: dated raw evidence directory under `/tmp`
- Create outside git: dated HTML metric report using the `eval-report-html` skill
- Modify remotely: standalone GitHub PR description

**Interfaces:**
- Consumes: all preceding code/tests and a pushed source SHA
- Produces: green focused/full verification, published HTML report URL, live endpoint proof, and attached PR

- [ ] **Step 1: Run focused and package-wide verification**

```bash
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks pytest integrations/agentbricks/tests/unit_tests/agent_project_test.py integrations/agentbricks/tests/unit_tests/deploy_auth_test.py integrations/agentbricks/tests/unit_tests/runtime_app_test.py integrations/agentbricks/tests/unit_tests/request_bound_app_test.py -q
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks ruff check integrations/agentbricks
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks ruff format --check integrations/agentbricks
UV_CACHE_DIR=/tmp/mason-auth-scope-uv-cache uv run --project integrations/agentbricks ty check integrations/agentbricks/src
```

Run the full unit suite with a bounded timeout; record the known baseline `genie_adapters_test.py::test_real_native_ask_invokes_fixed_space[langgraph-None]` anomaly separately if it recurs.

- [ ] **Step 2: Run pre-PR verification and inspect the diff**

Follow `pre-pr-verifier`: verify no credentials/private identifiers, confirm all required files are tracked, inspect `git diff origin/main...HEAD`, and run tests matched to every production change.

- [ ] **Step 3: Commit remaining plan/evidence-support changes and push**

```bash
git add docs/superpowers/plans/2026-09-26-declarative-user-auth-scopes.md
git commit -m "docs: plan declarative user auth scopes"
git push -u origin feat/mason-declarative-auth-scopes
```

- [ ] **Step 4: Build from and deploy the pushed SHA**

Build a wheel, compute SHA-256, run `auth_scope_matrix.py` with the pushed SHA, and monitor until the four-case matrix succeeds. Re-run `--verify-evidence` on the final JSON.

- [ ] **Step 5: Generate and publish the HTML metric report**

Use `eval-report-html` with the final evidence. Include source/wheel hashes, exact matrix axes and scope assertions, positive/negative identity controls, durations, terminal states, sanitized logs, cleanup verification, and exact verification commands. Publish to the standard dogfood metric viewer and verify both the content SHA and viewer route.

- [ ] **Step 6: Create one clean standalone PR**

Follow `pr-structuring`. The PR description must link the PRD/design, summarize the contract, list exact local/live checks, link the HTML report, include live endpoint/curl proof with private details redacted, and call out the full-suite baseline anomaly if present.

- [ ] **Step 7: Attach and babysit the PR to green**

Attach the PR URL to this task, invoke `pr-babysitter`, classify/fix PR-caused failures, rerun flaky/pre-existing failures only with evidence, update the report/description after any code push, and stop only when required checks and the deployed matrix are green.
