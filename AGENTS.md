# AGENTS.md

## Overview

`databricks-ai-bridge` is a monorepo that bridges upstream AI SDKs to Databricks platform services. It provides:

- **`databricks-ai-bridge`** (core) -- shared utilities: Vector Search retriever mixin, Genie client, Lakebase client, OBO credential strategy
- **`databricks-langchain`** -- LangChain/LangGraph integration (chat models, embeddings, vector stores, tools, checkpointing)
- **`databricks-openai`** -- OpenAI SDK integration (client wrappers, agent session, MCP toolkits)
- **`databricks-mcp`** -- MCP client for Databricks-hosted MCP servers (UC functions, Vector Search, Genie, DBSQL)
- **`databricks-agentbricks`** -- Python distribution for the Agent Bricks CLI (`agentbricks`), AgentKit SDK (`databricks_agentkit`), and agent runtime. See [`integrations/agentbricks/README.md`](integrations/agentbricks/README.md) for the dev loop

## Repo Structure

```
src/databricks_ai_bridge/          # Core package
integrations/langchain/            # databricks-langchain (own pyproject.toml, .venv, tests/)
integrations/openai/               # databricks-openai (own pyproject.toml, .venv, tests/)
databricks_mcp/                    # databricks-mcp (own pyproject.toml, .venv, tests/)
integrations/agentbricks/          # Agent Bricks CLI and AgentKit SDK (databricks-agentbricks distribution)
tests/                             # Core package tests + integration tests
```

Each package has its own `pyproject.toml` and virtual environment. Use `uv` for dependency management.

## Development

```bash
# Install a package in dev mode (from its directory)
cd integrations/langchain && uv sync --all-extras

# Run unit tests
python -m pytest tests/unit_tests/ -v

# Lint
ruff check && ruff format

# Type check
ty check
```

## Software Factory

For factory-assigned work, start with [factory-author](.claude/skills/factory-author/SKILL.md),
then use [factory-evidence](.claude/skills/factory-evidence/SKILL.md) and
[factory-pr](.claude/skills/factory-pr/SKILL.md) as needed. Use
[factory-session](.claude/skills/factory-session/SKILL.md) only to launch or locate
a session. These repository files are the source of truth; `.agents/skills/`
links to the same files for Codex. The factory reads them from the task checkout,
so a new checkout of `main` picks up updates after they merge.

## Code Review

AgentBricksReviewer (`.github/workflows/agentbricks-reviewer.yml`) reviews all eligible PRs in this repository and posts its review as a PR comment; comment `/review` to re-run it. Its process lives in [`.claude/skills/agentbricks-review/SKILL.md`](.claude/skills/agentbricks-review/SKILL.md), which also runs locally: `/agentbricks-review [PR number or URL]`, or with no argument to review the current branch against `origin/main`. Reviews always load root [`REVIEW.md`](REVIEW.md), then auto-discover additional `REVIEW.md` files scoped to changed paths. CI uses these files from the default branch, so edits take effect after they merge.

Pushes to eligible PRs also request a new review, with at most three automatic push-review attempts per PR by default (`AI_REVIEW_MAX_PUSH_REVIEWS`; set `0` to disable). Failed attempts count toward the limit. A bot comment tracks the count; the initial review and manually requested `/review` runs do not consume this budget. Pushing still dismisses stale approvals after the limit is reached.

When the `AI_REVIEW_AUTO_APPROVE` repo variable is `true`, the workflow also approves PRs that pass [`.github/scripts/agentbricks_auto_approve.py`](.github/scripts/agentbricks_auto_approve.py): a high review score, no blocking, security, or uncertain findings, a small diff that only changes `integrations/agentbricks/` or Markdown/reST docs and doesn't touch CI, review rules, dependencies, or credential code, and an author with write access. The approval comes from `github-actions[bot]`. Pushing new commits dismisses that approval.

Reviews separately assess change safety (0–100, higher is safer to automate) based on affected components, public behavior, sensitive operations, tests, and reversibility. CI applies minimum risk levels for sensitive paths/components and large changes, then assigns one of `risk:low`, `risk:medium` (moderate), `risk:high`, `risk:critical`, or `risk:unrated` even when automatic approval is disabled. Only low change risk (90–100) can pass automatic approval, alongside the existing review-score and eligibility checks. Moderate is 70–89; high is 40–69; critical is 0–39. Unavailable or invalid assessments are unrated and cannot be auto-approved. Pushing clears labels for the old head until the new head is reviewed.

## Integration Tests

Integration tests hit live Databricks APIs (no mocks). See [`tests/README.md`](tests/README.md) for what tests exist, their coverage, and how to run them. See [`.claude/skills/integration-tests.md`](.claude/skills/integration-tests.md) for principles and patterns when writing new tests (trigger with `/integration-tests`).

Tests are gated by environment variables (e.g., `RUN_VS_INTEGRATION_TESTS=1`) so they don't run during normal development. CI runs nightly in a private runner repo that injects workspace credentials.

Running integration tests is **optional** and separate from PR CI. After creating a PR, ask the user if they want to trigger integration tests. If so, the process is:

```bash
# Switch to account with runner repo access
gh auth switch --user <your-runner-repo-username>

# Clone runner repo and create a branch pointing to the PR branch
gh repo clone databricks-eng/ai-oss-integration-tests-runner /tmp/runner-repo
cd /tmp/runner-repo
git checkout -b test/<your-bridge-feature-branch>

# Point the workflow at the PR branch instead of main
uv run python -c "
import pathlib
f = pathlib.Path('.github/workflows/integration-tests.yml')
f.write_text(f.read_text().replace('ref: main', 'ref: <your-bridge-feature-branch>'))
"
git add . && git commit -m "Point to <your-bridge-feature-branch> for testing"
git push -u origin test/<your-bridge-feature-branch>

# Trigger and watch
gh workflow run integration-tests.yml --ref test/<your-bridge-feature-branch>
gh run watch

# Cleanup: delete the temporary runner branch after tests pass
git push origin --delete test/<your-bridge-feature-branch>
```
