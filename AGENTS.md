# AGENTS.md

## Overview

`databricks-ai-bridge` is a monorepo that bridges upstream AI SDKs to Databricks platform services. It provides:

- **`databricks-ai-bridge`** (core) -- shared utilities: Vector Search retriever mixin, Genie client, Lakebase client, OBO credential strategy
- **`databricks-langchain`** -- LangChain/LangGraph integration (chat models, embeddings, vector stores, tools, checkpointing)
- **`databricks-openai`** -- OpenAI SDK integration (client wrappers, agent session, MCP toolkits)
- **`databricks-mcp`** -- MCP client for Databricks-hosted MCP servers (UC functions, Vector Search, Genie, DBSQL)

## Repo Structure

```
src/databricks_ai_bridge/          # Core package
integrations/langchain/            # databricks-langchain (own pyproject.toml, .venv, tests/)
integrations/openai/               # databricks-openai (own pyproject.toml, .venv, tests/)
databricks_mcp/                    # databricks-mcp (own pyproject.toml, .venv, tests/)
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

## Integration Tests

Integration tests hit live Databricks APIs (no mocks) and cover 6 areas: Vector Search, Genie, MCP, FMAPI Tool Calling, Lakebase, and OBO Credentials.

- **Documentation:** [`tests/README.md`](tests/README.md) -- what tests exist, their coverage, how to run them locally, and the motivation behind each area
- **Claude Code skill:** [`.claude/skills/integration-tests.md`](.claude/skills/integration-tests.md) -- principles and patterns for writing new integration tests (trigger with `/integration-tests`)

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
sed -i '' 's/ref: main/ref: <your-bridge-feature-branch>/g' .github/workflows/integration-tests.yml
git add . && git commit -m "Point to <your-bridge-feature-branch> for testing"
git push -u origin test/<your-bridge-feature-branch>

# Trigger and watch
gh workflow run integration-tests.yml --ref test/<your-bridge-feature-branch>
gh run watch
```

Delete the temporary runner branch after tests pass.
