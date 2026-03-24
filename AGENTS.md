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
```

## Integration Tests

Integration tests hit live Databricks APIs (no mocks) and cover 6 areas: Vector Search, Genie, MCP, FMAPI Tool Calling, Lakebase, and OBO Credentials.

- **Documentation:** [`tests/README.md`](tests/README.md) -- what tests exist, their coverage, how to run them locally, and the motivation behind each area
- **Claude Code skill:** [`.claude/skills/integration-tests.md`](.claude/skills/integration-tests.md) -- principles and patterns for writing new integration tests (trigger with `/integration-tests`)

Tests are gated by environment variables (e.g., `RUN_VS_INTEGRATION_TESTS=1`) so they don't run during normal development. CI runs nightly in a private runner repo that injects workspace credentials.
