# Agent Development Guide

This project runs your framework or harness with `databricks_agentkit.DurableAgentServer`.

Read [AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md) before changing runtime integration.
[README.md](README.md) contains setup, invocation, and managed-memory examples.
For CLI workflows, read [.agents/skills/agent-bricks-workflow/SKILL.md](.agents/skills/agent-bricks-workflow/SKILL.md).

Replace `run_agent` in `agent/agent.py` with native framework execution. Keep HTTP translation,
and managed store integration in `runtime/adapter.py`; keep server startup in
`runtime/main.py`. Add your framework dependencies to `pyproject.toml`.

Run `uv run pytest` for credential-free tests, `agentbricks dev` for local development, and
`agentbricks --profile <profile> deploy <name> --source .` to deploy.
