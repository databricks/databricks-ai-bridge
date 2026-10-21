# Agent Development Guide

This is a Microsoft Agent Framework workload hosted by `databricks_agentkit.DurableAgentServer`.
Read [AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md) before changing integration points and [README.md](README.md)
for setup, managed state, request format, and recovery limits.

For CLI workflows, read [.agents/skills/agent-bricks-workflow/SKILL.md](.agents/skills/agent-bricks-workflow/SKILL.md).
If it is absent, use the bundled skill path printed by the CLI.

`agent/agent.py` owns the native agent and its managed history/memory/tools; `agent/tools/` contains local
native tools. `runtime/adapter.py` maps the HTTP input and emits events; `runtime/main.py` registers the
runtime hooks. The optional browser app uses the `read_history` export from `agent.agent`.

Run `agentbricks dev`, `uv run pytest`, and `agentbricks deploy <app-name> --source .`.
Recovery replays a turn, so side-effecting tools must be idempotent. Do not add approval-requiring tools
or user-authenticated bindings until their runtime continuation/error handling is implemented.
