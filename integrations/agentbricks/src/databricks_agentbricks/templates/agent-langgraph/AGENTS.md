# Agent Development Guide

This project is a LangGraph workload hosted by `databricks_agentkit.DurableAgentServer`.

Read [AGENTKIT_CONTRACT.md](AGENTKIT_CONTRACT.md) before changing integration points. It owns command
requirements, tool/state/tracing wiring, and recovery. [README.md](README.md) owns setup and client
examples. Keep this file as a development map rather than repeating those rules.

## Commands

```bash
ab dev
uv run pytest
ab --profile <profile> deploy <name> --source .
```

## Code map

| Change | File |
| --- | --- |
| Framework-native agent and `run_agent` | `agent/agent.py` |
| Local tools | `agent/tools/` |
| MCP servers | `agent/mcps.py` |
| Managed runtime `invoke`/`recover` hooks and input/output translation | `runtime/adapter.py` |
| `DurableAgentServer` construction and hook registration | `runtime/main.py` |
| Browser and managed-state routes | `runtime/ui.py` |
| Browser behavior | `ui/app.js` |

Shared adapters come from `databricks_agentkit.langgraph` and `databricks_agentkit.runtime`. Consult
their docstrings for API details. Update the shared contract and relevant tests when integration
requirements change.
