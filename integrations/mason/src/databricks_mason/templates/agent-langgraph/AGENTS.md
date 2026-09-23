# Agent Development Guide

This project is a LangGraph workload hosted by `databricks_mason.DurableAgentServer`.

Read [MASON_CONTRACT.md](MASON_CONTRACT.md) before changing integration points. It owns command
requirements, tool/state/tracing wiring, and recovery. [README.md](README.md) owns setup and client
examples. Keep this file as a development map rather than repeating those rules.

## Commands

```bash
mason dev
uv run pytest
mason --profile <profile> deploy <name> --source .
```

## Code map

| Change | File |
| --- | --- |
| Framework-native agent and `run_agent` | `agent/agent.py` |
| Local tools | `agent/tools/` |
| MCP servers | `agent/mcps.py` |
| Mason `invoke`/`recover` hooks and input/output translation | `runtime/adapter.py` |
| Mason server construction and hook registration | `runtime/main.py` |
| Browser and managed-state routes | `runtime/ui.py` |
| Browser behavior | `ui/app.js` |

Shared adapters come from `databricks_mason.langgraph` and `databricks_mason.runtime`. Consult
their docstrings for API details. Update the shared contract and relevant tests when integration
requirements change.
