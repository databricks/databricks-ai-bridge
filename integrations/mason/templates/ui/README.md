# Mason Demo UI Template

This directory is the optional overlay installed by `mason add ui` into a LangGraph Mason agent.
Keeping it beside the agent templates makes the generated source reviewable without mixing UI code
into the base agent template.

## Installed files

- `ui/` contains the zero-build chat client.
- `runtime/ui.py` serves the assets and exposes demo APIs for memory, sessions, and recovery.
- `agent/mason/durability.py` and `agent/mason/recovery.py` implement the checkpointed stop/start
  sequence and heartbeat metadata.
- `agent/tools/long_running.py` provides deterministic `tool_step_1` through `tool_step_4` tools.
- `tests/` contains the UI, durability, and recovery tests copied into the generated project.

## Behavior

`mason add ui` always installs the Start App / Stop App controls. It updates `.env`, `.env.example`,
and `app.yaml` with `MASON_DEMO_STOP_ENABLED=true`. Rerun `mason add ui --refresh` to overwrite the
Mason-managed overlay with the current template; `mason dev` and `mason deploy` do not rewrite source.

The capability indicators are automatic. Streaming and background reflect the runtime contract;
Session reflects checkpoint history; Memory requires `AGENT_MEMORY_STORE`; Durability and Heartbeat
require both a managed Session Store and stop/start control. The transport selector is the only
manual capability choice.

The UI reads local history from the LangGraph checkpoint and managed history from Session Store
items. It reuses the same public `session_id` for chat, HITL resume, transcript lookup, and recovery.
The Databricks Apps `__Host-databricks-app-router` cookie remains a separate sticky-routing concern.

The recovery flow is intentionally a demo, not a production lease implementation: ownership is
last-writer-wins, heartbeats are browser-driven, and stale work is resumed at the first node whose
output was not checkpointed.
